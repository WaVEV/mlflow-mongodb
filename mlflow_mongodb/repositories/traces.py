"""Persistence operations for traces, spans, and assessments."""

from collections.abc import Mapping
from itertools import islice
from typing import Any

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.constant import SpansLocation, TraceMetadataKey, TraceTagKey
from pymongo import ASCENDING, ReplaceOne, ReturnDocument, UpdateOne
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.repositories._helpers import (
    build_merge_array_expression,
    build_remove_array_element_update,
    translate_database_errors,
)
from mlflow_mongodb.repositories._trace_metrics import (
    build_trace_metrics_pipeline,
    to_metric_data_points,
)
from mlflow_mongodb.repositories.errors import (
    RepositoryPersistenceError,
    TraceNotFoundError,
    TraceWriteConflictError,
)
from mlflow_mongodb.repositories.types import SpanRecord, SpanSummaryRecord, TraceRecord
from mlflow_mongodb.retry import retry_on_exception
from mlflow_mongodb.settings import MongoDBSettings


class TraceRepository:
    """Own trace metadata and the separate span and assessment collections."""

    _READ_BATCH_SIZE = 500

    @translate_database_errors
    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        self._collection = database[self._settings.traces_collection_name]
        # _id is the supplied trace ID and already has a unique index.
        self._collection.create_index(
            [("experiment_id", ASCENDING), ("request_time", ASCENDING), ("_id", ASCENDING)],
            name="traces_experiment_request_time_id",
        )
        self._collection.create_index(
            [
                ("experiment_id", ASCENDING),
                ("run_ids", ASCENDING),
                ("request_time", ASCENDING),
            ],
            name="traces_experiment_run_ids_request_time",
        )
        self._collection.create_index(
            [("linked_prompts.name", ASCENDING), ("linked_prompts.version", ASCENDING)],
            name="traces_linked_prompts_name_version",
        )
        self._spans_collection = database[self._settings.spans_collection_name]
        self._spans_collection.create_index(
            [("trace_id", ASCENDING), ("span_id", ASCENDING)],
            unique=True,
            name="spans_trace_span_unique",
        )
        self._spans_collection.create_index(
            [("trace_id", ASCENDING), ("start_time_ns", ASCENDING)],
            name="spans_trace_start_time",
        )
        self._assessments_collection = database[self._settings.assessments_collection_name]
        self._assessments_collection.create_index(
            [("trace_id", ASCENDING), ("assessment_id", ASCENDING)],
            unique=True,
            name="assessments_trace_assessment_unique",
        )
        self._assessments_collection.create_index(
            [("experiment_id", ASCENDING), ("assessment_id", ASCENDING)],
            name="assessments_experiment_assessment",
        )

    @translate_database_errors
    def start_trace(
        self,
        *,
        trace_id: str,
        experiment_id: str,
        request_time: int,
        state: str,
        execution_duration: int | None,
        client_request_id: str | None,
        request_preview: str | None,
        response_preview: str | None,
        tags: Mapping[str, str],
        trace_metadata: Mapping[str, str],
        assessments: list[dict[str, Any]],
        run_ids: list[str] | None = None,
        span_stats: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> TraceRecord:
        unique_run_ids = list(dict.fromkeys(run_ids or []))
        fields = {
            "experiment_id": {"$literal": experiment_id},
            "request_time": {"$literal": request_time},
            "state": {"$literal": state},
            "execution_duration": {"$literal": execution_duration},
            "client_request_id": {"$literal": client_request_id},
            "run_ids": {
                "$setUnion": [
                    {"$ifNull": ["$run_ids", []]},
                    {"$literal": unique_run_ids},
                ]
            },
            # Authoritative metadata may arrive after spans. This does not close
            # the trace to further span delivery, even when its state is final.
            "trace_info_finalized": True,
            "authoritative_metadata_keys": {
                "$setUnion": [
                    {"$ifNull": ["$authoritative_metadata_keys", []]},
                    {"$literal": list(trace_metadata)},
                ]
            },
            "tags": build_merge_array_expression(
                "$tags", [{"k": k, "v": v} for k, v in tags.items()], "k"
            ),
            "trace_metadata": build_merge_array_expression(
                "$trace_metadata",
                [{"k": k, "v": v} for k, v in trace_metadata.items()],
                "k",
            ),
        }
        # Write the native stats and serialized metadata together. Omitted stats
        # preserve the existing expected count during partial metadata updates.
        if span_stats is not None:
            fields["span_stats"] = {"$literal": dict(span_stats)}
        for key, value in (metrics or {}).items():
            fields[key] = {"$literal": value}
        for field, value in (
            ("request_preview", request_preview),
            ("response_preview", response_preview),
        ):
            fields[field] = (
                {"$literal": value} if value is not None else {"$ifNull": [f"${field}", None]}
            )

        document = self._upsert_trace(trace_id, experiment_id, [{"$set": fields}])
        self._upsert_assessments(
            trace_id=trace_id,
            experiment_id=experiment_id,
            assessments=assessments,
        )
        return self._trace_record(document)

    @retry_on_exception(
        DuplicateKeyError,
        attempts=3,
        on_exhausted=lambda _: RepositoryPersistenceError(
            "Trace upsert conflicts persisted after 3 attempts."
        ),
    )
    def _upsert_trace(self, trace_id: str, experiment_id: str, update) -> dict[str, Any]:
        # Ownership remains in the filter; failures do not trigger diagnostic reads.
        return self._collection.find_one_and_update(
            {"_id": trace_id, "experiment_id": experiment_id},
            update,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    @translate_database_errors
    def ensure_traces(self, *, experiment_id: str, request_times: Mapping[str, int]) -> None:
        """Create missing trace placeholders through a single bulk write.

        Existing traces retain their metadata. Unordered writes can partially succeed.
        """
        if not request_times:
            return
        operations = [
            UpdateOne(
                {"_id": trace_id, "experiment_id": experiment_id},
                {"$setOnInsert": self._trace_placeholder(request_times[trace_id])},
                upsert=True,
            )
            for trace_id in request_times
        ]
        self._collection.bulk_write(operations, ordered=False)

    @staticmethod
    def _trace_placeholder(request_time: int) -> dict[str, Any]:
        return {
            "request_time": request_time,
            "state": TraceState.IN_PROGRESS.value,
            "execution_duration": None,
            "client_request_id": None,
            "request_preview": None,
            "response_preview": None,
            "trace_info_finalized": False,
            "authoritative_metadata_keys": [],
            "run_ids": [],
            "tags": [{"k": TraceTagKey.SPANS_LOCATION, "v": SpansLocation.TRACKING_STORE.value}],
            "trace_metadata": [],
        }

    @translate_database_errors
    def _upsert_assessments(
        self,
        *,
        trace_id: str,
        experiment_id: str,
        assessments: list[dict[str, Any]],
    ) -> None:
        """Replace supplied assessments in the separate assessment collection."""
        if not assessments:
            return

        assessments_by_id = {assessment["assessment_id"]: assessment for assessment in assessments}
        operations = []
        for assessment_id, assessment in assessments_by_id.items():
            content = dict(assessment)
            value_json = content.pop("_metric_value_json")
            content["trace_id"] = trace_id
            operations.append(
                ReplaceOne(
                    {"trace_id": trace_id, "assessment_id": assessment_id},
                    {
                        "trace_id": trace_id,
                        "experiment_id": experiment_id,
                        "assessment_id": assessment_id,
                        "value_json": value_json,
                        "content": content,
                    },
                    upsert=True,
                )
            )
        self._assessments_collection.bulk_write(operations, ordered=True)

    @translate_database_errors
    def _get_assessments(self, trace_id: str) -> tuple[dict[str, Any], ...]:
        cursor = self._assessments_collection.find(
            {"trace_id": trace_id},
            {"_id": 0, "content": 1},
        ).sort([("assessment_id", ASCENDING)])
        return tuple(document["content"] for document in cursor)

    def _trace_record(self, document: dict[str, Any]) -> TraceRecord:
        return TraceRecord.from_document(
            document,
            assessments=self._get_assessments(document["_id"]),
        )

    @translate_database_errors
    def log_spans(self, documents: list[dict[str, Any]]) -> None:
        """Insert immutable spans; delivery of an existing identity is a no-op."""
        if not documents:
            return
        operations = [
            UpdateOne(
                {"trace_id": document["trace_id"], "span_id": document["span_id"]},
                {"$setOnInsert": document},
                upsert=True,
            )
            for document in documents
        ]
        self._spans_collection.bulk_write(operations, ordered=False)

    @translate_database_errors
    def span_summary_snapshot(
        self, *, trace_id: str, experiment_id: str
    ) -> tuple[int, list[SpanSummaryRecord]]:
        # Increment AFTER span writes and BEFORE reading them. A later writer
        # invalidates earlier snapshots and then recomputes from persisted spans.
        trace = self._collection.find_one_and_update(
            {"_id": trace_id, "experiment_id": experiment_id},
            {"$inc": {"span_revision": 1}},
            projection={"span_revision": 1},
            return_document=ReturnDocument.AFTER,
        )
        if trace is None:
            raise TraceNotFoundError(trace_id)
        cursor = self._spans_collection.find(
            {"trace_id": trace_id},
            {
                "_id": 0,
                "span_id": 1,
                "parent_span_id": 1,
                "status": 1,
                "start_time_ns": 1,
                "end_time_ns": 1,
                "token_usage": 1,
                "cost": 1,
                "trace_fields": 1,
            },
        ).sort([("start_time_ns", ASCENDING), ("span_id", ASCENDING)])
        with cursor:
            return trace["span_revision"], [SpanSummaryRecord.from_document(d) for d in cursor]

    @translate_database_errors
    def update_span_summary(
        self, *, trace_id: str, experiment_id: str, revision: int, summary: dict[str, Any]
    ) -> None:
        fields = {
            "state": {
                "$cond": [
                    {
                        "$in": [
                            "$state",
                            [TraceState.IN_PROGRESS.value, TraceState.STATE_UNSPECIFIED.value],
                        ]
                    },
                    {"$literal": summary["state"]},
                    "$state",
                ]
            },
            # Existing tags/metadata include start_trace and explicit tag writes.
            "tags": self._merge_missing_records("$tags", summary["tags"]),
            "trace_metadata": self._merge_missing_records("$trace_metadata", summary["metadata"]),
        }
        for key in ("request_time", "execution_duration"):
            fields[key] = {
                "$cond": [
                    {"$ifNull": ["$trace_info_finalized", False]},
                    f"${key}",
                    {"$literal": summary[key]},
                ]
            }
        for key in ("request_preview", "response_preview"):
            fields[key] = {"$ifNull": [f"${key}", {"$literal": summary[key]}]}

        aggregate_records = []
        for field, metadata_key in (
            ("token_usage", TraceMetadataKey.TOKEN_USAGE),
            ("cost", TraceMetadataKey.COST),
        ):
            if summary[field] is not None:
                fields[field] = {
                    "$cond": [
                        {"$in": [metadata_key, {"$ifNull": ["$authoritative_metadata_keys", []]}]},
                        f"${field}",
                        {"$literal": summary[field]},
                    ]
                }
                aggregate_records.append(
                    {
                        "k": metadata_key,
                        "v": summary["aggregate_metadata"][metadata_key],
                    }
                )
        metadata_update = build_merge_array_expression(
            "$trace_metadata",
            aggregate_records,
            "k",
            protected_keys="$authoritative_metadata_keys",
        )
        result = self._collection.update_one(
            {"_id": trace_id, "experiment_id": experiment_id, "span_revision": revision},
            [{"$set": fields}, {"$set": {"trace_metadata": metadata_update}}],
        )
        if not result.matched_count:
            raise TraceWriteConflictError(trace_id)

    @classmethod
    def _merge_missing_records(cls, field: str, values: Mapping[str, str]) -> dict:
        return build_merge_array_expression(
            field,
            [{"k": k, "v": v} for k, v in values.items()],
            "k",
            protected_keys=f"{field}.k",
        )

    @translate_database_errors
    def get_trace_info(self, trace_id: str) -> TraceRecord:
        document = self._collection.find_one({"_id": trace_id})
        if document is None:
            raise TraceNotFoundError(trace_id)
        return self._trace_record(document)

    @translate_database_errors
    def batch_get_trace_infos(
        self, trace_ids: list[str], *, experiment_ids: list[str] | None = None
    ) -> list[TraceRecord]:
        """Read metadata and assessments without loading span payloads."""
        return [
            record
            for record, _ in self._batch_get_trace_records(
                trace_ids, experiment_ids=experiment_ids, include_spans=False
            )
        ]

    @translate_database_errors
    def batch_get_traces(
        self, trace_ids: list[str], *, experiment_ids: list[str] | None = None
    ) -> list[tuple[TraceRecord, list[SpanRecord]]]:
        """Join scoped trace metadata, assessments, and spans in each read batch."""
        return self._batch_get_trace_records(
            trace_ids, experiment_ids=experiment_ids, include_spans=True
        )

    def _batch_get_trace_records(
        self,
        trace_ids: list[str],
        *,
        experiment_ids: list[str] | None,
        include_spans: bool,
    ) -> list[tuple[TraceRecord, list[SpanRecord]]]:
        if not trace_ids or experiment_ids == []:
            return []

        trace_id_order = {trace_id: index for index, trace_id in enumerate(trace_ids)}
        unique_trace_ids = list(trace_id_order)
        records = []
        for start in range(0, len(unique_trace_ids), self._READ_BATCH_SIZE):
            query = {"_id": {"$in": unique_trace_ids[start : start + self._READ_BATCH_SIZE]}}
            if experiment_ids is not None:
                query["experiment_id"] = {"$in": experiment_ids}
            pipeline = [
                {"$match": query},
                {
                    "$lookup": {
                        "from": self._assessments_collection.name,
                        "localField": "_id",
                        "foreignField": "trace_id",
                        "pipeline": [
                            {"$sort": {"assessment_id": ASCENDING}},
                            {"$project": {"_id": 0, "content": 1}},
                        ],
                        "as": "_assessments",
                    }
                },
            ]
            if include_spans:
                pipeline.append(
                    {
                        "$lookup": {
                            "from": self._spans_collection.name,
                            "localField": "_id",
                            "foreignField": "trace_id",
                            "as": "_spans",
                        }
                    }
                )

            with self._collection.aggregate(pipeline) as cursor:
                for document in cursor:
                    record = TraceRecord.from_document(
                        document,
                        assessments=tuple(
                            assessment["content"] for assessment in document["_assessments"]
                        ),
                    )
                    spans = (
                        [SpanRecord.from_document(span) for span in document["_spans"]]
                        if include_spans
                        else []
                    )
                    records.append((record, spans))

        records.sort(key=lambda item: trace_id_order[item[0].trace_id])
        return records

    @translate_database_errors
    def query_trace_metrics(
        self,
        *,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None,
        filters: list[str] | None,
        time_interval_seconds: int | None,
        start_time_ms: int | None,
        end_time_ms: int | None,
        max_results: int,
    ) -> list[MetricDataPoint]:
        """Aggregate metrics over trace, span, and assessment documents."""
        pipeline = build_trace_metrics_pipeline(
            experiment_ids=experiment_ids,
            view_type=view_type,
            metric_name=metric_name,
            aggregations=aggregations,
            dimensions=dimensions,
            filters=filters,
            time_interval_seconds=time_interval_seconds,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            max_results=max_results,
            spans_collection=self._settings.spans_collection_name,
            assessments_collection=self._settings.assessments_collection_name,
        )
        if max_results == 0:
            return []

        rows = list(self._collection.aggregate(pipeline, allowDiskUse=True))
        if (
            not rows
            and not dimensions
            and not time_interval_seconds
            and aggregations
            and all(agg.aggregation_type == AggregationType.COUNT for agg in aggregations)
        ):
            rows = [{f"agg_{index}": 0 for index in range(len(aggregations))}]
        return to_metric_data_points(rows, metric_name, aggregations)

    @translate_database_errors
    def set_trace_tag(self, *, trace_id: str, key: str, value: str) -> None:
        document = self._collection.find_one_and_update(
            {"_id": trace_id},
            [
                {
                    "$set": {
                        "tags": build_merge_array_expression("$tags", [{"k": key, "v": value}], "k")
                    }
                }
            ],
            projection={"_id": 1},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise TraceNotFoundError(trace_id)

    @translate_database_errors
    def delete_trace_tag(self, *, trace_id: str, key: str) -> None:
        document = self._collection.find_one_and_update(
            {"_id": trace_id, "tags.k": key},
            build_remove_array_element_update(array_field="tags", key_field="k", key=key),
            projection={"_id": 1},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise TraceNotFoundError(trace_id)

    @translate_database_errors
    def link_prompts(self, *, trace_id: str, prompt_versions: list[Mapping[str, str]]) -> None:
        """Add prompt-version references to a trace without duplicate entries."""
        if not prompt_versions:
            return

        document = self._collection.find_one_and_update(
            {"_id": trace_id},
            {"$addToSet": {"linked_prompts": {"$each": [dict(pv) for pv in prompt_versions]}}},
            projection={"_id": 1},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise TraceNotFoundError(trace_id)

    @translate_database_errors
    def get_spans(self, trace_id: str) -> list[SpanRecord]:
        cursor = self._spans_collection.find(
            {"trace_id": trace_id},
            {
                "_id": 0,
                "trace_id": 1,
                "span_id": 1,
                "parent_span_id": 1,
                "start_time_ns": 1,
                "end_time_ns": 1,
                "content": 1,
            },
        ).sort([("start_time_ns", ASCENDING), ("span_id", ASCENDING)])
        return [SpanRecord.from_document(document) for document in cursor]

    @translate_database_errors
    def delete_traces(
        self,
        *,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """Best-effort hard deletion of traces selected by validated store criteria.

        Span, assessment, and trace deletion are separate writes. Failures can leave
        partial deletion, and concurrent writers can leave child documents or recreate
        a trace.
        """
        query = {"experiment_id": experiment_id}
        if max_timestamp_millis is not None:
            query["request_time"] = {"$lte": max_timestamp_millis}
        if trace_ids:
            query["_id"] = {"$in": trace_ids}

        cursor = self._collection.find(query, {"_id": 1}).sort(
            [
                ("request_time", ASCENDING),
                ("_id", ASCENDING),
            ]
        )
        if max_traces is not None:
            cursor = cursor.limit(max_traces)

        deleted_count = 0
        with cursor:
            while True:
                # Bound memory and the size of each deletion command even when the
                # timestamp selection matches a large number of traces.
                selected_ids = {document["_id"] for document in islice(cursor, 500)}
                if not selected_ids:
                    break
                # Resolve ownership through traces before touching spans. Orphan
                # spans cannot safely be scoped to an experiment with this schema.
                self._spans_collection.delete_many({"trace_id": {"$in": selected_ids}})
                self._assessments_collection.delete_many({"trace_id": {"$in": selected_ids}})
                # Keeping metadata until span deletion succeeds allows retries
                # to find traces whose span cleanup failed or was interrupted.
                result = self._collection.delete_many(
                    {
                        "experiment_id": experiment_id,
                        "_id": {"$in": selected_ids},
                    }
                )
                deleted_count += result.deleted_count
        return deleted_count
