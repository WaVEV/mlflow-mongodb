"""Persistence operations for traces, spans, and assessments."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from types import MappingProxyType
from typing import Any, ClassVar, Final

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    AssessmentMetricDimensionKey,
    AssessmentMetricKey,
    AssessmentMetricSearchKey,
    SpanAttributeKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    SpanMetricSearchKey,
    SpansLocation,
    TraceMetadataKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
    TraceMetricSearchKey,
    TraceTagKey,
)
from mlflow.utils.search_utils import SearchTraceMetricsUtils, SearchUtils
from pymongo import ASCENDING, DESCENDING, ReplaceOne, ReturnDocument, UpdateOne
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.infrastructure._array_updates import (
    build_array_value_expression,
    build_merge_array_expression,
    build_remove_array_element_update,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking._retry import retry_on_exception
from mlflow_mongodb.tracking.errors import (
    RepositoryPersistenceError,
    TraceNotFoundError,
    TraceWriteConflictError,
    translate_database_errors,
)
from mlflow_mongodb.tracking.types import SpanRecord, SpanSummaryRecord, TraceRecord


@dataclass(frozen=True)
class TraceSearchFilter:
    field_type: str
    key: str
    operator: str
    value: Any


@dataclass(frozen=True)
class TraceSearchOrder:
    field_type: str
    key: str
    ascending: bool


class TraceRepository:
    """Own trace metadata and the separate span and assessment collections."""

    _READ_BATCH_SIZE = 500
    COMPARISON_OPERATORS: ClassVar[dict[str, str]] = {
        "=": "$eq",
        "!=": "$ne",
        "<": "$lt",
        "<=": "$lte",
        ">": "$gt",
        ">=": "$gte",
    }
    SEARCH_ATTRIBUTE_FIELDS: ClassVar[dict[str, str]] = {
        "request_id": "_id",
        "experiment_id": "experiment_id",
        "timestamp_ms": "request_time",
        "execution_time_ms": "execution_duration",
        "status": "state",
        "client_request_id": "client_request_id",
    }

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
            [("experiment_id", ASCENDING), ("request_time", DESCENDING), ("_id", ASCENDING)],
            name="traces_experiment_request_time_desc_id",
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
    def search_trace_infos(
        self,
        *,
        experiment_ids: list[str] | None,
        filters: list[TraceSearchFilter],
        order_by: list[TraceSearchOrder],
        offset: int,
        limit: int,
    ) -> list[TraceRecord]:
        """Select trace metadata after scoped filtering, sorting, and pagination."""
        # TODO: Support span and assessment filters in search_trace_infos.
        clauses: list[dict[str, Any]] = []
        if experiment_ids is not None:
            if not experiment_ids:
                return []
            clauses.append({"experiment_id": {"$in": experiment_ids}})
        direct_filters = [
            item for item in filters if item.field_type in ("attribute", "tag", "request_metadata")
        ]
        clauses.extend(self._search_filter_condition(item) for item in direct_filters)

        pipeline: list[dict[str, Any]] = []
        if clauses:
            pipeline.append({"$match": {"$and": clauses}})
        pipeline.extend(self._search_order_stages(order_by))
        pipeline.extend([{"$skip": offset}, {"$limit": limit}])
        pipeline.append({"$project": {"_id": 1}})

        cursor = self._collection.aggregate(pipeline, allowDiskUse=True)
        with cursor:
            trace_ids = [document["_id"] for document in cursor]
        return self.batch_get_trace_infos(trace_ids, experiment_ids=experiment_ids)

    @classmethod
    def _search_value_condition(cls, operator: str, value: Any) -> dict[str, Any]:
        if operator in (SearchUtils.LIKE_OPERATOR, SearchUtils.ILIKE_OPERATOR):
            pattern = re.escape(value).replace("%", ".*").replace("_", ".")
            return {
                "$regex": f"\\A{pattern}\\z",
                "$options": "is" if operator == SearchUtils.ILIKE_OPERATOR else "s",
            }
        if operator == "RLIKE":
            return {"$regex": value}
        if operator in ("IN", "NOT IN"):
            return {"$in" if operator == "IN" else "$nin": list(value)}
        return {cls.COMPARISON_OPERATORS[operator]: value}

    @classmethod
    def _search_filter_condition(cls, item: TraceSearchFilter) -> dict[str, Any]:
        if item.field_type == "attribute":
            if item.key == "end_time_ms":
                expression = {"$add": ["$request_time", {"$ifNull": ["$execution_duration", 0]}]}
                return {
                    "$expr": {
                        cls.COMPARISON_OPERATORS[item.operator]: [
                            expression,
                            {"$literal": item.value},
                        ]
                    }
                }
            field = cls.SEARCH_ATTRIBUTE_FIELDS[item.key]
            condition = cls._search_value_condition(item.operator, item.value)
            if item.operator in ("!=", "NOT IN"):
                condition["$exists"] = True
                return {"$and": [{field: condition}, {field: {"$ne": None}}]}
            return {field: condition}

        if item.field_type in ("tag", "request_metadata"):
            if item.field_type == "tag" and item.key == TraceTagKey.LINKED_PROMPTS:
                name, version = item.value.split("/", 1)
                return {"linked_prompts": {"$elemMatch": {"name": name, "version": version}}}
            field = "tags" if item.field_type == "tag" else "trace_metadata"
            key_match = {"k": item.key}
            if item.operator == "IS NULL":
                return {field: {"$not": {"$elemMatch": key_match}}}
            if item.operator == "IS NOT NULL":
                return {field: {"$elemMatch": key_match}}
            if (
                item.field_type == "request_metadata"
                and item.key == TraceMetadataKey.SOURCE_RUN
                and item.operator == "="
            ):
                return {
                    "$or": [
                        {"run_ids": item.value},
                        {field: {"$elemMatch": {"k": item.key, "v": item.value}}},
                    ]
                }
            return {
                field: {
                    "$elemMatch": {
                        "k": item.key,
                        "v": cls._search_value_condition(item.operator, item.value),
                    }
                }
            }

        raise ValueError(f"Unsupported trace search field type: {item.field_type}")

    def _search_order_stages(self, order_by: list[TraceSearchOrder]) -> list[dict[str, Any]]:
        """Build ordering stages with nulls last and deterministic tie breakers."""
        stages: list[dict[str, Any]] = []
        sort_fields: dict[str, int] = {}
        sort_values: dict[str, Any] = {}
        observed = set()
        for index, item in enumerate(order_by):
            observed.add((item.field_type, item.key))
            if item.field_type == "attribute" and item.key in (
                "timestamp_ms",
                "request_id",
                "experiment_id",
            ):
                sort_fields[self.SEARCH_ATTRIBUTE_FIELDS[item.key]] = (
                    ASCENDING if item.ascending else DESCENDING
                )
                continue
            expression = self._search_order_expression(item)
            value_field = f"_search_sort_{index}"
            null_field = f"_search_null_{index}"
            sort_values[value_field] = expression
            sort_values[null_field] = {"$cond": [{"$eq": [expression, None]}, 1, 0]}
            sort_fields[null_field] = ASCENDING
            sort_fields[value_field] = ASCENDING if item.ascending else DESCENDING
        if sort_values:
            stages.append({"$addFields": sort_values})
        if ("attribute", "timestamp_ms") not in observed:
            sort_fields["request_time"] = DESCENDING
        if ("attribute", "request_id") not in observed:
            sort_fields["_id"] = ASCENDING
        stages.append({"$sort": sort_fields})
        return stages

    @classmethod
    def _search_order_expression(cls, item: TraceSearchOrder) -> Any:
        if item.field_type == "attribute":
            if item.key == "end_time_ms":
                return {"$add": ["$request_time", {"$ifNull": ["$execution_duration", 0]}]}
            return f"${cls.SEARCH_ATTRIBUTE_FIELDS[item.key]}"
        field = "$tags" if item.field_type == "tag" else "$trace_metadata"
        return build_array_value_expression(field, item.key)

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

        # Preserve the last occurrence's position when IDs are repeated.
        unique_trace_ids = list(dict.fromkeys(reversed(trace_ids)))[::-1]
        records = []
        for start in range(0, len(unique_trace_ids), self._READ_BATCH_SIZE):
            batch_trace_ids = unique_trace_ids[start : start + self._READ_BATCH_SIZE]
            query = {"_id": {"$in": batch_trace_ids}}
            if experiment_ids is not None:
                query["experiment_id"] = {"$in": experiment_ids}
            pipeline = [
                {"$match": query},
                {
                    "$set": {
                        "_trace_order": {"$indexOfArray": [{"$literal": batch_trace_ids}, "$_id"]}
                    }
                },
                {"$sort": {"_trace_order": ASCENDING}},
                {"$unset": "_trace_order"},
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


TIME_BUCKET_LABEL = "time_bucket"
ASSESSMENT_TYPE_FIELDS: Final = MappingProxyType(
    {
        "feedback": "content.feedback",
        "expectation": "content.expectation",
        "issue": "content.issue",
    }
)


def _metric_value(view_type: MetricViewType, metric_name: str) -> Any:
    if view_type == MetricViewType.TRACES:
        if metric_name == TraceMetricKey.LATENCY:
            return "$execution_duration"
        if metric_name in TraceMetricKey.token_usage_keys():
            return f"$token_usage.{metric_name}"
    elif view_type == MetricViewType.SPANS:
        if metric_name == SpanMetricKey.LATENCY:
            return {
                "$floor": {
                    "$divide": [
                        {"$subtract": ["$_metric_row.end_time_ns", "$_metric_row.start_time_ns"]},
                        1_000_000,
                    ]
                }
            }
        if metric_name in SpanMetricKey.cost_keys():
            return f"$_metric_row.cost.{metric_name}"
    elif view_type == MetricViewType.ASSESSMENTS:
        if metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
            value = "$_assessment_value"
            return {
                "$switch": {
                    "branches": [
                        {"case": {"$in": [value, [True, "yes"]]}, "then": 1.0},
                        {"case": {"$in": [value, [False, "no"]]}, "then": 0.0},
                        {"case": {"$isNumber": value}, "then": value},
                    ],
                    "default": None,
                }
            }
    return None


def _dimension_value(view_type: MetricViewType, dimension: str) -> Any:
    if view_type == MetricViewType.TRACES:
        if dimension == TraceMetricDimensionKey.TRACE_NAME:
            return build_array_value_expression("$tags", TraceTagKey.TRACE_NAME)
        if dimension == TraceMetricDimensionKey.TRACE_STATUS:
            return "$state"
    elif view_type == MetricViewType.SPANS:
        fields = {
            SpanMetricDimensionKey.SPAN_NAME: "name",
            SpanMetricDimensionKey.SPAN_TYPE: "type",
            SpanMetricDimensionKey.SPAN_STATUS: "status",
        }
        if dimension in fields:
            return f"$_metric_row.{fields[dimension]}"
        if dimension == SpanMetricDimensionKey.SPAN_MODEL_NAME:
            return f"$_metric_row.dimension_attributes.{SpanAttributeKey.MODEL}"
        if dimension == SpanMetricDimensionKey.SPAN_MODEL_PROVIDER:
            return f"$_metric_row.dimension_attributes.{SpanAttributeKey.MODEL_PROVIDER}"
    elif view_type == MetricViewType.ASSESSMENTS:
        if dimension == AssessmentMetricDimensionKey.ASSESSMENT_NAME:
            return "$_metric_row.content.assessment_name"
        if dimension == AssessmentMetricDimensionKey.ASSESSMENT_VALUE:
            return "$_metric_row.value_json"
    raise MlflowException.invalid_parameter_value(
        f"Unsupported dimension `{dimension}` with view type {view_type}"
    )


def _filter_stages(filters: list[str] | None, view_type: MetricViewType) -> tuple[list, list]:
    trace_clauses = []
    row_clauses = []
    if view_type == MetricViewType.ASSESSMENTS:
        row_clauses.append({"content.valid": {"$ne": False}})
    for filter_string in filters or []:
        parsed = SearchTraceMetricsUtils.parse_search_filter(filter_string)
        value = parsed.value
        if parsed.view_type == TraceMetricSearchKey.VIEW_TYPE:
            if parsed.entity == TraceMetricSearchKey.STATUS:
                trace_clauses.append({"state": value})
            elif parsed.entity == TraceMetricSearchKey.TAG:
                trace_clauses.append({"tags": {"$elemMatch": {"k": parsed.key, "v": value}}})
            elif parsed.entity == TraceMetricSearchKey.METADATA:
                clause: dict[str, Any] = {
                    "trace_metadata": {"$elemMatch": {"k": parsed.key, "v": value}}
                }
                if parsed.key == TraceMetadataKey.SOURCE_RUN:
                    clause = {"$or": [clause, {"run_ids": value}]}
                trace_clauses.append(clause)
        elif parsed.view_type == SpanMetricSearchKey.VIEW_TYPE:
            if view_type != MetricViewType.SPANS:
                # TODO, just need to raise a custom exception and the raise this in the
                # controller.
                raise MlflowException.invalid_parameter_value(
                    f"Filtering by span is only supported for {MetricViewType.SPANS} view "
                    f"type, got {view_type}"
                )
            row_clauses.append({parsed.entity: value})
        elif parsed.view_type == AssessmentMetricSearchKey.VIEW_TYPE:
            if view_type != MetricViewType.ASSESSMENTS:
                raise MlflowException.invalid_parameter_value(
                    "Filtering by assessment is only supported for "
                    f"{MetricViewType.ASSESSMENTS} view type, got {view_type}"
                )
            field = "assessment_name" if parsed.entity == AssessmentMetricSearchKey.NAME else None
            if field:
                row_clauses.append({f"content.{field}": value})
            else:
                assessment_field = ASSESSMENT_TYPE_FIELDS.get(value)
                row_clauses.append(
                    {assessment_field: {"$exists": True}} if assessment_field else {"$expr": False}
                )
    return trace_clauses, row_clauses


def build_trace_metrics_pipeline(
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
    spans_collection: str,
    assessments_collection: str,
) -> list[dict[str, Any]]:
    """Build a trace-scoped pipeline with row-level filters for each view."""
    trace_clauses, row_clauses = _filter_stages(filters, view_type)
    if experiment_ids:
        trace_clauses.insert(0, {"experiment_id": {"$in": list(map(str, experiment_ids))}})
    if start_time_ms is not None or end_time_ms is not None:
        bounds = {}
        if start_time_ms is not None:
            bounds["$gte"] = start_time_ms
        if end_time_ms is not None:
            bounds["$lte"] = end_time_ms
        trace_clauses.insert(0, {"request_time": bounds})

    pipeline = []
    if trace_clauses:
        pipeline.append({"$match": {"$and": trace_clauses}})

    if view_type in (MetricViewType.SPANS, MetricViewType.ASSESSMENTS):
        collection = (
            spans_collection if view_type == MetricViewType.SPANS else assessments_collection
        )

        lookup: dict[str, Any] = {
            "from": collection,
            "localField": "_id",
            "foreignField": "trace_id",
            "as": "_metric_row",
        }
        if row_clauses:
            match = row_clauses[0] if len(row_clauses) == 1 else {"$and": row_clauses}
            lookup["pipeline"] = [{"$match": match}]
        pipeline.extend(
            [
                {"$lookup": lookup},
                {"$unwind": "$_metric_row"},
            ]
        )

    metric_match = {}
    if view_type == MetricViewType.TRACES:
        if TraceMetricDimensionKey.TRACE_NAME in (dimensions or []):
            metric_match["tags"] = {"$elemMatch": {"k": TraceTagKey.TRACE_NAME}}
        if metric_name == TraceMetricKey.SESSION_COUNT:
            metric_match["trace_metadata"] = {"$elemMatch": {"k": TraceMetadataKey.TRACE_SESSION}}
        elif metric_name in TraceMetricKey.token_usage_keys():
            metric_match[f"token_usage.{metric_name}"] = {"$exists": True}
    elif view_type == MetricViewType.SPANS:
        if metric_name in SpanMetricKey.cost_keys():
            metric_match[f"_metric_row.cost.{metric_name}"] = {"$exists": True}
    if metric_match:
        pipeline.append({"$match": metric_match})

    pipeline.extend(_metric_preparation_stages(view_type, metric_name))

    group_keys = {}
    if time_interval_seconds:
        if view_type == MetricViewType.SPANS:
            timestamp = {"$divide": ["$_metric_row.start_time_ns", 1_000_000]}
        elif view_type == MetricViewType.ASSESSMENTS:
            timestamp = {
                "$toLong": {"$dateFromString": {"dateString": "$_metric_row.content.create_time"}}
            }
        else:
            timestamp = "$request_time"
        size_ms = time_interval_seconds * 1000
        group_keys[TIME_BUCKET_LABEL] = {
            "$multiply": [{"$floor": {"$divide": [timestamp, size_ms]}}, size_ms]
        }
    for dimension in dimensions or []:
        group_keys[dimension] = {"$ifNull": [_dimension_value(view_type, dimension), None]}

    group = {"_id": group_keys or None}
    for index, aggregation in enumerate(aggregations):
        if aggregation.aggregation_type == AggregationType.COUNT:
            if metric_name == TraceMetricKey.SESSION_COUNT:
                group["_sessions"] = {"$addToSet": {"$ifNull": ["$_session", "$$REMOVE"]}}
            else:
                group[f"agg_{index}"] = {"$sum": 1}
        elif aggregation.aggregation_type == AggregationType.SUM:
            group[f"agg_{index}"] = {"$sum": "$_metric_value"}
            group["_has_numeric_value"] = {"$max": {"$isNumber": "$_metric_value"}}
        elif aggregation.aggregation_type == AggregationType.AVG:
            group[f"agg_{index}"] = {"$avg": "$_metric_value"}
        elif aggregation.aggregation_type == AggregationType.PERCENTILE:
            group[f"agg_{index}"] = {
                "$percentile": {
                    "input": "$_metric_value",
                    "p": [aggregation.percentile_value / 100.0],
                    "method": "approximate",
                }
            }
    pipeline.append({"$group": group})

    calculated = {}
    if metric_name == TraceMetricKey.SESSION_COUNT and view_type == MetricViewType.TRACES:
        for index in range(len(aggregations)):
            calculated[f"agg_{index}"] = {"$size": "$_sessions"}
    for index, aggregation in enumerate(aggregations):
        if aggregation.aggregation_type == AggregationType.SUM:
            calculated[f"agg_{index}"] = {"$cond": ["$_has_numeric_value", f"$agg_{index}", None]}
        elif aggregation.aggregation_type == AggregationType.PERCENTILE:
            calculated[f"agg_{index}"] = {"$arrayElemAt": [f"$agg_{index}", 0]}
    if calculated:
        pipeline.append({"$set": calculated})

    if group_keys:
        pipeline.append({"$sort": {f"_id.{key}": 1 for key in group_keys}})
    if max_results is not None:
        pipeline.append({"$limit": max_results})
    return pipeline


def _metric_preparation_stages(view_type: MetricViewType, metric_name: str) -> list[dict[str, Any]]:
    """Build the temporary metric fields needed by each view before grouping."""
    if view_type == MetricViewType.TRACES:
        if metric_name == TraceMetricKey.SESSION_COUNT:
            return [
                {
                    "$set": {
                        "_session": build_array_value_expression(
                            "$trace_metadata", TraceMetadataKey.TRACE_SESSION
                        )
                    }
                }
            ]
        value = _metric_value(view_type, metric_name)
        if value is not None:
            return [{"$set": {"_metric_value": value}}]
    elif view_type == MetricViewType.SPANS:
        value = _metric_value(view_type, metric_name)
        if value is not None:
            return [{"$set": {"_metric_value": value}}]
    elif view_type == MetricViewType.ASSESSMENTS:
        if metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
            return [
                {
                    "$set": {
                        "_assessment_value": {
                            "$ifNull": [
                                "$_metric_row.content.feedback.value",
                                {
                                    "$ifNull": [
                                        "$_metric_row.content.expectation.value",
                                        "$_metric_row.content.issue",
                                    ]
                                },
                            ]
                        }
                    }
                },
                {"$set": {"_metric_value": _metric_value(view_type, metric_name)}},
            ]
    return []


def to_metric_data_points(
    rows: list[dict[str, Any]], metric_name: str, aggregations: list[MetricAggregation]
) -> list[MetricDataPoint]:
    """Apply MLflow's result conversion, including null groups and UTC buckets."""
    points = []
    for row in rows:
        dimensions = dict(row.get("_id") or {})
        if any(value is None for value in dimensions.values()):
            continue
        if TIME_BUCKET_LABEL in dimensions:
            dimensions[TIME_BUCKET_LABEL] = datetime.fromtimestamp(
                float(dimensions[TIME_BUCKET_LABEL]) / 1000, tz=timezone.utc
            ).isoformat()
        values = {
            str(aggregation): row[f"agg_{index}"]
            for index, aggregation in enumerate(aggregations)
            if row.get(f"agg_{index}") is not None
        }
        if values:
            points.append(
                MetricDataPoint(metric_name=metric_name, dimensions=dimensions, values=values)
            )
    return points
