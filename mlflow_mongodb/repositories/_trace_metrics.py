"""Trace repository aggregation for MLflow trace, span, and assessment metrics."""

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    AssessmentMetricDimensionKey,
    AssessmentMetricKey,
    AssessmentMetricSearchKey,
    SpanAttributeKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    SpanMetricSearchKey,
    TraceMetadataKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
    TraceMetricSearchKey,
    TraceTagKey,
)
from mlflow.utils.search_utils import SearchTraceMetricsUtils

TIME_BUCKET_LABEL = "time_bucket"
ASSESSMENT_TYPE_FIELDS: Final = MappingProxyType(
    {
        "feedback": "content.feedback",
        "expectation": "content.expectation",
        "issue": "content.issue",
    }
)


def _array_value(field: str, key: str) -> dict[str, Any]:
    """Read one k/v record from an embedded trace array."""
    return {
        "$getField": {
            "field": {"$literal": key},
            "input": {"$arrayToObject": {"$ifNull": [field, []]}},
        }
    }


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
            return _array_value("$tags", TraceTagKey.TRACE_NAME)
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


def _percentile_expression(percentile: float) -> dict[str, Any]:
    rank = {
        "$multiply": [
            {"$subtract": [{"$size": "$_sorted_values"}, 1]},
            percentile / 100.0,
        ]
    }
    low = {"$toInt": {"$floor": rank}}
    high = {"$toInt": {"$ceil": rank}}
    low_value = {"$arrayElemAt": ["$_sorted_values", low]}
    high_value = {"$arrayElemAt": ["$_sorted_values", high]}
    return {
        "$add": [
            low_value,
            {"$multiply": [{"$subtract": [high_value, low_value]}, {"$subtract": [rank, low]}]},
        ]
    }


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
                        "_session": _array_value("$trace_metadata", TraceMetadataKey.TRACE_SESSION)
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
