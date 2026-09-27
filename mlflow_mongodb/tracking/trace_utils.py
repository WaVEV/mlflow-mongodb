"""Adapt MLflow spans to BSON documents and derive trace summaries."""

import json
import logging
import math
from typing import Any

from bson import BSON, ObjectId
from bson.errors import InvalidDocument
from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS,
    GenAiSemconvKey,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.otel.translation import translate_span_when_storing
from mlflow.tracing.utils import (
    SpanAggregationNode,
    aggregate_cost_from_span_nodes,
    aggregate_usage_from_span_nodes,
)
from mlflow.tracing.utils.truncation import _get_truncated_preview
from mlflow.utils.validation import _validate_trace_tag

from mlflow_mongodb.tracking.types import SpanSummaryRecord

_logger = logging.getLogger(__name__)
_MAX_BSON_DOCUMENT_BYTES = 16 * 1024 * 1024


def _attribute_value(value: Any) -> Any:
    # Span.to_dict() retains MLflow's JSON-encoded OTel attribute values.
    # Decode only the fields used for queries; preserve content for Span.from_dict().
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def numeric_stats(value: Any) -> dict | None:
    """Read optional numeric usage/cost attributes without failing span ingestion."""
    value = _attribute_value(value)
    if not isinstance(value, dict):
        return None
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or (isinstance(number, float) and not math.isfinite(number))
        for number in value.values()
    ):
        return None
    return value


def _add_tag(tags: dict[str, str], key: str, value: Any) -> None:
    try:
        value = value if isinstance(value, str) else json.dumps(value)
        key, value = _validate_trace_tag(key, value)
    except (MlflowException, TypeError, ValueError):
        _logger.debug("Skipping invalid span-derived trace tag %r", key)
        return
    tags[key] = value


def _preview(value: Any, role: str) -> str | None:
    try:
        if value is not None and not isinstance(value, (str, dict)):
            value = json.dumps(value)
        preview = _get_truncated_preview(value, role=role)
        # MLflow's helper consults the process-global URI for its size limit.
        # A direct MongoDB store always uses the OSS limit, even in mixed clients.
        if preview is not None and len(preview) > TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS:
            return preview[: TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS - 3] + "..."
        return preview
    except (TypeError, ValueError, AttributeError):
        _logger.debug("Could not extract the %s trace preview", role, exc_info=True)
        return None


def span_to_document(span: Span) -> dict[str, Any]:
    """Keep full span content plus compact native fields for summary queries."""

    content = translate_span_when_storing(span)
    attributes = content.get("attributes", {})
    root = span.parent_id is None
    resource_tags = {}
    resource = getattr(span._span, "resource", None)
    if resource is not None:
        for key, value in resource.attributes.items():
            if not key.startswith(("telemetry.sdk.", "mlflow.")):
                _add_tag(resource_tags, key, value)
    root_tags = {}
    if root:
        for key, value in attributes.items():
            if key.startswith(SpanAttributeKey.TRACE_TAG_PREFIX):
                tag_key = key[len(SpanAttributeKey.TRACE_TAG_PREFIX) :]
                if tag_key != TraceTagKey.SPANS_LOCATION:
                    _add_tag(root_tags, tag_key, _attribute_value(value))

    metadata = {}
    session_id = attributes.get(SpanAttributeKey.SESSION_ID) or attributes.get(
        GenAiSemconvKey.CONVERSATION_ID
    )
    for key, value in (
        (TraceMetadataKey.TRACE_SESSION, session_id),
        (TraceMetadataKey.TRACE_USER, attributes.get(SpanAttributeKey.USER_ID)),
    ):
        if value is not None:
            metadata[key] = str(_attribute_value(value))

    document = {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_id,
        "name": span.name,
        "type": _attribute_value(attributes.get(SpanAttributeKey.SPAN_TYPE)) or span.span_type,
        "status": span.status.status_code.value,
        "start_time_ns": span.start_time_ns,
        "end_time_ns": span.end_time_ns,
        "content": content,
        "dimension_attributes": {
            key: _attribute_value(attributes[key])
            for key in (SpanAttributeKey.MODEL, SpanAttributeKey.MODEL_PROVIDER)
            if key in attributes
        },
        "token_usage": numeric_stats(attributes.get(SpanAttributeKey.CHAT_USAGE)),
        "cost": numeric_stats(attributes.get(SpanAttributeKey.LLM_COST)),
        "trace_fields": {
            "metadata": metadata,
            "resource_tags": resource_tags,
            "root_tags": root_tags,
            "request_preview": _preview(attributes.get(SpanAttributeKey.INPUTS), "user")
            if root
            else None,
            "response_preview": _preview(attributes.get(SpanAttributeKey.OUTPUTS), "assistant")
            if root
            else None,
        },
    }
    # Include the automatically assigned _id in the size check. Validate the whole
    # batch before any writes, including BSON type/size failures in later spans.
    try:
        size = len(BSON.encode({"_id": ObjectId(), **document}))
    except (InvalidDocument, OverflowError, ValueError, TypeError) as exc:
        raise MlflowException.invalid_parameter_value(
            f"Span '{span.span_id}' in trace '{span.trace_id}' cannot be stored as BSON."
        ) from exc
    if size > _MAX_BSON_DOCUMENT_BYTES:
        raise MlflowException.invalid_parameter_value(
            f"Span '{span.span_id}' in trace '{span.trace_id}' exceeds MongoDB's "
            "16 MiB document limit."
        )
    return document


def summarize_spans(documents: list[SpanSummaryRecord]) -> dict[str, Any]:
    """Recompute from unique persisted spans, including parents arriving in later batches."""

    start_ms = min(document.start_time_ns for document in documents) // 1_000_000
    end_times = [d.end_time_ns for d in documents if d.end_time_ns is not None]
    root = next((d for d in documents if d.parent_span_id is None), None)
    metadata = {}
    tags = {}
    for document in documents:
        for key, value in document.trace_fields["metadata"].items():
            metadata.setdefault(key, value)
        for key, value in document.trace_fields["resource_tags"].items():
            tags.setdefault(key, value)
    if root:
        tags.update(root.trace_fields["root_tags"])

    if not root:
        state = TraceState.IN_PROGRESS.value
    elif root.status == SpanStatusCode.ERROR.value:
        state = TraceState.ERROR.value
    else:
        state = TraceState.OK.value

    return {
        "request_time": start_ms,
        "execution_duration": max(end_times) // 1_000_000 - start_ms if end_times else None,
        "state": state,
        "request_preview": root.trace_fields["request_preview"] if root else None,
        "response_preview": root.trace_fields["response_preview"] if root else None,
        "tags": tags,
        "metadata": metadata,
        "token_usage": aggregate_usage_from_span_nodes(
            [SpanAggregationNode(d.span_id, d.parent_span_id, d.token_usage) for d in documents]
        ),
        "cost": aggregate_cost_from_span_nodes(
            [SpanAggregationNode(d.span_id, d.parent_span_id, d.cost) for d in documents]
        ),
    }
