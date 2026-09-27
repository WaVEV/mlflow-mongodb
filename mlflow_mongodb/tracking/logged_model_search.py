"""Validate MLflow logged-model searches before repository access."""

import binascii
import math
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchLoggedModelsPaginationToken

from mlflow_mongodb.tracking.repositories.logged_models import LoggedModelFilter, LoggedModelOrder

_ATTRIBUTE_ALIASES = {
    "creation_time": "creation_timestamp",
    "creation_timestamp_ms": "creation_timestamp",
    "last_updated_timestamp_ms": "last_updated_timestamp",
}
_ATTRIBUTES = {
    "model_id",
    "experiment_id",
    "name",
    "artifact_location",
    "creation_timestamp",
    "last_updated_timestamp",
    "lifecycle_stage",
    "model_type",
    "source_run_id",
    "status_message",
}
_NUMERIC_ATTRIBUTES = {"creation_timestamp", "last_updated_timestamp"}
_ENTITY_TYPES = {"attributes": "attribute", "metrics": "metric", "params": "param", "tags": "tag"}


def _attribute_key(key: str) -> str:
    key = _ATTRIBUTE_ALIASES.get(key, key)
    if key not in _ATTRIBUTES:
        raise MlflowException.invalid_parameter_value(f"Invalid logged model attribute: {key!r}.")
    return key


def validate_logged_model_datasets(datasets: list[dict[str, Any]] | None) -> None:
    if datasets is None:
        return
    if not isinstance(datasets, list):
        raise MlflowException.invalid_parameter_value("`datasets` must be a list of dictionaries.")
    for dataset in datasets:
        if not isinstance(dataset, dict) or not dataset.get("dataset_name"):
            raise MlflowException.invalid_parameter_value(
                "`dataset_name` in the `datasets` clause must be specified."
            )
        if not isinstance(dataset["dataset_name"], str) or (
            dataset.get("dataset_digest") is not None
            and not isinstance(dataset["dataset_digest"], str)
        ):
            raise MlflowException.invalid_parameter_value(
                "Dataset names and digests must be strings."
            )


def parse_logged_model_filters(filter_string: str | None) -> tuple[LoggedModelFilter, ...]:
    # The parser imports SQL models; defer it until MLflow finishes store discovery.
    from mlflow.utils.search_logged_model_utils import (  # ruff: ignore[import-outside-top-level]
        parse_filter_string,
    )

    if filter_string is not None and not isinstance(filter_string, str):
        raise MlflowException.invalid_parameter_value("`filter_string` must be a string.")
    try:
        comparisons = parse_filter_string(filter_string)
    except (ValueError, TypeError, SyntaxError):
        raise MlflowException.invalid_parameter_value(
            "Invalid logged model filter string."
        ) from None

    filters = []
    for comparison in comparisons:
        field_type = _ENTITY_TYPES[comparison.entity.type.value]
        key = comparison.entity.key
        if field_type == "attribute":
            key = _attribute_key(key)
        if not key:
            raise MlflowException.invalid_parameter_value("Search keys must not be empty.")
        value = comparison.value
        if field_type == "metric" or (field_type == "attribute" and key in _NUMERIC_ATTRIBUTES):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MlflowException.invalid_parameter_value(
                    "Numeric filters require finite numbers."
                )
        elif comparison.op in ("IN", "NOT IN"):
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise MlflowException.invalid_parameter_value(
                    "IN and NOT IN require a list of string values."
                )
            value = tuple(value)
        elif not isinstance(value, str):
            raise MlflowException.invalid_parameter_value("String filters require string values.")
        filters.append(LoggedModelFilter(field_type, key, comparison.op, value))
    return tuple(filters)


def parse_logged_model_order(order_by: list[dict[str, Any]] | None) -> tuple[LoggedModelOrder, ...]:
    if order_by is not None and not isinstance(order_by, list):
        raise MlflowException.invalid_parameter_value("`order_by` must be a list of dictionaries.")
    orders = []
    seen = set()
    for order in order_by or []:
        if not isinstance(order, dict) or not isinstance(order.get("field_name"), str):
            raise MlflowException.invalid_parameter_value(
                "`field_name` in the `order_by` clause must be specified as a string."
            )
        field = order["field_name"]
        if "." in field:
            entity, key = field.split(".", 1)
            if entity != "metrics" or not key:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order by field name: {field!r}. Only metrics support a prefix."
                )
            field_type = "metric"
        else:
            key = _attribute_key(field)
            field_type = "attribute"
        ascending = order.get("ascending", True)
        if not isinstance(ascending, bool):
            raise MlflowException.invalid_parameter_value("`ascending` must be a boolean.")
        dataset_name = order.get("dataset_name")
        dataset_digest = order.get("dataset_digest")
        if any(
            value is not None and not isinstance(value, str)
            for value in (dataset_name, dataset_digest)
        ):
            raise MlflowException.invalid_parameter_value(
                "Dataset names and digests must be strings."
            )
        if dataset_digest and not dataset_name:
            raise MlflowException.invalid_parameter_value(
                "`dataset_digest` can only be specified if `dataset_name` is also specified."
            )
        if field_type != "metric" and (dataset_name or dataset_digest):
            raise MlflowException.invalid_parameter_value(
                "Dataset ordering applies only to metrics."
            )
        identity = (field_type, key, dataset_name or None, dataset_digest or None)
        # Later repetitions of the same sort expression cannot change its ordering.
        if identity not in seen:
            seen.add(identity)
            orders.append(
                LoggedModelOrder(field_type, key, ascending, dataset_name, dataset_digest)
            )
    for key, ascending in (("creation_timestamp", False), ("model_id", True)):
        if not any(order.field_type == "attribute" and order.key == key for order in orders):
            orders.append(LoggedModelOrder("attribute", key, ascending))
    sort_keys = sum(
        2
        if order.field_type == "metric"
        or order.key
        in {
            "model_type",
            "source_run_id",
            "status_message",
        }
        else 1
        for order in orders
    )
    if sort_keys > 32:
        raise MlflowException.invalid_parameter_value("Too many order_by fields.")
    return tuple(orders)


def parse_logged_model_page_token(
    page_token: str | None,
    experiment_ids: list[str],
    filter_string: str | None,
    order_by: list[dict[str, Any]] | None,
) -> int:
    # TODO: MOVE THIS LOGIC
    if page_token is not None and not isinstance(page_token, str):
        raise MlflowException.invalid_parameter_value("Invalid logged model page token.")
    if not page_token:
        return 0
    try:
        token = SearchLoggedModelsPaginationToken.decode(page_token)
    except (MlflowException, ValueError, TypeError, AttributeError, binascii.Error):
        raise MlflowException.invalid_parameter_value("Invalid logged model page token.") from None
    if (
        isinstance(token.offset, bool)
        or not isinstance(token.offset, int)
        or not 0 <= token.offset < 2**63
    ):
        raise MlflowException.invalid_parameter_value("Invalid logged model page token offset.")
    token.validate(experiment_ids, filter_string or None, order_by or None)
    return token.offset
