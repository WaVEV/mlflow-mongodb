"""Internal helpers for repository operations."""

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from bson.errors import BSONError
from pymongo.errors import PyMongoError

from mlflow_mongodb.repositories.errors import RepositoryPersistenceError

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def translate_database_errors(
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Translate driver and BSON failures, preserving domain errors and the cause."""

    @wraps(function)
    def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        try:
            return function(*args, **kwargs)
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError(
                f"Database operation '{function.__name__}' failed."
            ) from exc

    return wrapper


def build_merge_array_expression(
    field: str,
    records: list[Mapping[str, Any]],
    identity: str,
    *,
    protected_keys: str | None = None,
) -> dict[str, Any]:
    """Build a MongoDB expression that merges embedded records by identity.

    The returned aggregation expression starts with the existing array stored at ``field``.
    For each supplied record, it removes existing records with the same ``identity`` value
    and appends the supplied record. This replaces matching records without disturbing
    unrelated records, and also works when the stored array is missing or null.

    When ``protected_keys`` is provided, incoming records whose identity is already present
    in that field are ignored, so the existing value wins. This is useful when a later update
    must preserve authoritative or otherwise immutable values. Both ``field`` and
    ``protected_keys`` are MongoDB aggregation field paths, such as ``"$tags"`` or
    ``"$authoritative_metadata_keys"``.
    Caller-provided records are wrapped as literals so their values are not interpreted as
    aggregation expressions.
    """
    incoming = {"$literal": [dict(record) for record in records]}
    if protected_keys is not None:
        incoming = {
            "$filter": {
                "input": incoming,
                "as": "record",
                "cond": {
                    "$not": [{"$in": [f"$$record.{identity}", {"$ifNull": [protected_keys, []]}]}]
                },
            }
        }
    return {
        "$reduce": {
            "input": incoming,
            "initialValue": {"$ifNull": [field, []]},
            "in": {
                "$concatArrays": [
                    {
                        "$filter": {
                            "input": "$$value",
                            "as": "stored",
                            "cond": {
                                "$ne": [
                                    f"$$stored.{identity}",
                                    f"$$this.{identity}",
                                ]
                            },
                        }
                    },
                    ["$$this"],
                ]
            },
        }
    }


def build_replace_array_element_pipeline(
    *,
    array_field: str,
    key_field: str,
    element: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a pipeline that replaces an array element by its identity field.

    The identity value is read from ``element[key_field]``. The pipeline treats
    a missing or null array as empty, removes all existing elements with the
    same identity value, and appends ``element``. Caller-provided values are
    wrapped with ``$literal`` so they are treated as data rather than
    aggregation expressions.
    """
    return [
        {
            "$set": {
                array_field: build_merge_array_expression(f"${array_field}", [element], key_field)
            }
        }
    ]


def build_remove_array_element_update(
    *,
    array_field: str,
    key_field: str,
    key: str,
) -> dict[str, Any]:
    """Build a MongoDB ``$pull`` update that removes matching array elements."""
    return {"$pull": {array_field: {key_field: key}}}
