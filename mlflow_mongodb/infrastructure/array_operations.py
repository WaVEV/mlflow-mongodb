"""Private MongoDB array expression and update helpers shared by both stores."""

from collections.abc import Mapping
from typing import Any


def build_array_value_expression(field: str, key: str) -> dict[str, Any]:
    """Read a value by key from an embedded ``{k, v}`` array expression."""
    return {
        "$getField": {
            "field": {"$literal": key},
            "input": {"$arrayToObject": {"$ifNull": [field, []]}},
        }
    }


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


def build_array_value_expression(field: str, key: str) -> dict[str, Any]:
    """Read a value by key from an embedded ``{k, v}`` array expression."""
    return {
        "$getField": {
            "field": {"$literal": key},
            "input": {"$arrayToObject": {"$ifNull": [field, []]}},
        }
    }


def build_merge_array_expression(
    field: str,
    records: list[Mapping[str, Any]],
    identity: str,
) -> dict[str, Any]:
    """Build a MongoDB expression that merges embedded records by identity."""
    incoming = {"$literal": [dict(record) for record in records]}
    keys = {"$literal": [record[identity] for record in records]}
    remaining_records = {
        "$filter": {
            "input": {"$ifNull": [field, []]},
            "as": "stored",
            "cond": {"$not": [{"$in": [f"$$stored.{identity}", keys]}]},
        }
    }
    return {"$concatArrays": [remaining_records, incoming]}


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
