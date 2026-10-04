"""Private MongoDB search primitives without entity-specific null semantics."""

import re
from types import MappingProxyType

COMPARISON_OPERATORS = MappingProxyType(
    {"=": "$eq", "!=": "$ne", "<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}
)


def like_regex(value: str, comparator: str) -> re.Pattern[str]:
    """Build the registry LIKE pattern, including its final-newline behavior."""
    pattern = re.escape(value).replace("%", ".*").replace("_", ".")
    if not value.startswith("%"):
        pattern = f"^{pattern}"
    if not value.endswith("%"):
        pattern = f"{pattern}$"
    flags = re.DOTALL | (re.IGNORECASE if comparator == "ILIKE" else 0)
    return re.compile(pattern, flags)
