"""Shared validation of clauses already parsed by MLflow."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mlflow_mongodb.infrastructure.errors import (
    RepositoryInvalidAttributeError,
    RepositoryInvalidRegexError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
)


@dataclass(frozen=True)
class SearchFilterClause:
    """The common clause fields, without converting the parsed value."""

    field_type: str
    key: str
    comparator: str
    value: Any


class SearchFilterValidator(ABC):
    """Validate field/operator rules, required domain rules, then regex syntax."""

    def __init__(
        self,
        *,
        field_types: Collection[str],
        attribute_keys: Collection[str] | None = None,
        comparators: Mapping[str, Collection[str]] | None = None,
        field_comparators: Mapping[tuple[str, str], Collection[str]] | None = None,
        regex_comparators: Collection[str] = (),
        uppercase_comparators: bool = True,
    ):
        self._field_types = frozenset(field_types)
        self._attribute_keys = frozenset(attribute_keys) if attribute_keys is not None else None
        self._comparators = {
            field_type: frozenset(allowed) for field_type, allowed in (comparators or {}).items()
        }
        self._field_comparators = {
            field: frozenset(allowed) for field, allowed in (field_comparators or {}).items()
        }
        self._regex_comparators = frozenset(regex_comparators)
        self._uppercase_comparators = uppercase_comparators

    @abstractmethod
    def _validate_domain_rules(self, clause: SearchFilterClause) -> None:
        """Implement the domain validation required after field/operator validation."""
        raise NotImplementedError

    def validate(self, parsed: Mapping[str, Any]) -> SearchFilterClause:
        """Return normalized clause fields or a condition-specific shared error."""
        comparator = parsed["comparator"]
        clause = SearchFilterClause(
            parsed["type"],
            parsed["key"],
            comparator.upper() if self._uppercase_comparators else comparator,
            parsed["value"],
        )
        if clause.field_type not in self._field_types:
            raise RepositoryUnsupportedFieldTypeError(clause.field_type)
        if (
            clause.field_type == "attribute"
            and self._attribute_keys is not None
            and clause.key not in self._attribute_keys
        ):
            raise RepositoryInvalidAttributeError(clause.key)

        field = (clause.field_type, clause.key)
        allowed = self._field_comparators.get(field, self._comparators.get(clause.field_type))
        if allowed is not None and clause.comparator not in allowed:
            raise RepositoryUnsupportedComparatorError(
                clause.field_type,
                clause.key,
                clause.comparator,
                allowed,
                field_specific=field in self._field_comparators,
            )
        self._validate_domain_rules(clause)
        if clause.comparator in self._regex_comparators:
            try:
                re.compile(clause.value)
            except re.error as exc:
                raise RepositoryInvalidRegexError(
                    f"Invalid search filter regular expression: {exc}"
                ) from exc
        return clause


class ConfiguredSearchFilterValidator(SearchFilterValidator):
    """Supply the required domain hook through configured validation callbacks."""

    def __init__(
        self,
        *,
        rules: Sequence[Callable[[SearchFilterClause], None]] = (),
        **configuration: Any,
    ):
        super().__init__(**configuration)
        self._rules = tuple(rules)

    def _validate_domain_rules(self, clause: SearchFilterClause) -> None:
        for rule in self._rules:
            rule(clause)
