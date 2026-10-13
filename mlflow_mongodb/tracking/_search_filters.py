"""Private tracking search rules built on the shared infrastructure validator."""

from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey

from mlflow_mongodb.infrastructure.errors import (
    RepositoryInvalidPromptFilterError,
    RepositoryUnsupportedComparatorError,
)
from mlflow_mongodb.infrastructure.search_filters import SearchFilterClause, SearchFilterValidator


class _SearchTrackingFilterValidator(SearchFilterValidator):
    """Group trace domain rules while inheriting the common validation mechanics."""

    def _validate_domain_rules(self, clause: SearchFilterClause) -> None:
        self._validate_linked_prompt_filter(clause)
        self._validate_reserved_metadata_filter(clause)

    @staticmethod
    def _validate_linked_prompt_filter(clause: SearchFilterClause) -> None:
        if (
            clause.field_type == "tag"
            and clause.key == TraceTagKey.LINKED_PROMPTS
            and (clause.comparator != "=" or clause.value.count("/") != 1)
        ):
            raise RepositoryInvalidPromptFilterError("Invalid linked-prompt comparison.")

    @staticmethod
    def _validate_reserved_metadata_filter(clause: SearchFilterClause) -> None:
        allowed = ("=", "!=", "IS NULL", "IS NOT NULL")
        if (
            clause.field_type == "request_metadata"
            and clause.key in (TraceMetadataKey.TOKEN_USAGE, TraceMetadataKey.COST)
            and clause.comparator not in allowed
        ):
            raise RepositoryUnsupportedComparatorError(
                clause.field_type, clause.key, clause.comparator, allowed, field_specific=True
            )
