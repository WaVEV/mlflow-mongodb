"""Unit tests for shared search validation and LIKE matching."""

from dataclasses import FrozenInstanceError

import pytest

from mlflow_mongodb.infrastructure.errors import (
    RepositoryInvalidAttributeError,
    RepositoryInvalidRegexError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
)
from mlflow_mongodb.infrastructure.search_filters import (
    ConfiguredSearchFilterValidator,
    SearchFilterClause,
    SearchFilterValidator,
    like_regex,
)


@pytest.mark.parametrize(
    ("value", "comparator", "candidate", "matches"),
    [
        ("model", "LIKE", "model", True),
        ("model", "LIKE", "other-model", False),
        ("model", "LIKE", "model-other", False),
        ("model", "LIKE", "MODEL", False),
        ("model", "ILIKE", "MODEL", True),
        ("model%", "LIKE", "model\nversion", True),
        ("%model", "LIKE", "other-model", True),
        ("%model%", "LIKE", "other-model-version", True),
        ("model_", "LIKE", "model1", True),
        ("model_", "LIKE", "model12", False),
        ("model_", "LIKE", "model\n", True),
        ("a.b[1]+(x)?^$\\", "LIKE", "a.b[1]+(x)?^$\\", True),
        ("a.b", "LIKE", "axb", False),
        ("model", "LIKE", "model\n", True),
        ("model", "LIKE", "model\nother", False),
        ("", "LIKE", "", True),
        ("", "LIKE", "model", False),
        ("%", "LIKE", "", True),
    ],
)
def test_like_regex_matching(value, comparator, candidate, matches):
    assert (like_regex(value, comparator).search(candidate) is not None) is matches


@pytest.fixture
def validator():
    return ConfiguredSearchFilterValidator(
        field_types={"attribute", "tag"},
        attribute_keys={"name"},
        comparators={"attribute": {"=", "LIKE"}, "tag": {"=", "RLIKE"}},
        regex_comparators={"RLIKE"},
    )


def test_validate_normalizes_comparator_without_converting_value(validator):
    value = ["first", "second"]
    parsed = {"type": "tag", "key": "team", "comparator": "=", "value": value}

    clause = validator.validate(parsed)

    assert clause == SearchFilterClause("tag", "team", "=", value)
    assert clause.value is value
    assert parsed == {"type": "tag", "key": "team", "comparator": "=", "value": value}
    assert (
        validator.validate(
            {"type": "attribute", "key": "name", "comparator": "like", "value": "model%"}
        ).comparator
        == "LIKE"
    )


def test_clause_fields_are_immutable():
    clause = SearchFilterClause("attribute", "name", "=", "model")

    with pytest.raises(FrozenInstanceError):
        clause.key = "other"


@pytest.mark.parametrize(
    ("field_type", "key", "error_type"),
    [
        ("unknown", "unknown", RepositoryUnsupportedFieldTypeError),
        ("attribute", "unknown", RepositoryInvalidAttributeError),
        ("attribute", "name", RepositoryUnsupportedComparatorError),
    ],
)
def test_field_validation_precedes_domain_and_regex_checks(field_type, key, error_type):
    calls = []
    validator = ConfiguredSearchFilterValidator(
        field_types={"attribute"},
        attribute_keys={"name"},
        comparators={"attribute": {"="}},
        regex_comparators={"RLIKE"},
        rules=(calls.append,),
    )

    with pytest.raises(error_type):
        validator.validate({"type": field_type, "key": key, "comparator": "RLIKE", "value": "["})

    assert calls == []


def test_domain_rules_run_in_order_before_regex_validation():
    calls = []

    def first_rule(clause):
        calls.append(("first", clause.comparator))

    def second_rule(clause):
        calls.append(("second", clause.comparator))
        raise ValueError("Domain rule rejected the filter")

    validator = ConfiguredSearchFilterValidator(
        field_types={"tag"},
        regex_comparators={"RLIKE"},
        rules=(first_rule, second_rule),
    )

    with pytest.raises(ValueError, match="Domain rule rejected the filter"):
        validator.validate({"type": "tag", "key": "team", "comparator": "rlike", "value": "["})

    assert calls == [("first", "RLIKE"), ("second", "RLIKE")]


def test_invalid_regex_raises_shared_error_after_domain_validation(validator):
    with pytest.raises(RepositoryInvalidRegexError):
        validator.validate({"type": "tag", "key": "team", "comparator": "RLIKE", "value": "["})


def test_regex_validation_preserves_pattern_and_ignores_other_comparators(validator):
    pattern = r"^team\d+$"
    clause = validator.validate(
        {"type": "tag", "key": "team", "comparator": "RLIKE", "value": pattern}
    )

    assert clause.value == pattern
    assert (
        validator.validate({"type": "tag", "key": "team", "comparator": "=", "value": "["}).value
        == "["
    )


def test_field_comparators_override_field_type_defaults():
    validator = ConfiguredSearchFilterValidator(
        field_types={"tag"},
        comparators={"tag": {"="}},
        field_comparators={("tag", "team"): {"LIKE", "ILIKE"}},
    )
    assert (
        validator.validate(
            {"type": "tag", "key": "team", "comparator": "like", "value": "platform%"}
        ).comparator
        == "LIKE"
    )
    assert (
        validator.validate(
            {"type": "tag", "key": "other", "comparator": "=", "value": "platform"}
        ).comparator
        == "="
    )

    with pytest.raises(RepositoryUnsupportedComparatorError) as caught:
        validator.validate({"type": "tag", "key": "team", "comparator": "=", "value": "x"})

    assert caught.value.field_type == "tag"
    assert caught.value.key == "team"
    assert caught.value.comparator == "="
    assert caught.value.allowed == ("ILIKE", "LIKE")
    assert caught.value.field_specific is True


def test_validator_snapshots_mutable_configuration():
    field_types = {"attribute"}
    attribute_keys = {"name"}
    allowed = {"="}
    comparators = {"attribute": allowed}
    field_allowed = {"LIKE"}
    field_comparators = {("attribute", "name"): field_allowed}
    regex_comparators = {"LIKE"}
    calls = []
    rules = [calls.append]
    validator = ConfiguredSearchFilterValidator(
        field_types=field_types,
        attribute_keys=attribute_keys,
        comparators=comparators,
        field_comparators=field_comparators,
        regex_comparators=regex_comparators,
        rules=rules,
    )
    for configuration in (
        field_types,
        attribute_keys,
        allowed,
        comparators,
        field_allowed,
        field_comparators,
        regex_comparators,
        rules,
    ):
        configuration.clear()

    clause = validator.validate(
        {"type": "attribute", "key": "name", "comparator": "like", "value": "model"}
    )

    assert calls == [clause]
    with pytest.raises(RepositoryInvalidRegexError):
        validator.validate({"type": "attribute", "key": "name", "comparator": "LIKE", "value": "["})


def test_comparator_case_can_be_preserved():
    validator = ConfiguredSearchFilterValidator(
        field_types={"attribute"},
        comparators={"attribute": {"like"}},
        uppercase_comparators=False,
    )

    assert (
        validator.validate(
            {"type": "attribute", "key": "name", "comparator": "like", "value": "model%"}
        ).comparator
        == "like"
    )
    with pytest.raises(RepositoryUnsupportedComparatorError):
        validator.validate(
            {"type": "attribute", "key": "name", "comparator": "LIKE", "value": "model%"}
        )


def test_base_validator_requires_domain_hook():
    with pytest.raises(TypeError, match="_validate_domain_rules"):
        SearchFilterValidator(field_types={"attribute"})
