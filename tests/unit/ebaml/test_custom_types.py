"""Tests for the custom types."""

import pytest

from ebaml.custom_types import (
    ExplainableBoostingModel,
    FeatureType,
    Normalization,
    coerce_enum,
    normalize_feature_type,
    normalize_feature_types,
)
from ebaml.exceptions import InvalidOptionError


def test_normalization_values() -> None:
    assert {n.name: n.value for n in Normalization} == {"NONE": "none", "MIN_MAX": "min_max", "Z_SCORE": "z_score"}


def test_feature_type_values() -> None:
    assert [f.value for f in FeatureType] == ["continuous", "categorical", "ordinal", "nominal", "boolean"]


def test_str_enum_compares_to_str() -> None:
    assert Normalization.NONE == "none"
    assert str(FeatureType.BOOLEAN) == "boolean"


def test_model_union_has_two_members() -> None:
    assert len(ExplainableBoostingModel.__args__) == 2


@pytest.mark.parametrize("name", ["continuous", "CONTINUOUS", "Continuous"])
def test_normalize_feature_type(name: str) -> None:
    assert normalize_feature_type(name) is FeatureType.CONTINUOUS


def test_normalize_feature_type_invalid() -> None:
    with pytest.raises(InvalidOptionError, match="bogus"):
        normalize_feature_type("bogus")


def test_normalize_feature_types_default_reject_set() -> None:
    assert normalize_feature_types(["continuous", "Boolean"]) == [FeatureType.CONTINUOUS, FeatureType.BOOLEAN]


def test_normalize_feature_types_rejected() -> None:
    with pytest.raises(InvalidOptionError, match="boolean"):
        normalize_feature_types(["continuous", "boolean"], reject_set={FeatureType.BOOLEAN})


def test_normalize_feature_types_accepted_with_reject_set() -> None:
    assert normalize_feature_types(["ordinal"], reject_set={FeatureType.BOOLEAN}) == [FeatureType.ORDINAL]


def test_normalize_feature_types_invalid() -> None:
    with pytest.raises(InvalidOptionError, match="bogus"):
        normalize_feature_types(["bogus"])


def test_normalize_feature_types_error_lists_only_accepted() -> None:
    with pytest.raises(InvalidOptionError) as info:
        normalize_feature_types(["bogus"], reject_set={FeatureType.BOOLEAN})
    assert "CONTINUOUS" in str(info.value)
    assert "BOOLEAN" not in str(info.value)


def test_normalize_feature_types_rejected_error_lists_only_accepted() -> None:
    with pytest.raises(InvalidOptionError) as info:
        normalize_feature_types(["boolean"], reject_set={FeatureType.BOOLEAN})
    assert "CONTINUOUS" in str(info.value)
    assert "BOOLEAN" not in str(info.value)


@pytest.mark.parametrize("value", ["z_score", Normalization.Z_SCORE])
def test_coerce_enum(value: str | Normalization) -> None:
    assert coerce_enum(Normalization, value) is Normalization.Z_SCORE


def test_coerce_enum_invalid() -> None:
    with pytest.raises(InvalidOptionError, match="bogus"):
        coerce_enum(Normalization, "bogus")
