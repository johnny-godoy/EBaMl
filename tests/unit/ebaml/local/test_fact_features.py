"""Tests for reading the features of an instance from an EBM."""

import re
from decimal import Decimal

import numpy as np
import pytest

from ebaml.core.custom_types import FeatureValue
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.local import fact_features as fact_features_module
from tests.unit.ebaml.conftest import (
    fake_model,
    returning,
)
from tests.unit.ebaml.local.conftest import hand, model_features


def test_bin_label() -> None:
    assert fact_features_module.bin_label((1.5, 2.5)) == "[1.5, 2.5)"
    assert fact_features_module.bin_label("red") == "red"


def test_bin_index_of_a_category_is_its_position() -> None:
    graph = EBMGraph("f", "categorical", ["a", "b", "c"], [0.0] * 3, [0.0] * 3)
    assert fact_features_module._bin_index(fake_model(), 0, graph, "b") == 1


@pytest.mark.parametrize(
    ("value", "category"),
    [
        (2, "2"),
        (2.0, "2"),
        (np.int64(3), "3"),
        (0.5, "0.5"),
        (np.float32(1.5), "1.5"),
        (True, "1"),
        (np.False_, "0"),
        ("x", "x"),
        (np.str_("y"), "y"),
    ],
)
def test_category_writes_a_value_as_interpret_writes_its_categories(value: FeatureValue, category: str) -> None:
    assert fact_features_module._category(value, "f") == category


def test_category_rejects_a_value_that_cannot_be_a_category() -> None:
    with pytest.raises(
        InvalidInstanceError,
        match=r"^The value np.complex128\(1j\) of feature 'f' cannot be a category\.$",
    ):
        fact_features_module._category(np.complex128(1j), "f")


def test_bin_index_of_a_number_finds_its_category() -> None:
    # interpret keeps the categories 1, 2 and 3 as text, so the integer 2 was an unknown category
    graph = EBMGraph("f", "nominal", ["1", "2", "3"], [0.0] * 3, [0.0] * 3)
    assert fact_features_module._bin_index(fake_model(), 0, graph, np.int64(2)) == 1


def test_bin_index_names_the_value_of_an_unknown_category() -> None:
    graph = EBMGraph("f", "nominal", ["1", "2"], [0.0] * 2, [0.0] * 2)
    with pytest.raises(InvalidInstanceError, match=r"^Unknown category 7 for feature 'f'\.$"):
        fact_features_module._bin_index(fake_model(), 0, graph, 7)


def test_bin_index_of_a_number_uses_the_first_cuts_and_the_right_side() -> None:
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [0.0] * 3, [0.0] * 3)
    # the second array stands for a finer resolution that the function must not use
    ebm = fake_model(bins_=[[np.array([1.0, 2.0]), np.array([5.0])]])
    assert fact_features_module._bin_index(ebm, 0, graph, 1.5) == 1
    assert fact_features_module._bin_index(ebm, 0, graph, 2.0) == 2  # a value on a cut belongs to the bin on its right


@pytest.mark.parametrize("value", ["abc", "", np.str_("1,5"), np.datetime64("2020-01-01")])
def test_bin_index_of_a_continuous_feature_rejects_a_value_that_is_not_a_number(value: FeatureValue) -> None:
    # float() raised a bare ValueError or TypeError, which is not an EBaMlError, so a caller could not catch it as one
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0)], [0.0] * 2, [0.0] * 2)
    ebm = fake_model(bins_=[[np.array([1.0])]])
    with pytest.raises(InvalidInstanceError, match=r"^Feature 'f' is continuous, but its value .* is not a number\.$"):
        fact_features_module._bin_index(ebm, 0, graph, value)


@pytest.mark.parametrize(("value", "expected"), [("1.5", 1), (np.str_("0.5"), 0), (True, 1), (Decimal("0.5"), 0)])
def test_bin_index_of_a_continuous_feature_accepts_a_value_that_float_reads(value: FeatureValue, expected: int) -> None:
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0)], [0.0] * 2, [0.0] * 2)
    ebm = fake_model(bins_=[[np.array([1.0])]])
    assert fact_features_module._bin_index(ebm, 0, graph, value) == expected


def test_plausible_categories_need_the_support() -> None:
    weights = np.array([1.0, 6.0, 1.0])  # the shares are 0.125, 0.75 and 0.125
    plausible = fact_features_module._plausible_bins(weights, continuous=False, min_support=0.2)
    assert plausible.tolist() == [False, True, False]
    # a share equal to the minimum is enough
    assert fact_features_module._plausible_bins(weights, continuous=False, min_support=0.125).tolist() == [
        True,
        True,
        True,
    ]


def test_plausible_bins_of_a_continuous_feature_trim_the_tails() -> None:
    weights = np.array([1.0, 6.0, 1.0])
    # the bins cover 0 to 0.125, 0.125 to 0.875 and 0.875 to 1 of the data
    assert fact_features_module._plausible_bins(weights, continuous=True, min_support=0.2).tolist() == [
        False,
        True,
        False,
    ]
    # a bin that ends exactly at the minimum, or starts exactly at 1 minus the minimum, is out
    assert fact_features_module._plausible_bins(weights, continuous=True, min_support=0.125).tolist() == [
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "term_features",
    [[(0,), (1,), (0, 1)], [(0, 1), (1,)]],
    ids=["extra term", "interaction"],
)
def test_features_reject_interactions(term_features: list[tuple[int, ...]]) -> None:
    with pytest.raises(UnsupportedModelError, match="interactions"):
        fact_features_module.read_features(model_features(term_features=term_features, classes=["n", "y"]), {}, 0.01)


@pytest.mark.parametrize(
    ("term_features", "missing"),
    [([(0,)], "['b']"), ([(1,)], "['a']"), ([(), (0,)], "['b']"), ([], "['a', 'b']")],
    ids=["last feature", "first feature", "empty term", "no term"],
)
def test_features_name_the_features_without_a_main_effect_term(
    term_features: list[tuple[int, ...]],
    missing: str,
) -> None:
    # a model fit with exclude, or after remove_terms, was rejected with a message about interactions it did not have
    model = model_features(term_features=term_features, classes=["n", "y"])
    expected = rf"main-effect term for every feature\. These features have none: {re.escape(missing)}\.$"
    with pytest.raises(UnsupportedModelError, match=expected):
        fact_features_module.read_features(model, {}, 0.01)


def test_features_read_the_bin_weights_of_the_term_of_each_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    # the terms are in another order than the features, so the weights of feature "a" are those of the second term
    graph = EBMGraph("a", "nominal", ["p", "q"], [0.0, 0.0], [0.0, 0.0])
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(returning(graph)))
    ebm = fake_model(
        feature_names_in_=["a", "b"],
        term_features_=[(1,), (0,)],
        classes_=np.array(["n", "y"]),
        link_="logit",
        bin_weights_=[np.array([0.0, 9.0, 9.0, 0.0]), np.array([0.0, 1.0, 3.0, 0.0])],
    )
    features = fact_features_module.read_features(ebm, {"a": "p", "b": "p"}, 0.01)
    assert features[0].weights.tolist() == [1.0, 3.0]


@pytest.mark.parametrize(
    ("feature_type", "expected"),
    [
        ("categorical", [False, False, True]),  # sorts before "continuous"
        ("nominal", [False, False, True]),  # sorts after "continuous"
        ("continuous", [False, True, True]),
    ],
)
def test_features_trim_by_the_type_of_the_feature(
    monkeypatch: pytest.MonkeyPatch,
    feature_type: str,
    expected: list[bool],
) -> None:
    # the shares are 0.25, 0.25 and 0.5: with a support of 0.3 a category needs 0.3 and the first bin of a number is cut
    graph = EBMGraph("a", feature_type, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [0.0] * 3, [0.0] * 3)
    if feature_type != "continuous":
        graph = EBMGraph("a", feature_type, ["p", "q", "r"], [0.0] * 3, [0.0] * 3)
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(returning(graph)))
    ebm = fake_model(
        feature_names_in_=["a"],
        term_features_=[(0,)],
        classes_=np.array(["n", "y"]),
        link_="logit",
        bins_=[[np.array([1.0, 2.0])]],
        bin_weights_=[np.array([0.0, 2.0, 2.0, 4.0, 0.0])],
    )
    value = "r" if feature_type != "continuous" else 2.5
    (feature,) = fact_features_module.read_features(ebm, {"a": value}, 0.3)
    assert feature.plausible.tolist() == expected
    assert feature.bin_index == 2


def test_robust_scores_move_by_z_standard_deviations() -> None:
    feature = hand("f", [1.0, -1.0, 4.0], stds=[0.5, 0.25, 1.0], plausible=[True, True, False])
    # z is 2, so the scores move by 1, 0.5 and 2; the unplausible bin is inf
    lowered = fact_features_module.robust_scores(feature, 1, 2.0, towards_outcome=True)
    raised = fact_features_module.robust_scores(feature, 1, 2.0, towards_outcome=False)
    assert lowered.tolist() == [0.0, -1.5, np.inf]
    assert raised.tolist() == [2.0, -0.5, np.inf]
    # for the negative class the scores are turned around first
    assert fact_features_module.robust_scores(feature, -1, 2.0, towards_outcome=True).tolist() == [-2.0, 0.5, np.inf]
