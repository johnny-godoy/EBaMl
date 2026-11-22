"""Tests of the limits of a recourse plan: immutable features and allowed values."""

import numpy as np
import pytest

from ebaml.core.exceptions import InvalidOptionError, RecourseOptionError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.local.fact_features import Feature
from ebaml.local.recourse import Levels, RangeMatch, RecourseOptions, ValueRange
from tests.unit.ebaml.local.conftest import scored_feature

# a continuous feature with the bins [0, 10), [10, 20), [20, 30) and [30, 40), the instance in the second one
_EDGES = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]


def _continuous(name: str = "x", plausible: list[bool] | None = None) -> Feature:
    graph = EBMGraph(name, "continuous", _EDGES, [0.0, 1.0, 2.0, 3.0], [0.0] * 4)
    return Feature(name, graph, np.ones(4), 1, 15.0, np.array(plausible or [True] * 4))


def _nominal(name: str = "f") -> Feature:
    feature = scored_feature([0.0, 1.0, 2.0], 0)
    return Feature(name, feature.graph, feature.weights, 0, "0", feature.plausible)


@pytest.mark.parametrize(
    ("value_range", "expected"),
    [
        # 15 to 25 touches the second and the third bin
        (ValueRange(low=15, high=25), [False, True, True, False]),
        (ValueRange(low=15, high=25, match=RangeMatch.INSIDE), [False, False, False, False]),
        (ValueRange(low=10, high=30, match="inside"), [False, True, True, False]),
        # a bin ends before its high edge, so a range that starts at 10 leaves out [0, 10)
        (ValueRange(low=10), [False, True, True, True]),
        (ValueRange(high=10), [True, True, False, False]),
    ],
)
def test_a_value_range_allows_the_bins_that_its_match_names(value_range: ValueRange, expected: list[bool]) -> None:
    assert value_range.bins(_continuous()).tolist() == expected


def test_a_value_range_accepts_the_text_of_a_match() -> None:
    assert ValueRange(match="overlap").match is RangeMatch.OVERLAP


def test_an_unknown_match_names_the_valid_ones() -> None:
    with pytest.raises(InvalidOptionError, match="OVERLAP"):
        ValueRange(match="around")


def test_a_value_range_with_its_ends_swapped_is_rejected() -> None:
    with pytest.raises(RecourseOptionError, match="low <= high"):
        ValueRange(low=2, high=1)


def test_a_value_range_rejects_a_nominal_feature() -> None:
    with pytest.raises(RecourseOptionError, match="does not accept a value range"):
        ValueRange(high=1).bins(_nominal())


def test_levels_allow_the_bins_of_their_categories() -> None:
    assert Levels(["1", "2"]).bins(_nominal()).tolist() == [False, True, True]


def test_levels_reject_a_category_that_the_feature_does_not_have() -> None:
    with pytest.raises(RecourseOptionError, match=r"no categories \['9'\]"):
        Levels(["1", "9"]).bins(_nominal())


def test_levels_reject_a_continuous_feature() -> None:
    with pytest.raises(RecourseOptionError, match="does not accept levels"):
        Levels(["1"]).bins(_continuous())


def test_an_immutable_feature_can_end_only_in_the_bin_of_the_instance() -> None:
    (feature,) = RecourseOptions(immutable=["x"]).restrict([_continuous()])
    assert feature.reachable.tolist() == [False, True, False, False]
    assert feature.plausible.tolist() == [True] * 4


def test_allowed_values_keep_only_plausible_bins_and_the_bin_of_the_instance() -> None:
    options = RecourseOptions(allowed={"x": ValueRange(low=25)})
    (feature,) = options.restrict([_continuous(plausible=[True, True, True, False])])
    assert feature.reachable.tolist() == [False, True, True, False]


def test_a_feature_with_no_option_keeps_every_plausible_bin() -> None:
    original = _continuous()
    (feature,) = RecourseOptions(immutable=["y"]).restrict([original, _continuous("y")])[:1]
    assert feature is original
    assert feature.reachable.tolist() == feature.plausible.tolist()


def test_allowed_values_that_leave_no_plausible_bin_are_rejected() -> None:
    options = RecourseOptions(allowed={"x": ValueRange(low=31)})
    with pytest.raises(RecourseOptionError, match="leave no plausible bin"):
        options.restrict([_continuous(plausible=[True, True, True, False])])


def test_a_name_that_the_model_does_not_have_is_rejected() -> None:
    with pytest.raises(RecourseOptionError, match=r"does not have: \['age'\]"):
        RecourseOptions(immutable=["age"]).restrict([_continuous()])


def test_a_feature_that_is_both_immutable_and_allowed_is_rejected() -> None:
    options = RecourseOptions(immutable=["x"], allowed={"x": ValueRange(high=5)})
    with pytest.raises(RecourseOptionError, match=r"both immutable and in allowed, so the options disagree: \['x'\]"):
        options.restrict([_continuous()])


def test_the_options_put_the_allowed_values_in_words() -> None:
    options = RecourseOptions(
        allowed={"x": ValueRange(low=10, high=20, match="inside"), "f": Levels(["b", "a"])},
    )
    assert options.describe() == {
        "f": "one of a, b",
        "x": "from 10 to 20, bins that lie fully inside the range",
    }
    assert ValueRange().describe() == "from -inf to inf, bins that overlap the range"


def test_the_options_take_a_copy_of_the_allowed_values() -> None:
    allowed: dict[str, ValueRange | Levels] = {"x": ValueRange()}
    options = RecourseOptions(allowed=allowed)
    allowed["y"] = ValueRange()
    assert list(options.allowed) == ["x"]
