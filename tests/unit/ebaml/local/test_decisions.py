"""Tests for the decisions of a regressor."""

import math

import pytest

from ebaml.core.exceptions import OptionRangeError
from ebaml.local.decisions import Interval, Margin, Side, Threshold


def test_a_value_above_the_threshold_holds_the_high_side() -> None:
    assert Threshold(10).side(10.5) == Side("above 10", (Margin(1, 10.0, "at most 10"),))


@pytest.mark.parametrize("value", [10.0, 9.0])
def test_a_value_on_or_below_the_threshold_holds_the_low_side(value: float) -> None:
    assert Threshold(10).side(value) == Side("at most 10", (Margin(-1, 10.0, "above 10"),))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_threshold_is_finite(value: float) -> None:
    with pytest.raises(OptionRangeError, match="needs a finite number"):
        Threshold(value)


@pytest.mark.parametrize("value", [9.9, 10.0])
def test_a_value_on_or_below_the_low_end_holds_the_claim_at_most(value: float) -> None:
    assert Interval(10, 20).side(value) == Side("at most 10", (Margin(-1, 10.0, "above 10"),))


def test_a_value_above_the_high_end_holds_the_claim_above() -> None:
    assert Interval(10, 20).side(20.1) == Side("above 20", (Margin(1, 20.0, "at most 20"),))


@pytest.mark.parametrize("value", [10.1, 15.0, 20.0])
def test_a_value_inside_the_interval_has_a_margin_for_each_end(value: float) -> None:
    assert Interval(10, 20).side(value) == Side(
        "above 10 and at most 20",
        (Margin(1, 10.0, "at most 10"), Margin(-1, 20.0, "above 20")),
    )


def test_the_margin_is_the_signed_distance_from_its_threshold() -> None:
    assert Margin(1, 10.0, "x").value(12.5) == pytest.approx(2.5)
    assert Margin(-1, 10.0, "x").value(12.5) == pytest.approx(-2.5)


def test_the_binding_margin_is_the_one_closest_to_failing() -> None:
    side = Interval(10, 20).side(18.0)
    assert side.binding(18.0) == side.margins[1]
    assert side.binding(11.0) == side.margins[0]


@pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (2.0, 1.0)])
def test_an_interval_needs_low_below_high(low: float, high: float) -> None:
    with pytest.raises(OptionRangeError, match="needs low < high"):
        Interval(low, high)


@pytest.mark.parametrize(("low", "high"), [(-math.inf, 1.0), (0.0, math.inf), (math.nan, 1.0)])
def test_the_ends_of_an_interval_are_finite(low: float, high: float) -> None:
    with pytest.raises(OptionRangeError, match="needs a finite number"):
        Interval(low, high)
