"""Tests for the decisions of a regressor."""

import math

import pytest

from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.local.decisions import Interval, Side, Threshold


def test_a_value_above_the_threshold_holds_the_high_side() -> None:
    assert Threshold(10).side(10.5) == Side("above 10", "at most 10", 1, 10.0)


@pytest.mark.parametrize("value", [10.0, 9.0])
def test_a_value_on_or_below_the_threshold_holds_the_low_side(value: float) -> None:
    assert Threshold(10).side(value) == Side("at most 10", "above 10", -1, 10.0)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_threshold_is_finite(value: float) -> None:
    with pytest.raises(OptionRangeError, match="needs a finite number"):
        Threshold(value)


def test_a_value_below_the_interval_holds_the_claim_below() -> None:
    assert Interval(10, 20).side(9.9) == Side("below 10", "at least 10", -1, 10.0)


def test_a_value_above_the_interval_holds_the_claim_above() -> None:
    assert Interval(10, 20).side(20.1) == Side("above 20", "at most 20", 1, 20.0)


@pytest.mark.parametrize("value", [10.0, 15.0, 20.0])
def test_a_value_inside_the_interval_is_not_supported_yet(value: float) -> None:
    with pytest.raises(UnsupportedModelError, match=r"inside the interval \[10, 20\]"):
        Interval(10, 20).side(value)


@pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (2.0, 1.0)])
def test_an_interval_needs_low_below_high(low: float, high: float) -> None:
    with pytest.raises(OptionRangeError, match="needs low < high"):
        Interval(low, high)


@pytest.mark.parametrize(("low", "high"), [(-math.inf, 1.0), (0.0, math.inf), (math.nan, 1.0)])
def test_the_ends_of_an_interval_are_finite(low: float, high: float) -> None:
    with pytest.raises(OptionRangeError, match="needs a finite number"):
        Interval(low, high)
