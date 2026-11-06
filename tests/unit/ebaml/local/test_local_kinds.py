"""Tests for the kinds of model that the local explanations read."""

import math

import pytest

from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.graphs.model_kinds import BinaryClassifier, MulticlassClassifier, Regressor
from ebaml.local.decisions import Interval, Side, Threshold
from ebaml.local.local_kinds import BinaryLocal, OutputKind, RegressorLocal, local_kind

Z95 = 1.959963984540054


def test_a_binary_classifier_keeps_its_predicted_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame(0.5, 0.0, intercept=0.1, confidence_level=0.0, decision=None)
    assert frame.side == Side("yes", "no", 1, 0.0)
    assert frame.output == pytest.approx(1 / (1 + math.exp(-0.5)))
    assert (frame.lower, frame.upper) == pytest.approx((frame.output, frame.output))


def test_a_logit_of_zero_belongs_to_the_negative_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame(0.0, 0.0, intercept=0.0, confidence_level=0.0, decision=None)
    assert frame.side == Side("no", "yes", -1, 0.0)


def test_the_probability_interval_of_a_binary_classifier_is_ordered() -> None:
    frame = BinaryLocal(("no", "yes")).frame(-1.0, 0.5, intercept=0.0, confidence_level=0.95, decision=None)
    assert frame.lower == pytest.approx(1 - 1 / (1 + math.exp(-(-1.0 + Z95 * 0.5))))
    assert frame.upper == pytest.approx(1 - 1 / (1 + math.exp(-(-1.0 - Z95 * 0.5))))


def test_the_probability_after_the_changes_is_that_of_the_other_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame(0.5, 0.0, intercept=0.0, confidence_level=0.0, decision=None)
    # the margin after the changes is -1.5, so the log-odds of "yes" is -1.5 and "no" has the probability sigmoid(1.5)
    assert frame.output_after(-1.5) == pytest.approx(1 / (1 + math.exp(-1.5)))


def test_a_binary_classifier_rejects_a_decision() -> None:
    with pytest.raises(OptionRangeError, match="Only a regressor accepts a decision"):
        BinaryLocal(("no", "yes")).frame(0.5, 0.0, intercept=0.0, confidence_level=0.0, decision=Threshold(0.0))


def test_a_regressor_asks_by_default_whether_the_value_is_above_the_intercept() -> None:
    frame = RegressorLocal().frame(12.0, 0.0, intercept=10.0, confidence_level=0.0, decision=None)
    assert frame.side == Side("above 10", "at most 10", 1, 10.0)
    assert frame.output == pytest.approx(12.0)


def test_a_regressor_interval_is_the_difference_of_the_value_and_the_confidence() -> None:
    frame = RegressorLocal().frame(12.0, 0.5, intercept=10.0, confidence_level=0.95, decision=Interval(1.0, 5.0))
    assert frame.side == Side("above 5", "at most 5", 1, 5.0)
    assert (frame.lower, frame.upper) == pytest.approx((12.0 - Z95 * 0.5, 12.0 + Z95 * 0.5))


@pytest.mark.parametrize(("value", "margin_after"), [(12.0, -1.0), (3.0, -2.5)])
def test_the_value_after_the_changes_is_the_threshold_plus_the_margin(value: float, margin_after: float) -> None:
    frame = RegressorLocal().frame(value, 0.0, intercept=10.0, confidence_level=0.0, decision=None)
    assert frame.output_after(margin_after) == pytest.approx(10.0 + frame.side.sign * margin_after)


def test_the_output_kind_of_each_kind() -> None:
    assert BinaryLocal(("no", "yes")).output_kind is OutputKind.PROBABILITY
    assert RegressorLocal().output_kind is OutputKind.VALUE


def test_local_kind_selects_the_class_of_the_model_kind() -> None:
    assert local_kind(BinaryClassifier(("no", "yes"))) == BinaryLocal(("no", "yes"))
    assert local_kind(Regressor()) == RegressorLocal()


def test_local_kind_rejects_a_multiclass_classifier() -> None:
    with pytest.raises(UnsupportedModelError, match=r"binary classifier or a regressor\. Got a MulticlassClassifier"):
        local_kind(MulticlassClassifier(("a", "b", "c")))
