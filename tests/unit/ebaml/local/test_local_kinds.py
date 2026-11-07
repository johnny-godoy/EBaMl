"""Tests for the kinds of model that the local explanations read."""

import math

import pytest

from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.graphs.model_kinds import BinaryClassifier, MulticlassClassifier, Regressor
from ebaml.local.decisions import Interval, Margin, Side, Threshold
from ebaml.local.local_kinds import BinaryLocal, MulticlassLocal, Outcome, OutputKind, RegressorLocal, local_kind
from tests.unit.ebaml.conftest import UnknownKind

Z95 = 1.959963984540054


def test_a_binary_classifier_keeps_its_predicted_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame([0.5], [0.0], intercepts=[0.1], confidence_level=0.0, decision=None)
    assert frame.side == Side("yes", (Margin(1, 0.0, "no"),))
    assert frame.output == pytest.approx(1 / (1 + math.exp(-0.5)))
    assert (frame.lower, frame.upper) == pytest.approx((frame.output, frame.output))


def test_a_logit_of_zero_belongs_to_the_negative_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame([0.0], [0.0], intercepts=[0.0], confidence_level=0.0, decision=None)
    assert frame.side == Side("no", (Margin(-1, 0.0, "yes"),))


def test_the_probability_interval_of_a_binary_classifier_is_ordered() -> None:
    frame = BinaryLocal(("no", "yes")).frame([-1.0], [0.5], intercepts=[0.0], confidence_level=0.95, decision=None)
    assert frame.lower == pytest.approx(1 - 1 / (1 + math.exp(-(-1.0 + Z95 * 0.5))))
    assert frame.upper == pytest.approx(1 - 1 / (1 + math.exp(-(-1.0 - Z95 * 0.5))))


def test_the_probability_after_the_changes_is_that_of_the_other_class() -> None:
    frame = BinaryLocal(("no", "yes")).frame([0.5], [0.0], intercepts=[0.0], confidence_level=0.0, decision=None)
    # the log-odds of "yes" after the changes is -1.5, so "no" has the probability sigmoid(1.5)
    assert frame.after(frame.side.margins[0], [-1.5]).output == pytest.approx(1 / (1 + math.exp(-1.5)))


def test_a_binary_classifier_rejects_a_decision() -> None:
    with pytest.raises(OptionRangeError, match="Only a regressor accepts a decision"):
        BinaryLocal(("no", "yes")).frame([0.5], [0.0], intercepts=[0.0], confidence_level=0.0, decision=Threshold(0.0))


def test_a_regressor_asks_by_default_whether_the_value_is_above_the_intercept() -> None:
    frame = RegressorLocal().frame([12.0], [0.0], intercepts=[10.0], confidence_level=0.0, decision=None)
    assert frame.side == Side("above 10", (Margin(1, 10.0, "at most 10"),))
    assert frame.output == pytest.approx(12.0)


def test_a_regressor_interval_is_the_difference_of_the_value_and_the_confidence() -> None:
    frame = RegressorLocal().frame([12.0], [0.5], intercepts=[10.0], confidence_level=0.95, decision=Interval(1.0, 5.0))
    assert frame.side == Side("above 5", (Margin(1, 5.0, "at most 5"),))
    assert (frame.lower, frame.upper) == pytest.approx((12.0 - Z95 * 0.5, 12.0 + Z95 * 0.5))


@pytest.mark.parametrize(("value", "value_after"), [(12.0, 9.0), (3.0, 11.0)])
def test_the_value_after_the_changes_is_the_new_score(value: float, value_after: float) -> None:
    frame = RegressorLocal().frame([value], [0.0], intercepts=[10.0], confidence_level=0.0, decision=None)
    outcome = frame.after(frame.side.margins[0], [value_after])
    assert outcome == Outcome(frame.side.margins[0].opposite, value_after)


def test_a_multiclass_classifier_keeps_its_predicted_class_against_each_rival() -> None:
    kind = MulticlassLocal(("a", "b", "c"))
    frame = kind.frame([0.0, 2.0, 1.0], [0.0] * 3, intercepts=[0.0] * 3, confidence_level=0.0, decision=None)
    assert frame.side == Side("b", (Margin(1, 0.0, "a", (1, 0)), Margin(1, 0.0, "c", (1, 2))))
    assert frame.column == 1
    assert frame.output == pytest.approx(math.exp(2) / (1 + math.exp(1) + math.exp(2)))
    assert frame.side.binding([0.0, 2.0, 1.0]).opposite == "c"


def test_the_first_class_wins_a_tie_between_classes() -> None:
    frame = MulticlassLocal(("a", "b", "c")).frame(
        [1.0, 1.0, 0.0],
        [0.0] * 3,
        intercepts=[0.0] * 3,
        confidence_level=0.0,
        decision=None,
    )
    assert frame.side.label == "a"


def test_the_probability_interval_of_a_multiclass_classifier_moves_every_score_against_the_class() -> None:
    kind = MulticlassLocal(("a", "b", "c"))
    scores, stds = [2.0, 0.5, 0.0], [0.3, 0.2, 0.1]
    frame = kind.frame(scores, stds, intercepts=[0.0] * 3, confidence_level=0.95, decision=None)
    low = [2.0 - Z95 * 0.3, 0.5 + Z95 * 0.2, 0.0 + Z95 * 0.1]
    high = [2.0 + Z95 * 0.3, 0.5 - Z95 * 0.2, 0.0 - Z95 * 0.1]
    assert frame.lower == pytest.approx(math.exp(low[0]) / sum(math.exp(score) for score in low))
    assert frame.upper == pytest.approx(math.exp(high[0]) / sum(math.exp(score) for score in high))


def test_the_class_after_the_changes_is_the_class_with_the_highest_score() -> None:
    kind = MulticlassLocal(("a", "b", "c"))
    frame = kind.frame([2.0, 0.5, 0.0], [0.0] * 3, intercepts=[0.0] * 3, confidence_level=0.0, decision=None)
    outcome = frame.after(frame.side.margins[0], [0.0, 0.5, 1.5])
    assert outcome.decision == "c"
    assert outcome.output == pytest.approx(math.exp(1.5) / (1 + math.exp(0.5) + math.exp(1.5)))


def test_a_multiclass_classifier_rejects_a_decision() -> None:
    with pytest.raises(OptionRangeError, match="Only a regressor accepts a decision"):
        MulticlassLocal(("a", "b", "c")).frame(
            [0.0, 1.0, 2.0],
            [0.0] * 3,
            intercepts=[0.0] * 3,
            confidence_level=0.0,
            decision=Threshold(0.0),
        )


def test_the_output_kind_of_each_kind() -> None:
    assert BinaryLocal(("no", "yes")).output_kind is OutputKind.PROBABILITY
    assert RegressorLocal().output_kind is OutputKind.VALUE
    assert MulticlassLocal(("a", "b", "c")).output_kind is OutputKind.CLASS_PROBABILITY


def test_local_kind_selects_the_class_of_the_model_kind() -> None:
    assert local_kind(BinaryClassifier(("no", "yes"))) == BinaryLocal(("no", "yes"))
    assert local_kind(Regressor()) == RegressorLocal()


def test_local_kind_selects_the_class_of_a_multiclass_classifier() -> None:
    assert local_kind(MulticlassClassifier(("a", "b", "c"))) == MulticlassLocal(("a", "b", "c"))


def test_local_kind_rejects_a_kind_without_local_explanations() -> None:
    with pytest.raises(UnsupportedModelError, match=r"classifier or a regressor\. Got a UnknownKind"):
        local_kind(UnknownKind())
