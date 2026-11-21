"""Tests for the business constraints and the check that finds their violations."""

import copy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ebaml.core.custom_types import MonotoneDirection
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EditError, InvalidOptionError, OptionRangeError, SelectionError, UnsupportedModelError
from ebaml.edit.constraints import (
    Bounded,
    Constraint,
    FixedScore,
    MaxStep,
    Monotone,
    Unused,
    check_constraints,
)
from ebaml.edit.edits import Clip, Edit, LimitSteps, Monotonize, RemoveFeature, SetScores, ZeroTerm
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, EveryBin, MissingBin, WholeGraph

type Model = ExplainableBoostingClassifier | ExplainableBoostingRegressor


def _repaired[M: ExplainableBoostingClassifier | ExplainableBoostingRegressor](ebm: M, edit: Edit) -> M:
    edited = copy.deepcopy(ebm)
    edit.run(edited)
    return edited


def _real_scores(ebm: Model, feature: str) -> np.ndarray:
    return term_view(ebm, feature).scores[1:-1]


def _pair_feature(ebm: ExplainableBoostingClassifier) -> str:
    pair = next(features for features in ebm.term_features_ if len(features) > 1)
    return ebm.feature_names_in_[pair[0]]


def test_monotone_names_the_bins_of_each_fall_and_the_largest_fall(edit_ebm: ExplainableBoostingClassifier) -> None:
    scores = _real_scores(edit_ebm, "score")
    falls = -np.diff(scores)
    (violation,) = check_constraints(edit_ebm, [Monotone("score", "increasing")])
    view = term_view(edit_ebm, "score")
    in_fall = sorted({bin_number + offset for bin_number in np.flatnonzero(falls > 0).tolist() for offset in (0, 1)})
    assert violation.bins == tuple(view.label(bin_number + 1) for bin_number in in_fall)
    assert violation.excess == pytest.approx(falls.max())
    assert violation.repair == Monotonize("score", "increasing", WholeGraph())
    assert violation.terms == ()
    assert violation.description.startswith(f"The graph of score falls between {int((falls > 0).sum())} pairs")


def test_monotone_decreasing_counts_the_rises(edit_ebm: ExplainableBoostingClassifier) -> None:
    rises = np.diff(_real_scores(edit_ebm, "score"))
    (violation,) = check_constraints(edit_ebm, [Monotone("score", "decreasing", Bins(0, 9))])
    assert violation.excess == pytest.approx(rises[:9].max())
    assert " rises between " in violation.description
    assert violation.repair == Monotonize("score", "decreasing", Bins(0, 9))


def test_monotone_of_one_bin_always_holds(edit_ebm: ExplainableBoostingClassifier) -> None:
    assert check_constraints(edit_ebm, [Monotone("score", "increasing", Bins(3, 3))]) == []


def test_monotone_refuses_a_feature_with_no_order_and_the_missing_bin(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(EditError, match="Monotone needs an order, but 'group' is nominal"):
        check_constraints(edit_ebm, [Monotone("group", "increasing")])
    with pytest.raises(EditError, match="Monotone needs real bins"):
        check_constraints(edit_ebm, [Monotone("age", "increasing", MissingBin())])


def test_max_step_names_each_step_above_the_limit(edit_ebm: ExplainableBoostingClassifier) -> None:
    steps = np.abs(np.diff(_real_scores(edit_ebm, "score")))
    limit = float(steps.max()) / 2
    (violation,) = check_constraints(edit_ebm, [MaxStep("score", limit)])
    assert violation.excess == pytest.approx(steps.max() - limit)
    assert violation.description.startswith(f"{int((steps > limit).sum())} steps of the graph of score")
    assert violation.repair == LimitSteps("score", limit, WholeGraph())
    assert check_constraints(edit_ebm, [MaxStep("score", float(steps.max()))]) == []


@pytest.mark.parametrize("step", [-0.1, float("inf"), float("nan")])
def test_max_step_refuses_a_step_that_is_negative_or_not_finite(
    edit_ebm: ExplainableBoostingClassifier,
    step: float,
) -> None:
    with pytest.raises(EditError, match="0 or more and finite"):
        check_constraints(edit_ebm, [MaxStep("score", step)])


def test_max_step_refuses_a_feature_with_no_order(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(EditError, match="MaxStep needs an order"):
        check_constraints(edit_ebm, [MaxStep("group", 0.1)])


def test_bounded_covers_the_missing_bin_by_default(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited = _repaired(edit_ebm, SetScores("age", MissingBin(), 50.0))
    (violation,) = check_constraints(edited, [Bounded("age", -10.0, 10.0)])
    assert violation.bins == ("missing",)
    assert violation.excess == pytest.approx(40.0)
    assert violation.repair == Clip("age", EveryBin(), -10.0, 10.0)
    assert violation.description == "1 scores of age are outside the range -10 to 10, by up to 40"


def test_bounded_names_the_bins_below_and_above_the_range(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "group")
    scores = view.scores[1:-1]
    low, high = float(np.sort(scores)[1]), float(np.sort(scores)[1])
    (violation,) = check_constraints(edit_ebm, [Bounded("group", low, high, WholeGraph())])
    assert violation.bins == tuple(name for name, score in zip(view.categories(), scores, strict=True) if score != low)


def test_bounded_refuses_a_low_bound_above_the_high_bound(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(EditError, match="low not above high"):
        check_constraints(edit_ebm, [Bounded("age", 1.0, -1.0)])


def test_fixed_score_names_the_bins_that_differ(edit_ebm: ExplainableBoostingClassifier) -> None:
    (violation,) = check_constraints(edit_ebm, [FixedScore("group", Categories(["x", "z"]), 0.0)])
    view = term_view(edit_ebm, "group")
    assert violation.bins == ("x", "z")
    assert violation.excess == pytest.approx(np.abs(view.scores[[1, 3]]).max())
    assert violation.repair == SetScores("group", Categories(["x", "z"]), 0.0)


def test_a_difference_at_the_tolerance_is_not_a_violation(edit_ebm: ExplainableBoostingClassifier) -> None:
    missing_score = float(term_view(edit_ebm, "age").scores[0])
    near = FixedScore("age", MissingBin(), missing_score + 1e-12)
    assert check_constraints(edit_ebm, [near]) == []
    assert len(check_constraints(edit_ebm, [near], tolerance=0.0)) == 1
    assert check_constraints(edit_ebm, [FixedScore("age", MissingBin(), missing_score + 0.5)], tolerance=0.5) == []


def test_unused_names_each_term_of_the_feature_and_zero_term_leaves_the_pair_terms(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    feature = _pair_feature(edit_ebm)
    (violation,) = check_constraints(edit_ebm, [Unused(feature)])
    using = tuple(name for name in edit_ebm.term_names_ if feature in name.split(" & "))
    assert violation.terms == using
    assert violation.bins == ()
    assert violation.repair == RemoveFeature(feature)
    zeroed = _repaired(edit_ebm, ZeroTerm(feature))
    (after_zero,) = check_constraints(zeroed, [Unused(feature)])
    assert feature not in after_zero.terms
    assert after_zero.terms == tuple(name for name in using if name != feature)


def test_unused_holds_for_a_feature_whose_terms_score_zero(edit_regressor: ExplainableBoostingRegressor) -> None:
    assert check_constraints(_repaired(edit_regressor, ZeroTerm("age")), [Unused("age")]) == []


def test_unused_refuses_an_unknown_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(SelectionError, match="The model has no feature 'height'"):
        check_constraints(edit_ebm, [Unused("height")])


def _breaking_constraints(ebm: ExplainableBoostingClassifier) -> list[Constraint]:
    steps = np.abs(np.diff(_real_scores(ebm, "score")))
    return [
        Monotone("score", "increasing"),
        Monotone("score", "decreasing", Bins(2, 10)),
        Monotone("grade", "decreasing"),
        MaxStep("score", float(steps.max()) / 4),
        Bounded("age", -0.1, 0.1),
        FixedScore("age", MissingBin(), 0.0),
        FixedScore("group", Categories(["y"]), 0.25),
        Unused(_pair_feature(ebm)),
    ]


def test_the_violations_follow_the_order_of_the_constraints_and_the_check_changes_nothing(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    before = copy.deepcopy(edit_ebm)
    constraints = _breaking_constraints(edit_ebm)
    violations = check_constraints(edit_ebm, iter(constraints))
    assert [violation.constraint for violation in violations] == constraints
    for term_before, term_after in zip(before.term_scores_, edit_ebm.term_scores_, strict=True):
        assert np.array_equal(term_before, term_after)


def test_the_repair_of_each_violation_makes_the_constraint_hold(edit_ebm: ExplainableBoostingClassifier) -> None:
    for constraint in _breaking_constraints(edit_ebm):
        (violation,) = check_constraints(edit_ebm, [constraint])
        assert check_constraints(_repaired(edit_ebm, violation.repair), [constraint]) == [], constraint


@settings(max_examples=20, deadline=None)
@given(
    first=st.integers(0, 15),
    width=st.integers(1, 10),
    step=st.floats(0, 0.5),
    bound=st.floats(0, 0.5),
)
def test_the_repair_of_a_constraint_on_any_range_makes_it_hold(
    edit_ebm: ExplainableBoostingClassifier,
    first: int,
    width: int,
    step: float,
    bound: float,
) -> None:
    last = min(first + width, term_view(edit_ebm, "score").real_bin_count - 1)
    for constraint in (
        Monotone("score", "increasing", Bins(first, last)),
        MaxStep("score", step, Bins(first, last)),
        Bounded("score", -bound, bound),
    ):
        repaired = _repaired(edit_ebm, constraint.repair())
        assert check_constraints(repaired, [constraint]) == [], constraint


@pytest.mark.parametrize("tolerance", [-1e-9, float("inf"), float("nan")])
def test_the_check_refuses_a_tolerance_that_is_negative_or_not_finite(
    edit_ebm: ExplainableBoostingClassifier,
    tolerance: float,
) -> None:
    with pytest.raises(OptionRangeError, match="The tolerance must be 0 or more and finite"):
        check_constraints(edit_ebm, [], tolerance=tolerance)


def test_the_check_refuses_a_multiclass_model(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(UnsupportedModelError):
        check_constraints(multiclass_ebm, [Unused("a")])


def test_monotone_refuses_auto_because_a_constraint_needs_a_direction() -> None:
    with pytest.raises(InvalidOptionError, match="Invalid option: auto"):
        Monotone("score", "auto")


def test_monotone_keeps_its_direction_as_a_member_of_the_enum() -> None:
    assert Monotone("score", "increasing").direction is MonotoneDirection.INCREASING
