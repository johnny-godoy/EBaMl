"""Tests for the edits of a graph."""

import copy

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EditError, SelectionError, UnsupportedModelError
from ebaml.edit import edits
from ebaml.edit.edits import (
    Align,
    AlignTarget,
    Direction,
    Edit,
    FitLine,
    Interpolate,
    Monotonize,
    Recenter,
    RemoveFeature,
    RemoveTerms,
    Scale,
    SetScores,
    Shift,
    ZeroTerm,
)
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, MissingBin, Selection, UnseenBin, ValueInterval, WholeGraph
from tests.unit.ebaml.conftest import PandasFrame
from tests.unit.ebaml.edit.conftest import (
    assert_consistent,
    reference_interpolate,
    reference_isotonic,
    reference_regression,
    reference_term_scores,
)

type Model = ExplainableBoostingClassifier | ExplainableBoostingRegressor


def _run[M: ExplainableBoostingClassifier | ExplainableBoostingRegressor](
    edit: Edit,
    ebm: M,
) -> tuple[M, edits.EditOutcome]:
    edited = copy.deepcopy(ebm)
    return edited, edit.run(edited)


def _changed(before: Model, after: Model, name: str) -> list[int]:
    return np.flatnonzero(
        np.array(reference_term_scores(before, name)) != np.array(reference_term_scores(after, name)),
    ).tolist()


def test_shift_adds_the_value_to_the_selected_scores_only(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, outcome = _run(Shift("score", Bins(2, 4), 0.5), edit_ebm)
    before = reference_term_scores(edit_ebm, "score")
    after = reference_term_scores(edited, "score")
    assert outcome.positions == (3, 4, 5)
    assert after[3:6] == pytest.approx([value + 0.5 for value in before[3:6]])
    assert _changed(edit_ebm, edited, "score") == [3, 4, 5]
    assert outcome.description.startswith("Add 0.5 to the scores of 3 bins of score (")
    assert_consistent(edited, edit_data)


def test_shift_of_the_missing_bin_moves_the_prediction_of_a_missing_value_only(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, _ = _run(Shift("age", MissingBin(), 1.0), edit_ebm)
    missing = edit_data["age"].isna().to_numpy()
    scores_before = edit_ebm.decision_function(edit_data)
    scores_after = edited.decision_function(edit_data)
    assert np.allclose(scores_after[missing] - scores_before[missing], 1.0)
    assert np.allclose(scores_after[~missing], scores_before[~missing])


def test_set_scores_sets_one_value_and_the_text_names_it(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, outcome = _run(SetScores("age", MissingBin(), 0.0), edit_ebm)
    assert reference_term_scores(edited, "age")[0] == pytest.approx(0.0, abs=0)
    assert outcome.description == "Set the missing bin of age to 0"
    assert_consistent(edited, edit_data)
    edited, outcome = _run(SetScores("group", Categories(["x", "z"]), 0.38), edit_ebm)
    view = term_view(edited, "group")
    assert view.scores[[1, 3]].tolist() == [0.38, 0.38] or len(set(view.scores[[1, 3]].tolist())) == 1
    assert "to 0.38" in outcome.description


def test_set_scores_of_the_unseen_bin_changes_that_row_only(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited, outcome = _run(SetScores("score", UnseenBin(), 0.7), edit_ebm)
    view = term_view(edited, "score")
    assert outcome.positions == (view.unseen_position,)
    assert _changed(edit_ebm, edited, "score") == [view.unseen_position]


def test_an_edit_does_not_recenter_so_a_direct_edit_moves_the_weighted_mean_and_the_intercept_stays(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    edited, _ = _run(SetScores("score", WholeGraph(), 0.38), edit_ebm)
    assert set(term_view(edited, "score").scores[1:-1]) == {0.38}
    assert np.array_equal(edited.intercept_, edit_ebm.intercept_)


@pytest.mark.parametrize(("to", "neighbour_position"), [("left", 5), ("right", 10)])
def test_align_to_a_neighbour_copies_the_score_of_the_bin_outside_the_selection(
    edit_ebm: ExplainableBoostingClassifier,
    to: AlignTarget,
    neighbour_position: int,
) -> None:
    # the selection holds the bin numbers 5 to 8, which are the positions 6 to 9, so the neighbours are 5 and 10
    edited, outcome = _run(Align("score", Bins(5, 8), to), edit_ebm)
    neighbour = reference_term_scores(edit_ebm, "score")[neighbour_position]
    assert set(reference_term_scores(edited, "score")[6:10]) == {neighbour}
    assert outcome.description.startswith("Align 4 bins of score")
    assert f"to the {to}" in outcome.description


def test_align_to_the_mean_uses_the_training_weight(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited, _ = _run(Align("score", Bins(2, 5), "mean"), edit_ebm)
    view = term_view(edit_ebm, "score")
    expected = np.average(view.scores[3:7], weights=view.weights[3:7])
    assert set(term_view(edited, "score").scores[3:7].round(12)) == {round(float(expected), 12)}


def test_align_to_the_mean_counts_each_bin_once_when_no_selected_bin_has_weight(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1] = np.zeros_like(ebm.bin_weights_[1])
    edited, _ = _run(Align("score", Bins(2, 5), "mean"), ebm)
    assert term_view(edited, "score").scores[3:7].tolist() == pytest.approx(
        [term_view(ebm, "score").scores[3:7].mean()] * 4,
    )


def test_align_to_the_mean_works_on_a_nominal_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited, _ = _run(Align("group", WholeGraph(), "mean"), edit_ebm)
    assert len(set(term_view(edited, "group").scores[1:4].round(12))) == 1


def test_align_to_the_left_of_the_first_bin_and_to_the_right_of_the_last_bin_fail(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    last = term_view(edit_ebm, "score").real_bin_count - 1
    with pytest.raises(EditError, match="no bin on the left in 'score'"):
        Align("score", Bins(0, 2), "left").run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match="no bin on the right in 'score'"):
        Align("score", Bins(last - 2, last), "right").run(copy.deepcopy(edit_ebm))


def test_align_to_a_neighbour_needs_real_bins_of_a_feature_with_an_order(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    with pytest.raises(EditError, match="needs real bins of a feature with an order"):
        Align("group", Categories(["x"]), "left").run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match="needs real bins of a feature with an order"):
        Align("age", MissingBin(), "right").run(copy.deepcopy(edit_ebm))


def test_align_refuses_a_target_that_is_not_listed() -> None:
    with pytest.raises(BeartypeCallHintViolation, match=r"parameter to=.middle."):
        Align("score", WholeGraph(), "middle")  # pyrefly: ignore[bad-argument-type] -- the call passes a wrong literal on purpose


def test_monotonize_a_range_matches_the_reference_loop_and_leaves_the_rest(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    selection = slice(10, 41)
    expected = reference_isotonic(view.scores[1:-1][selection].tolist(), view.weights[1:-1][selection].tolist())
    edited, outcome = _run(Monotonize("score", "increasing", Bins(10, 40)), edit_ebm)
    after = term_view(edited, "score").scores
    assert after[1:-1][selection].tolist() == pytest.approx(expected)
    assert np.all(np.diff(after[1:-1][selection]) >= -1e-12)
    assert np.array_equal(after[:11], view.scores[:11])
    assert np.array_equal(after[42:], view.scores[42:])
    assert outcome.description.startswith("Make 31 bins of score")
    assert_consistent(edited, edit_data)


def test_monotonize_decreasing_matches_the_reference_loop(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "score")
    expected = reference_isotonic(view.scores[1:-1].tolist(), view.weights[1:-1].tolist(), increasing=False)
    edited, _ = _run(Monotonize("score", "decreasing"), edit_ebm)
    assert term_view(edited, "score").scores[1:-1].tolist() == pytest.approx(expected)


def test_monotonize_the_whole_graph_gives_the_scores_of_interpret_and_keeps_the_weighted_mean(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    reference = copy.deepcopy(edit_ebm)
    reference.monotonize(1, increasing=True)
    edited, _ = _run(Monotonize("score", "increasing"), edit_ebm)
    view = term_view(edit_ebm, "score")
    assert edited.term_scores_[1].tolist() == pytest.approx(reference.term_scores_[1].tolist())
    mean_before = np.average(view.scores, weights=view.weights)
    mean_after = np.average(edited.term_scores_[1], weights=view.weights)
    assert mean_after == pytest.approx(mean_before, abs=1e-9)


def test_monotonize_auto_picks_the_direction_with_the_smaller_change(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "age")
    increasing, _ = _run(Monotonize("age", "increasing"), edit_ebm)
    decreasing, _ = _run(Monotonize("age", "decreasing"), edit_ebm)
    auto, outcome = _run(Monotonize("age", "auto"), edit_ebm)

    def cost(model: Model) -> float:
        difference = term_view(model, "age").scores[1:-1] - view.scores[1:-1]
        return float(np.sum(view.weights[1:-1] * difference**2))

    chosen = increasing if cost(increasing) <= cost(decreasing) else decreasing
    assert auto.term_scores_[0].tolist() == chosen.term_scores_[0].tolist()
    assert outcome.description.endswith("increasing" if chosen is increasing else "decreasing")


def test_monotonize_gives_a_monotone_range_on_an_ordinal_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited, _ = _run(Monotonize("grade", "decreasing"), edit_ebm)
    scores = term_view(edited, "grade").scores[1:-1]
    assert np.all(np.diff(scores) <= 1e-12)


def test_monotonize_a_bin_with_no_training_weight_still_gives_a_monotone_range(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1][5] = 0.0
    edited, _ = _run(Monotonize("score", "increasing"), ebm)
    assert np.all(np.diff(edited.term_scores_[1][1:-1]) >= -1e-9)


def test_monotonize_with_no_weight_in_the_selection_counts_each_bin_once(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1] = np.zeros_like(ebm.bin_weights_[1])
    edited, _ = _run(Monotonize("score", "increasing", Bins(0, 9)), ebm)
    assert np.all(np.diff(edited.term_scores_[1][1:11]) >= -1e-9)


def test_monotonize_a_selection_of_one_bin_changes_nothing(edit_ebm: ExplainableBoostingClassifier) -> None:
    edited, _ = _run(Monotonize("score", "increasing", Bins(3, 3)), edit_ebm)
    assert edited.term_scores_[1].tolist() == pytest.approx(edit_ebm.term_scores_[1].tolist())


def test_monotonize_refuses_a_direction_that_is_not_listed() -> None:
    with pytest.raises(BeartypeCallHintViolation, match=r"parameter direction=.sideways."):
        Monotonize("score", "sideways")  # pyrefly: ignore[bad-argument-type] -- the call passes a wrong literal on purpose


@pytest.mark.parametrize(
    "edit",
    [Monotonize("group", "increasing"), Interpolate("group", WholeGraph()), FitLine("group", WholeGraph())],
    ids=lambda edit: type(edit).__name__,
)
def test_an_edit_that_needs_an_order_refuses_a_nominal_feature(
    edit_ebm: ExplainableBoostingClassifier,
    edit: Edit,
) -> None:
    with pytest.raises(EditError, match="needs an order, but 'group' is nominal"):
        edit.run(copy.deepcopy(edit_ebm))


@pytest.mark.parametrize("selection", [MissingBin(), UnseenBin()])
def test_an_edit_that_needs_an_order_refuses_the_missing_bin_and_the_unseen_bin(
    edit_ebm: ExplainableBoostingClassifier,
    selection: Selection,
) -> None:
    with pytest.raises(EditError, match="needs real bins"):
        Monotonize("age", "increasing", selection).run(copy.deepcopy(edit_ebm))


@pytest.mark.parametrize(
    "edit",
    [Interpolate("score", Bins(3, 3)), FitLine("score", Bins(3, 3))],
    ids=lambda edit: type(edit).__name__,
)
def test_a_line_edit_needs_two_bins(edit_ebm: ExplainableBoostingClassifier, edit: Edit) -> None:
    with pytest.raises(EditError, match="at least 2 bins of 'score', and the selection holds 1"):
        edit.run(copy.deepcopy(edit_ebm))


def test_interpolate_draws_the_line_between_the_first_and_the_last_score_at_the_left_edges(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    places = view.left_edges()[3:12].tolist()
    expected = reference_interpolate(places, view.scores[4:13].tolist())
    edited, _ = _run(Interpolate("score", Bins(3, 11)), edit_ebm)
    after = term_view(edited, "score").scores
    assert after[4:13].tolist() == pytest.approx(expected)
    assert after[4] == view.scores[4]
    assert after[12] == pytest.approx(view.scores[12])
    assert _changed(edit_ebm, edited, "score") == [
        index for index in range(5, 12) if view.scores[index] != after[index]
    ]
    assert_consistent(edited, edit_data)


def test_interpolate_an_ordinal_feature_places_the_bins_by_number(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "grade")
    edited, _ = _run(Interpolate("grade", WholeGraph()), edit_ebm)
    expected = reference_interpolate([0.0, 1.0, 2.0], view.scores[1:-1].tolist())
    assert term_view(edited, "grade").scores[1:-1].tolist() == pytest.approx(expected)


def test_fit_line_matches_the_reference_weighted_regression(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    places = view.left_edges()[5:25].tolist()
    expected = reference_regression(places, view.scores[6:26].tolist(), view.weights[6:26].tolist())
    edited, outcome = _run(FitLine("score", ValueInterval(view.bin_edges()[5][0], view.bin_edges()[24][0])), edit_ebm)
    assert term_view(edited, "score").scores[6:26].tolist() == pytest.approx(expected)
    assert outcome.description.startswith("Replace ")
    assert_consistent(edited, edit_data)


def test_fit_line_with_no_weight_in_the_selection_counts_each_bin_once(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1] = np.zeros_like(ebm.bin_weights_[1])
    view = term_view(ebm, "score")
    expected = reference_regression(view.left_edges()[2:8].tolist(), view.scores[3:9].tolist(), [1.0] * 6)
    edited, _ = _run(FitLine("score", Bins(2, 7)), ebm)
    assert term_view(edited, "score").scores[3:9].tolist() == pytest.approx(expected)


def test_scale_multiplies_the_graph_and_keeps_the_intercept(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    reference = copy.deepcopy(edit_ebm)
    reference.scale(1, 2.0)
    edited, outcome = _run(Scale("score", 2.0), edit_ebm)
    assert edited.term_scores_[1].tolist() == pytest.approx(reference.term_scores_[1].tolist())
    assert edited.standard_deviations_[1].tolist() == pytest.approx(reference.standard_deviations_[1].tolist())
    assert outcome.description == "Multiply the graph of score by 2"
    assert len(outcome.positions) == len(edit_ebm.term_scores_[1])
    assert_consistent(edited, edit_data)


def test_zero_term_sets_every_row_to_zero_and_keeps_the_term(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, outcome = _run(ZeroTerm("score"), edit_ebm)
    assert not edited.term_scores_[1].any()
    assert not edited.standard_deviations_[1].any()
    assert "score" in edited.term_names_
    assert outcome.description == "Set every score of score to 0"
    assert_consistent(edited, edit_data)


def test_recenter_moves_the_weighted_mean_into_the_intercept_and_changes_no_prediction(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    shifted, _ = _run(Shift("score", WholeGraph(), 0.8), edit_ebm)
    edited, outcome = _run(Recenter("score"), shifted)
    view = term_view(edited, "score")
    assert np.average(view.scores, weights=view.weights) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(edited.decision_function(edit_data), shifted.decision_function(edit_data))
    assert np.allclose(edited.predict_proba(edit_data), shifted.predict_proba(edit_data))
    assert not np.isclose(edited.intercept_[0], shifted.intercept_[0])
    assert outcome.description.startswith("Move the weighted mean of score (")
    assert_consistent(edited, edit_data)


def test_recenter_refuses_a_graph_with_no_training_weight(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1] = np.zeros_like(ebm.bin_weights_[1])
    with pytest.raises(EditError, match="No bin of 'score' has training weight"):
        Recenter("score").run(ebm)


def test_remove_terms_removes_the_named_terms_and_keeps_the_input_columns(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, outcome = _run(RemoveTerms(["score"]), edit_ebm)
    assert "score" not in edited.term_names_
    assert outcome.removed_terms == ("score",)
    assert outcome.feature is None
    assert outcome.description == "Remove the terms 'score'"
    assert_consistent(edited, edit_data)


def test_remove_terms_names_the_terms_that_exist_when_one_is_unknown(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(SelectionError, match=r"no term 'height'. The terms are \['age'"):
        RemoveTerms(["height"]).run(copy.deepcopy(edit_ebm))
    with pytest.raises(SelectionError, match="at least one term"):
        RemoveTerms([]).run(copy.deepcopy(edit_ebm))


def test_remove_feature_removes_every_term_that_uses_the_feature_pair_terms_included(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, outcome = _run(RemoveFeature("age"), edit_ebm)
    ages = [name for name in edit_ebm.term_names_ if "age" in name.split(" & ")]
    assert len(ages) > 1
    assert outcome.removed_terms == tuple(ages)
    assert not any("age" in name.split(" & ") for name in edited.term_names_)
    assert list(edited.feature_names_in_) == list(edit_ebm.feature_names_in_)
    assert edited.predict_proba(edit_data).shape == (len(edit_data), 2)
    assert_consistent(edited, edit_data)


def test_remove_feature_names_the_features_when_the_feature_is_unknown_or_unused(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    with pytest.raises(SelectionError, match="no feature 'height'"):
        RemoveFeature("height").run(copy.deepcopy(edit_ebm))
    ebm = copy.deepcopy(edit_ebm)
    ebm.remove_terms([name for name in ebm.term_names_ if "grade" in name.split(" & ")])
    with pytest.raises(SelectionError, match="No term of the model uses the feature 'grade'"):
        RemoveFeature("grade").run(ebm)


def test_every_edit_refuses_a_multiclass_model(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    for edit in (Shift("a", WholeGraph(), 1.0), ZeroTerm("a"), Scale("a", 2.0), RemoveFeature("a"), RemoveTerms(["a"])):
        with pytest.raises(UnsupportedModelError):
            edit.run(copy.deepcopy(multiclass_ebm))


def test_an_edit_works_on_a_regressor(edit_regressor: ExplainableBoostingRegressor, edit_data: PandasFrame) -> None:
    edited, _ = _run(Monotonize("score", "increasing"), edit_regressor)
    assert_consistent(edited, edit_data)
    shifted, _ = _run(Shift("age", WholeGraph(), 0.8), edit_regressor)
    recentred, _ = _run(Recenter("age"), shifted)
    assert np.allclose(recentred.predict(edit_data), shifted.predict(edit_data))
    assert not np.isclose(recentred.intercept_, shifted.intercept_)
    assert_consistent(recentred, edit_data)


def test_an_edit_on_a_model_that_interpret_monotonized_repairs_it(
    interpret_monotonized: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited, _ = _run(Shift("age", Bins(0, 3), 0.3), interpret_monotonized)
    assert_consistent(edited, edit_data)


def test_an_edit_leaves_the_model_that_it_started_from_to_the_caller(edit_ebm: ExplainableBoostingClassifier) -> None:
    # the session copies the model before it runs an edit, so the edit may write into the model that it receives
    before = copy.deepcopy(edit_ebm)
    edited, _ = _run(Shift("score", WholeGraph(), 1.0), edit_ebm)
    assert edited.term_scores_[1].tolist() != edit_ebm.term_scores_[1].tolist()
    assert edit_ebm.term_scores_[1].tolist() == before.term_scores_[1].tolist()


DIRECTIONS: tuple[Direction, ...] = ("increasing", "decreasing", "auto")


def _edit_strategy() -> st.SearchStrategy[Edit]:
    bin_range = st.tuples(st.integers(0, 15), st.integers(0, 15)).map(lambda pair: Bins(min(pair), max(pair)))
    value = st.floats(-2, 2, allow_nan=False)
    return st.one_of(
        st.builds(Shift, st.just("score"), bin_range, value),
        st.builds(SetScores, st.just("age"), st.sampled_from([MissingBin(), UnseenBin(), WholeGraph()]), value),
        st.builds(Monotonize, st.just("score"), st.sampled_from(DIRECTIONS), bin_range),
        st.builds(Interpolate, st.just("score"), bin_range.filter(lambda selection: selection.first < selection.last)),
        st.builds(FitLine, st.just("score"), bin_range.filter(lambda selection: selection.first < selection.last)),
        st.builds(Scale, st.just("group"), st.floats(-3, 3, allow_nan=False)),
        st.builds(Align, st.just("grade"), st.just(WholeGraph()), st.just("mean")),
        st.builds(Recenter, st.sampled_from(["age", "score", "group"])),
        st.builds(ZeroTerm, st.just("grade")),
    )


@settings(max_examples=25, deadline=None)
@given(sequence=st.lists(_edit_strategy(), min_size=1, max_size=4))
def test_after_any_sequence_of_edits_the_model_stays_consistent(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    sequence: list[Edit],
) -> None:
    model = copy.deepcopy(edit_ebm)
    for edit in sequence:
        edit.run(model)
        assert_consistent(model, edit_data)
