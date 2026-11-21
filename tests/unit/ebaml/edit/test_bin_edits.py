"""Tests for the edits that change the bins of a continuous feature."""

import copy
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EditError
from ebaml.edit.bin_edits import ExtendRange, Rebin, SplitBin
from ebaml.edit.edit_report import build_report
from ebaml.edit.edits import Edit, EditOutcome, SetScores
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, MissingBin, UnseenBin, WholeGraph
from tests.unit.ebaml.conftest import PandasFrame, frame_of_columns
from tests.unit.ebaml.edit.conftest import assert_consistent

type Model = ExplainableBoostingClassifier | ExplainableBoostingRegressor


@pytest.fixture(scope="module")
def shared_cuts_ebm() -> tuple[ExplainableBoostingClassifier, PandasFrame]:
    """Give a model whose pair term reads the same cuts as the main effects, and its data.

    Returns
    -------
    tuple[ExplainableBoostingClassifier, PandasFrame]
        The fitted model and the data.

    """
    random_generator = np.random.default_rng(7)
    data = frame_of_columns({"age": random_generator.normal(size=400), "score": random_generator.normal(size=400)})
    target = ((2 * data["age"] * data["score"] + random_generator.normal(size=400)) > 0).astype(int)
    model = ExplainableBoostingClassifier(
        interactions=1,
        random_state=0,
        outer_bags=2,
        max_bins=16,
        max_interaction_bins=16,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(data, target), data


def _run[M: ExplainableBoostingClassifier | ExplainableBoostingRegressor](edit: Edit, ebm: M) -> tuple[M, EditOutcome]:
    edited = copy.deepcopy(ebm)
    return edited, edit.run(edited)


def _cuts(ebm: Model, feature_index: int) -> list[float]:
    cuts: list[float] = np.asarray(ebm.bins_[feature_index][0], dtype=float).tolist()
    return cuts


def _assert_same_predictions(
    before: ExplainableBoostingClassifier,
    after: ExplainableBoostingClassifier,
    data: PandasFrame,
) -> None:
    assert np.allclose(before.decision_function(data), after.decision_function(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.allclose(before.predict_with_uncertainty(data), after.predict_with_uncertainty(data))


def test_split_bin_makes_two_bins_with_the_old_score_and_shares_the_weight_by_width(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    left, right = view.bin_edges()[4]
    at = left + (right - left) / 4
    edited, outcome = _run(SplitBin("score", at), edit_ebm)
    after = term_view(edited, "score")
    assert after.real_bin_count == view.real_bin_count + 1
    assert _cuts(edited, 1) == sorted([*_cuts(edit_ebm, 1), at])
    assert after.scores[5:7].tolist() == [view.scores[5]] * 2
    assert after.weights[5:7] == pytest.approx([view.weights[5] / 4, view.weights[5] * 3 / 4])
    assert after.weights.sum() == pytest.approx(view.weights.sum())
    assert outcome.positions == (5, 6)
    assert outcome.description.startswith("Split the bin ")
    _assert_same_predictions(edit_ebm, edited, edit_data)
    assert_consistent(edited, edit_data)


@settings(max_examples=25, deadline=None)
@given(place=st.floats(0.01, 0.99))
def test_a_split_anywhere_changes_no_prediction(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    place: float,
) -> None:
    low, high = term_view(edit_ebm, "age").bin_edges()[0][0], term_view(edit_ebm, "age").bin_edges()[-1][1]
    at = low + place * (high - low)
    if at in _cuts(edit_ebm, 0):
        return
    edited, _ = _run(SplitBin("age", at), edit_ebm)
    _assert_same_predictions(edit_ebm, edited, edit_data)


def test_a_split_lets_a_later_edit_change_one_part_only(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    left, right = view.bin_edges()[4]
    at = (left + right) / 2
    split, _ = _run(SplitBin("score", at), edit_ebm)
    edited, _ = _run(SetScores("score", Bins(5, 5), 3.0), split)
    values = edit_data["score"].to_numpy()
    moved = (values >= at) & (values < right)
    change: npt.NDArray[np.float64] = np.asarray(
        edited.decision_function(edit_data) - edit_ebm.decision_function(edit_data),
        dtype=float,
    )
    assert np.allclose(change[~moved], 0.0)
    assert np.all(np.abs(change[moved]) > 0)


def test_split_bin_refuses_an_edge_a_value_outside_the_range_and_a_feature_without_edges(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    edges = term_view(edit_ebm, "score").bin_edges()
    with pytest.raises(EditError, match="is already an edge of a bin"):
        SplitBin("score", edges[3][1]).run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match=r"is outside the range .* Extend the range first with ExtendRange"):
        SplitBin("score", edges[-1][1] + 1.0).run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match=r"SplitBin needs a continuous feature, .* but 'grade' is ordinal"):
        SplitBin("grade", 1.5).run(copy.deepcopy(edit_ebm))


def test_a_bin_edit_gives_the_main_effect_its_own_cuts_when_a_pair_term_shares_them(
    shared_cuts_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    ebm, data = shared_cuts_ebm
    assert len(ebm.bins_[0]) == 1
    view = term_view(ebm, "age")
    left, right = view.bin_edges()[3]
    edited, _ = _run(SplitBin("age", (left + right) / 2), ebm)
    assert len(edited.bins_[0]) == 2
    assert np.asarray(edited.bins_[0][1]).tolist() == _cuts(ebm, 0)
    assert np.array_equal(edited.term_scores_[2], ebm.term_scores_[2])
    _assert_same_predictions(ebm, edited, data)
    assert_consistent(edited, data)
    again, _ = _run(SplitBin("age", (left + right) / 4 + left / 2), edited)
    assert np.asarray(again.bins_[0][1]).tolist() == _cuts(ebm, 0)


def _reference_rebin(
    edges: list[float],
    scores: list[float],
    weights: list[float],
    bin_count: int,
) -> tuple[list[float], list[float]]:
    """Rebin with plain loops: each new bin takes each old bin's weight by the share of the width that it covers.

    Returns
    -------
    tuple[list[float], list[float]]
        The scores and the weights of the new bins.

    """
    width = (edges[-1] - edges[0]) / bin_count
    new_scores: list[float] = []
    new_weights: list[float] = []
    for new_bin in range(bin_count):
        new_left, new_right = edges[0] + new_bin * width, edges[0] + (new_bin + 1) * width
        taken: list[float] = []
        for old_left, old_right, weight in zip(edges, edges[1:], weights, strict=False):
            overlap = max(0.0, min(new_right, old_right) - max(new_left, old_left))
            taken.append(weight * overlap / (old_right - old_left))
        new_weights.append(sum(taken))
        new_scores.append(sum(share * score for share, score in zip(taken, scores, strict=True)) / sum(taken))
    return new_scores, new_weights


def test_rebin_makes_bins_of_equal_width_that_match_the_reference_loop(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "score")
    edges = [view.bin_edges()[1][0], *(right for _, right in view.bin_edges()[1:7])]
    expected_scores, expected_weights = _reference_rebin(
        edges,
        view.scores[2:8].tolist(),
        view.weights[2:8].tolist(),
        3,
    )
    edited, outcome = _run(Rebin("score", Bins(1, 6), 3), edit_ebm)
    after = term_view(edited, "score")
    assert after.real_bin_count == view.real_bin_count - 3
    assert [right - left for left, right in after.bin_edges()[1:4]] == pytest.approx([(edges[-1] - edges[0]) / 3] * 3)
    assert after.scores[2:5] == pytest.approx(expected_scores)
    assert after.weights[2:5] == pytest.approx(expected_weights)
    assert np.asarray(edited.standard_deviations_[1])[2:5].tolist() == [0.0] * 3
    assert after.scores[5:].tolist() == view.scores[8:].tolist()
    assert outcome.positions == (2, 3, 4)
    assert outcome.description.startswith("Replace 6 bins of score (")
    values = edit_data["score"].to_numpy()
    outside = (values < edges[0]) | (values >= edges[-1])
    assert np.allclose(edited.decision_function(edit_data)[outside], edit_ebm.decision_function(edit_data)[outside])
    assert_consistent(edited, edit_data)


def test_rebin_of_the_last_bins_describes_the_bins_that_it_replaced(edit_ebm: ExplainableBoostingClassifier) -> None:
    """Bug: the description read the bins after the edit, so a rebin of the last bins raised IndexError."""
    view = term_view(edit_ebm, "score")
    last = view.real_bin_count - 1
    edited, outcome = _run(Rebin("score", Bins(last - 5, last), 2), edit_ebm)
    right = view.bin_edges()[-1][1]
    assert outcome.description.startswith(
        f"Replace 6 bins of score ({view.label(last - 4).split(' to ')[0]} to {right:.6g})",
    )
    assert term_view(edited, "score").real_bin_count == view.real_bin_count - 4


def test_rebin_of_bins_with_no_weight_averages_the_scores_by_width(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    weights = np.asarray(ebm.bin_weights_[1], dtype=float)
    weights[2:4] = 0.0
    ebm.bin_weights_[1] = weights
    view = term_view(ebm, "score")
    edited, _ = _run(Rebin("score", Bins(1, 2), 1), ebm)
    left, middle = view.bin_edges()[1]
    right = view.bin_edges()[2][1]
    share = (middle - left) / (right - left)
    expected = share * view.scores[2] + (1 - share) * view.scores[3]
    assert term_view(edited, "score").scores[2] == pytest.approx(expected)


def test_rebin_refuses_outer_bins_a_count_below_one_and_a_feature_without_edges(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    with pytest.raises(EditError, match="Rebin needs real bins"):
        Rebin("age", MissingBin(), 2).run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match="Rebin needs a bin_count of 1 or more, but it is 0"):
        Rebin("score", WholeGraph(), 0).run(copy.deepcopy(edit_ebm))
    with pytest.raises(EditError, match="Rebin needs a continuous feature"):
        Rebin("group", Categories(["x"]), 2).run(copy.deepcopy(edit_ebm))


def test_extend_range_adds_an_empty_bin_with_the_outer_score_on_each_side_it_names(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    view = term_view(edit_ebm, "age")
    low, high = view.bin_edges()[0][0], view.bin_edges()[-1][1]
    edited, outcome = _run(ExtendRange("age", low - 1.0, high + 2.0), edit_ebm)
    after = term_view(edited, "age")
    assert after.real_bin_count == view.real_bin_count + 2
    assert after.bin_edges()[0] == (low - 1.0, low)
    assert after.bin_edges()[-1] == (high, high + 2.0)
    assert after.weights[[1, after.real_bin_count]].tolist() == [0.0, 0.0]
    assert after.scores[1] == view.scores[1]
    assert after.scores[after.real_bin_count] == view.scores[view.real_bin_count]
    assert outcome.positions == (1, after.real_bin_count)
    assert outcome.description.startswith("Add the bins ")
    _assert_same_predictions(edit_ebm, edited, edit_data)
    assert_consistent(edited, edit_data)
    upper, _ = _run(ExtendRange("age", high=high + 2.0), edit_ebm)
    assert term_view(upper, "age").bin_edges()[0][0] == low
    lower, outcome = _run(ExtendRange("age", low=low - 2.0), edit_ebm)
    assert term_view(lower, "age").bin_edges()[-1][1] == high
    assert outcome.positions == (1,)


def test_extend_range_lets_a_later_edit_score_the_values_beyond_the_old_bound(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    high = term_view(edit_ebm, "age").bin_edges()[-1][1]
    extended, _ = _run(ExtendRange("age", high=high + 10.0), edit_ebm)
    edited, _ = _run(
        SetScores(
            "age",
            Bins(term_view(extended, "age").real_bin_count - 1, term_view(extended, "age").real_bin_count - 1),
            0.0,
        ),
        extended,
    )
    far = edit_data.iloc[[30]].copy()
    far["age"] = high + 5.0
    far_scores = np.asarray([edit_ebm.decision_function(far)[0], edited.decision_function(far)[0]], dtype=float)
    assert not np.isclose(far_scores[0], far_scores[1])
    assert np.allclose(
        edited.decision_function(edit_data.iloc[30:40]),
        edit_ebm.decision_function(edit_data.iloc[30:40]),
    )


@pytest.mark.parametrize(("low", "high"), [(None, None), (0.0, None), (None, 0.0)])
def test_extend_range_needs_a_bound_outside_the_range(
    edit_ebm: ExplainableBoostingClassifier,
    low: float | None,
    high: float | None,
) -> None:
    with pytest.raises(EditError, match="ExtendRange needs a low bound below"):
        ExtendRange("age", low, high).run(copy.deepcopy(edit_ebm))


def test_the_report_of_a_split_shows_no_change_and_the_report_of_a_rebin_names_the_ranges(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    split_edit = SplitBin("score", float(np.mean(term_view(edit_ebm, "score").bin_edges()[4])))
    split, outcome = _run(split_edit, edit_ebm)
    report = build_report(edit_ebm, split, "SplitBin", outcome, "original")
    assert (report.bins_changed, report.largest_change, report.weight_share) == (0, 0.0, 0.0)
    rebinned, outcome = _run(Rebin("score", Bins(1, 6), 3), edit_ebm)
    report = build_report(edit_ebm, rebinned, "Rebin", outcome, "original")
    assert report.bins_changed > 0
    assert {change.position for change in report.changes} <= {2, 3, 4}
    assert all(" to " in change.label for change in report.changes)
    view = term_view(edit_ebm, "score")
    assert 0 < report.weight_share <= view.weights[2:8].sum() / view.weights.sum() + 1e-12
    assert report.covered_range is not None


def test_the_report_of_a_bin_edit_names_a_change_of_the_missing_bin_and_the_unseen_bin(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    rebinned, outcome = _run(Rebin("age", Bins(0, 3), 2), edit_ebm)
    SetScores("age", MissingBin(), 2.0).run(rebinned)
    SetScores("age", UnseenBin(), -2.0).run(rebinned)
    report = build_report(edit_ebm, rebinned, "Rebin", outcome, "original")
    labels = [change.label for change in report.changes]
    assert labels[0] == "missing"
    assert labels[-1] == "unseen"
    view = term_view(edit_ebm, "age")
    assert report.weight_share >= view.weights[0] / view.weights.sum()


def test_the_report_of_an_extension_and_a_score_of_the_new_bin_holds_no_training_weight(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    high = term_view(edit_ebm, "age").bin_edges()[-1][1]
    extended, outcome = _run(ExtendRange("age", high=high + 1.0), edit_ebm)
    report = build_report(edit_ebm, extended, "ExtendRange", outcome, "original")
    assert report.bins_changed == 0
    last = term_view(extended, "age").real_bin_count - 1
    edited, outcome = _run(SetScores("age", Bins(last, last), 1.0), extended)
    report = build_report(extended, edited, "SetScores", outcome, "original")
    assert (report.bins_changed, report.weight_share) == (1, 0.0)


def _mixed_edit_strategy() -> st.SearchStrategy[Edit]:
    bin_range = st.tuples(st.integers(0, 12), st.integers(0, 12)).map(lambda pair: Bins(min(pair), max(pair)))
    return st.one_of(
        st.builds(SplitBin, st.just("score"), st.floats(-2.5, 2.5)),
        st.builds(Rebin, st.just("score"), bin_range, st.integers(1, 5)),
        st.builds(ExtendRange, st.just("score"), st.none(), st.floats(5.0, 9.0)),
        st.builds(SetScores, st.just("score"), bin_range, st.floats(-2, 2)),
    )


@settings(max_examples=25, deadline=None)
@given(sequence=st.lists(_mixed_edit_strategy(), min_size=1, max_size=5))
def test_after_any_sequence_of_bin_edits_and_score_edits_the_model_stays_consistent(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    sequence: list[Edit],
) -> None:
    model = copy.deepcopy(edit_ebm)
    for edit in sequence:
        try:
            edit.run(model)
        except EditError:
            continue  # a random edit may name a bin or a value that the model does not have after the edits before it
        assert_consistent(model, edit_data)


def test_a_split_and_an_extension_of_a_multiclass_model_change_no_probability(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    for edit in (SplitBin("age", 0.01), ExtendRange("age", low=-10.0, high=10.0)):
        edited = copy.deepcopy(multiclass_edit_ebm)
        edit.run(edited)
        assert np.allclose(edited.predict_proba(edit_data), multiclass_edit_ebm.predict_proba(edit_data))
        assert_consistent(edited, edit_data)


def test_a_rebin_of_a_multiclass_model_mixes_the_scores_of_each_class(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edited = copy.deepcopy(multiclass_edit_ebm)
    outcome = Rebin("age", Bins(0, 5), 1).run(edited)
    before, after = term_view(multiclass_edit_ebm, "age"), term_view(edited, "age")
    mix = before.weights[1:7] / before.weights[1:7].sum()
    assert np.allclose(after.scores[1], mix @ before.scores[1:7])
    assert after.real_bin_count == before.real_bin_count - 5
    assert np.array_equal(np.asarray(edited.standard_deviations_[after.term])[1], np.zeros(3))
    assert np.allclose(np.asarray(edited.bagged_scores_[after.term])[:, 1], after.scores[1])
    assert outcome.positions == (1,)
    assert_consistent(edited, edit_data)
