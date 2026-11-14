"""Tests for the report of an edit."""

import copy

import numpy as np
import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.edit.edit_report import build_report
from ebaml.edit.edits import Edit, Recenter, RemoveTerms, SetScores, Shift
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, MissingBin, WholeGraph


def _report(edit: Edit, ebm: ExplainableBoostingClassifier):  # ruff: ignore[missing-return-type-private-function] -- the helper only builds the report that the tests read
    after = copy.deepcopy(ebm)
    outcome = edit.run(after)
    return build_report(ebm, after, type(edit).__name__, outcome, "original")


def test_the_report_lists_each_changed_bin_with_its_score_before_and_after(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    report = _report(Shift("score", Bins(2, 4), 0.5), edit_ebm)
    view = term_view(edit_ebm, "score")
    assert [change.position for change in report.changes] == [3, 4, 5]
    assert [change.label for change in report.changes] == [view.label(3), view.label(4), view.label(5)]
    assert [change.before for change in report.changes] == view.scores[3:6].tolist()
    assert [change.after for change in report.changes] == pytest.approx((view.scores[3:6] + 0.5).tolist())
    assert report.bins_changed == 3
    assert report.largest_change == pytest.approx(0.5)
    assert report.edit_type == "Shift"
    assert report.feature == "score"
    assert report.covered_range == view.range_text([3, 4, 5])


def test_the_weight_share_is_the_training_weight_of_the_changed_bins_over_the_total(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    report = _report(Shift("score", Bins(2, 4), 0.5), edit_ebm)
    view = term_view(edit_ebm, "score")
    assert report.weight_share == pytest.approx(view.weights[3:6].sum() / view.weights.sum())


def test_the_weighted_mean_shift_and_the_intercept_change_follow_the_edit(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    shifted = _report(Shift("score", WholeGraph(), 0.5), edit_ebm)
    view = term_view(edit_ebm, "score")
    real_weight = view.weights[1:-1].sum() / view.weights.sum()
    assert shifted.weighted_mean_shift == pytest.approx(0.5 * real_weight)
    assert shifted.intercept_change == pytest.approx(0.0, abs=0)
    centred = _report(Recenter("score"), edit_ebm)
    assert centred.weighted_mean_shift == pytest.approx(-float(np.average(view.scores, weights=view.weights)), abs=1e-9)
    assert centred.intercept_change == pytest.approx(float(np.average(view.scores, weights=view.weights)), abs=1e-9)


def test_an_edit_that_sets_the_score_that_a_bin_has_reports_no_change(edit_ebm: ExplainableBoostingClassifier) -> None:
    value = float(term_view(edit_ebm, "score").scores[3])
    report = _report(SetScores("score", Bins(2, 2), value), edit_ebm)
    assert report.bins_changed == 0
    assert report.largest_change == pytest.approx(0.0, abs=0)
    assert report.weight_share == pytest.approx(0.0, abs=0)
    assert "0 bins changed" in report.to_text()


def test_the_report_names_the_pair_terms_that_still_use_the_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    report = _report(Shift("age", MissingBin(), 1.0), edit_ebm)
    expected = tuple(name for name in edit_ebm.term_names_ if " & " in name and "age" in name.split(" & "))
    assert report.interactions_still_using == expected
    assert expected
    assert "still use the feature, and the edit does not change them" in report.to_text()


def test_the_report_holds_the_graph_text_before_and_after_for_a_continuous_and_a_nominal_feature(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    report = _report(Shift("score", WholeGraph(), 0.5), edit_ebm)
    assert report.graph_before
    assert report.graph_after
    assert report.graph_before != report.graph_after
    assert "This graph represents" not in report.graph_before
    nominal = _report(Shift("group", WholeGraph(), 0.5), edit_ebm)
    assert nominal.graph_before
    assert nominal.graph_after != nominal.graph_before


def test_the_report_has_no_graph_text_for_an_ordinal_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    report = _report(Shift("grade", WholeGraph(), 0.5), edit_ebm)
    assert report.graph_before is None
    assert report.graph_after is None
    assert report.bins_changed == 3


def test_the_report_of_a_graph_without_weight_has_no_share_and_no_mean_shift(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    ebm.bin_weights_[1] = np.zeros_like(ebm.bin_weights_[1])
    report = _report(Shift("score", Bins(0, 1), 0.5), ebm)
    assert report.weight_share == pytest.approx(0.0, abs=0)
    assert report.weighted_mean_shift == pytest.approx(0.0, abs=0)
    assert report.bins_changed == 2


def test_the_report_of_a_removal_names_the_terms_and_has_no_bins(edit_ebm: ExplainableBoostingClassifier) -> None:
    report = _report(RemoveTerms(["score"]), edit_ebm)
    assert report.removed_terms == ("score",)
    assert report.changes == ()
    assert report.graph_before is None
    assert report.to_text() == "Remove the terms 'score'.\nRemoved terms: score."


def test_the_text_of_a_long_report_shows_ten_bins_and_counts_the_rest(edit_ebm: ExplainableBoostingClassifier) -> None:
    report = _report(Shift("score", WholeGraph(), 0.5), edit_ebm)
    text = report.to_text()
    assert text.count("\n  ") == 11  # ten bins and the line that counts the rest
    assert f"and {report.bins_changed - 10} more bins" in text


def test_the_text_of_a_report_claims_no_more_than_its_numbers_show(edit_ebm: ExplainableBoostingClassifier) -> None:
    text = _report(Shift("score", Bins(2, 4), 0.5), edit_ebm).to_text().lower()
    for word in ("safer", "better", "minimal", "optimal", "improve"):
        assert word not in text


def test_the_text_of_a_report_of_one_bin_says_bin_and_not_bins(edit_ebm: ExplainableBoostingClassifier) -> None:
    assert "1 bin changed." in _report(Shift("age", MissingBin(), 1.0), edit_ebm).to_text()
