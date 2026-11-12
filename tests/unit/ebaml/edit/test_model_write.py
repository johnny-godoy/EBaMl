"""Tests for the functions that write the fitted attributes of an EBM."""

import copy

import numpy as np
import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import EditError, UnsupportedModelError
from ebaml.edit import model_write
from tests.unit.ebaml.conftest import PandasFrame
from tests.unit.ebaml.edit.conftest import assert_consistent, strip_uncertainty


def test_a_multiclass_model_is_not_editable(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(UnsupportedModelError, match="binary classifiers and regressors"):
        model_write.check_editable(multiclass_ebm)


def test_set_scores_writes_the_value_in_every_bag_and_zeroes_the_deviation(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = edit_ebm.copy()
    model_write.set_scores(ebm, 0, [0, 3], [0.25, -0.5])
    assert ebm.term_scores_[0][[0, 3]].tolist() == [0.25, -0.5]
    assert np.all(ebm.bagged_scores_[0][:, [0, 3]] == [0.25, -0.5])
    assert ebm.standard_deviations_[0][[0, 3]].tolist() == [0.0, 0.0]
    assert_consistent(ebm, edit_data)


def test_set_scores_changes_no_other_score_and_no_other_term(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = edit_ebm.copy()
    model_write.set_scores(ebm, 0, [3], 1.0)
    untouched = np.delete(np.arange(len(edit_ebm.term_scores_[0])), 3)
    assert np.array_equal(ebm.term_scores_[0][untouched], edit_ebm.term_scores_[0][untouched])
    assert np.array_equal(ebm.bagged_scores_[0][:, untouched], edit_ebm.bagged_scores_[0][:, untouched])
    assert np.array_equal(ebm.standard_deviations_[0][untouched], edit_ebm.standard_deviations_[0][untouched])
    for scores, original in list(zip(ebm.term_scores_, edit_ebm.term_scores_, strict=True))[1:]:
        assert np.array_equal(scores, original)
    assert np.array_equal(ebm.intercept_, edit_ebm.intercept_)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_set_scores_refuses_a_value_that_is_not_finite(edit_ebm: ExplainableBoostingClassifier, bad: float) -> None:
    ebm = edit_ebm.copy()
    with pytest.raises(EditError, match="finite scores"):
        model_write.set_scores(ebm, 0, [1], bad)
    assert np.array_equal(ebm.term_scores_[0], edit_ebm.term_scores_[0])


def test_shift_term_moves_the_scores_the_bags_and_the_intercept_and_keeps_the_deviations(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = edit_ebm.copy()
    model_write.shift_term(ebm, 1, -0.5, intercept_shift=0.5)
    assert np.allclose(ebm.term_scores_[1], edit_ebm.term_scores_[1] - 0.5)
    assert np.allclose(ebm.bagged_scores_[1], edit_ebm.bagged_scores_[1] - 0.5)
    assert np.array_equal(ebm.standard_deviations_[1], edit_ebm.standard_deviations_[1])
    assert np.allclose(ebm.intercept_, edit_ebm.intercept_ + 0.5)
    assert np.allclose(ebm.bagged_intercept_, edit_ebm.bagged_intercept_ + 0.5)
    assert np.allclose(ebm.predict_proba(edit_data), edit_ebm.predict_proba(edit_data))
    assert_consistent(ebm, edit_data)


def test_shift_term_without_an_intercept_shift_leaves_the_intercept(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = edit_ebm.copy()
    model_write.shift_term(ebm, 1, 0.25, intercept_shift=0.0)
    assert np.array_equal(ebm.intercept_, edit_ebm.intercept_)
    assert np.array_equal(ebm.bagged_intercept_, edit_ebm.bagged_intercept_)


def test_shift_term_refuses_a_value_that_is_not_finite(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(EditError, match="finite scores"):
        model_write.shift_term(edit_ebm.copy(), 1, 0.0, intercept_shift=float("nan"))


@pytest.mark.parametrize("factor", [2.0, -0.5, 0.0])
def test_scale_term_multiplies_the_scores_and_the_deviation_by_the_absolute_factor(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    factor: float,
) -> None:
    ebm = edit_ebm.copy()
    model_write.scale_term(ebm, 1, factor)
    assert np.allclose(ebm.term_scores_[1], edit_ebm.term_scores_[1] * factor)
    assert np.allclose(ebm.bagged_scores_[1], edit_ebm.bagged_scores_[1] * factor)
    assert np.allclose(ebm.standard_deviations_[1], edit_ebm.standard_deviations_[1] * abs(factor))
    assert np.array_equal(ebm.intercept_, edit_ebm.intercept_)
    assert_consistent(ebm, edit_data)


def test_scale_term_refuses_a_factor_that_is_not_finite(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(EditError, match="finite scores"):
        model_write.scale_term(edit_ebm.copy(), 1, float("inf"))


def test_repair_uncertainty_fills_what_interpret_leaves_as_none(
    interpret_monotonized: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(interpret_monotonized)
    assert ebm.bagged_scores_[1] is None
    assert ebm.standard_deviations_[1] is None
    assert ebm.bagged_intercept_ is None
    model_write.repair_uncertainty(ebm)
    bag_count = ebm.bagged_scores_[0].shape[0]
    assert ebm.bagged_scores_[1].shape == (bag_count, len(ebm.term_scores_[1]))
    assert np.array_equal(ebm.bagged_scores_[1][0], ebm.term_scores_[1])
    assert not ebm.standard_deviations_[1].any()
    assert ebm.bagged_intercept_.tolist() == [float(ebm.intercept_[0])] * bag_count
    assert_consistent(ebm, edit_data)


def test_repair_uncertainty_reads_the_bag_count_of_the_intercept_when_every_term_is_none(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    strip_uncertainty(ebm, terms=[term for term, _ in enumerate(ebm.term_scores_)], intercept=False)
    model_write.repair_uncertainty(ebm)
    assert ebm.bagged_scores_[0].shape[0] == len(edit_ebm.bagged_intercept_)


def test_repair_uncertainty_counts_one_bag_when_nothing_says_otherwise(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    strip_uncertainty(ebm, terms=[term for term, _ in enumerate(ebm.term_scores_)], intercept=True)
    model_write.repair_uncertainty(ebm)
    assert ebm.bagged_scores_[0].shape[0] == 1
    assert ebm.bagged_intercept_.shape == (1,)


def test_repair_uncertainty_changes_nothing_in_a_complete_model(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    model_write.repair_uncertainty(ebm)
    for term, bags in enumerate(edit_ebm.bagged_scores_):
        assert np.array_equal(ebm.bagged_scores_[term], bags)
        assert np.array_equal(ebm.standard_deviations_[term], edit_ebm.standard_deviations_[term])


def test_a_write_on_a_monotonized_model_repairs_it_first(
    interpret_monotonized: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(interpret_monotonized)
    model_write.set_scores(ebm, 0, [2], 0.1)
    assert_consistent(ebm, edit_data)


def test_remove_terms_keeps_the_input_columns_and_repairs_the_model(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(edit_ebm)
    model_write.remove_terms(ebm, ["score"])
    assert "score" not in ebm.term_names_
    assert list(ebm.feature_names_in_) == list(edit_ebm.feature_names_in_)
    assert ebm.predict_proba(edit_data).shape == (len(edit_data), 2)
    assert_consistent(ebm, edit_data)
