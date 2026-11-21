"""Tests for the functions that write the fitted attributes of an EBM."""

import copy

import numpy as np
import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import EditError, UnsupportedModelError
from ebaml.edit import model_write
from tests.unit.ebaml.conftest import PandasFrame
from tests.unit.ebaml.edit.conftest import assert_consistent, strip_uncertainty


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


def test_replace_bins_refuses_new_bins_whose_attributes_do_not_fit_their_number(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    replacement = model_write.BinReplacement(
        first=2,
        last=2,
        inner_cuts=[0.1],
        scores=[0.0, 0.0],
        weights=[1.0],
        deviations=[0.0, 0.0],
        bags=np.zeros((4, 2)),
    )
    with pytest.raises(EditError, match="must give 2 values for each attribute of the new bins"):
        model_write.replace_bins(copy.deepcopy(edit_ebm), 1, 1, replacement)


def test_every_kind_of_model_is_editable(
    edit_ebm: ExplainableBoostingClassifier,
    multiclass_edit_ebm: ExplainableBoostingClassifier,
) -> None:
    model_write.check_editable(edit_ebm)
    model_write.check_editable(multiclass_edit_ebm)


def test_a_model_with_an_unknown_link_is_not_editable(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(edit_ebm)
    vars(ebm)["link_"] = "probit"
    with pytest.raises(UnsupportedModelError, match="EBaMl reads the links"):
        model_write.check_editable(ebm)


def test_set_scores_of_one_class_writes_that_column_only(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    model_write.set_scores(ebm, 0, [0, 3], [0.25, -0.5], column=1)
    assert ebm.term_scores_[0][[0, 3], 1].tolist() == [0.25, -0.5]
    assert np.all(ebm.bagged_scores_[0][:, [0, 3], 1] == [0.25, -0.5])
    assert ebm.standard_deviations_[0][[0, 3], 1].tolist() == [0.0, 0.0]
    others = [0, 2]
    assert np.array_equal(ebm.term_scores_[0][:, others], multiclass_edit_ebm.term_scores_[0][:, others])
    assert np.array_equal(ebm.bagged_scores_[0][:, :, others], multiclass_edit_ebm.bagged_scores_[0][:, :, others])
    assert np.array_equal(
        ebm.standard_deviations_[0][:, others],
        multiclass_edit_ebm.standard_deviations_[0][:, others],
    )
    assert_consistent(ebm, edit_data)


def test_set_scores_without_a_column_writes_every_class(multiclass_edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    model_write.set_scores(ebm, 0, [2], 0.0)
    assert ebm.term_scores_[0][2].tolist() == [0.0, 0.0, 0.0]
    assert not np.any(ebm.bagged_scores_[0][:, 2])
    assert ebm.standard_deviations_[0][2].tolist() == [0.0, 0.0, 0.0]


def test_shift_term_of_one_class_moves_that_column_and_the_intercept_of_that_class(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    model_write.shift_term(ebm, 1, -0.5, intercept_shift=0.5, column=2)
    expected = np.asarray(multiclass_edit_ebm.term_scores_[1]).copy()
    expected[:, 2] -= 0.5
    assert np.allclose(ebm.term_scores_[1], expected)
    assert np.allclose(ebm.bagged_scores_[1][:, :, 2], multiclass_edit_ebm.bagged_scores_[1][:, :, 2] - 0.5)
    assert np.array_equal(ebm.bagged_scores_[1][:, :, :2], multiclass_edit_ebm.bagged_scores_[1][:, :, :2])
    third_class_only = np.array([0.0, 0.0, 0.5])
    assert np.allclose(ebm.intercept_, np.asarray(multiclass_edit_ebm.intercept_) + third_class_only)
    assert np.allclose(ebm.bagged_intercept_, np.asarray(multiclass_edit_ebm.bagged_intercept_) + third_class_only)
    assert np.allclose(ebm.predict_proba(edit_data), multiclass_edit_ebm.predict_proba(edit_data))


def test_shift_term_of_every_class_moves_every_intercept(multiclass_edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    model_write.shift_term(ebm, 1, 0.25, intercept_shift=-0.25)
    assert np.allclose(ebm.term_scores_[1], np.asarray(multiclass_edit_ebm.term_scores_[1]) + 0.25)
    assert np.allclose(ebm.intercept_, np.asarray(multiclass_edit_ebm.intercept_) - 0.25)


def test_scale_term_of_one_class_scales_that_column_only(multiclass_edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    model_write.scale_term(ebm, 0, -2.0, column=0)
    original = multiclass_edit_ebm
    assert np.allclose(ebm.term_scores_[0][:, 0], -2.0 * original.term_scores_[0][:, 0])
    assert np.allclose(ebm.bagged_scores_[0][:, :, 0], -2.0 * original.bagged_scores_[0][:, :, 0])
    assert np.allclose(ebm.standard_deviations_[0][:, 0], 2.0 * original.standard_deviations_[0][:, 0])
    assert np.array_equal(ebm.term_scores_[0][:, 1:], original.term_scores_[0][:, 1:])
    assert np.array_equal(ebm.standard_deviations_[0][:, 1:], original.standard_deviations_[0][:, 1:])


def test_repair_uncertainty_gives_a_multiclass_model_one_bagged_intercept_per_class(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    strip_uncertainty(ebm, terms=[0], intercept=True)
    model_write.repair_uncertainty(ebm)
    assert np.shape(ebm.bagged_intercept_) == (3, 3)
    assert np.array_equal(ebm.bagged_intercept_, np.tile(ebm.intercept_, (3, 1)))
    assert np.shape(ebm.bagged_scores_[0]) == (3, *np.shape(ebm.term_scores_[0]))


def test_replace_bins_splices_every_class_of_a_multiclass_model(
    multiclass_edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    ebm = copy.deepcopy(multiclass_edit_ebm)
    term = 1
    scores = np.asarray(ebm.term_scores_[term])
    replacement = model_write.BinReplacement(
        first=2,
        last=3,
        inner_cuts=[],
        scores=[[1.0, 2.0, 3.0]],
        weights=[5.0],
        deviations=[[0.0, 0.0, 0.0]],
        bags=np.full((3, 1, 3), [1.0, 2.0, 3.0]),
    )
    model_write.replace_bins(ebm, 1, term, replacement)
    assert np.array_equal(ebm.term_scores_[term], np.vstack((scores[:2], [[1.0, 2.0, 3.0]], scores[4:])))
    assert np.array_equal(ebm.bagged_scores_[term][:, 2], np.full((3, 3), [1.0, 2.0, 3.0]))
    assert ebm.bin_weights_[term][2] == pytest.approx(5.0)
    assert len(ebm.bins_[1][0]) == len(multiclass_edit_ebm.bins_[1][0]) - 1
    assert_consistent(ebm, edit_data)
