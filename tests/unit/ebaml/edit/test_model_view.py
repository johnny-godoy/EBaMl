"""Tests for the view of the main-effect term of a feature."""

import itertools

import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import SelectionError, UnsupportedModelError
from ebaml.edit.model_view import interaction_names, term_view


def test_the_view_of_a_continuous_feature_counts_the_real_bins_without_the_missing_and_unseen_bin(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "age")
    assert view.real_bin_count == len(edit_ebm.term_scores_[0]) - 2
    assert view.unseen_position == view.real_bin_count + 1
    assert view.ordered
    assert len(view.scores) == len(view.weights) == view.real_bin_count + 2


def test_a_name_that_the_model_lacks_lists_the_features(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(
        SelectionError,
        match=r"no feature 'height'. The features are \['age', 'score', 'group', 'grade'\]",
    ):
        term_view(edit_ebm, "height")


def test_a_feature_without_a_main_term_is_unsupported(edit_ebm: ExplainableBoostingClassifier) -> None:
    ebm = edit_ebm.copy()
    ebm.remove_terms(["age"])
    with pytest.raises(UnsupportedModelError, match="no main-effect term"):
        term_view(ebm, "age")


def test_the_edges_of_a_continuous_feature_start_and_end_at_the_bounds(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "score")
    edges = view.bin_edges()
    assert len(edges) == view.real_bin_count
    assert edges[0][0] == pytest.approx(edit_ebm.feature_bounds_[1][0])
    assert edges[-1][1] == pytest.approx(edit_ebm.feature_bounds_[1][1])
    assert all(right == next_left for (_, right), (next_left, _) in itertools.pairwise(edges))
    assert view.left_edges().tolist() == [left for left, _ in edges]


def test_an_ordinal_feature_places_its_bins_by_number_and_a_nominal_feature_has_no_places(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    assert term_view(edit_ebm, "grade").left_edges().tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(SelectionError, match="'group' is nominal, so its bins have no order"):
        term_view(edit_ebm, "group").left_edges()


def test_the_categories_follow_the_order_of_the_bins(edit_ebm: ExplainableBoostingClassifier) -> None:
    assert term_view(edit_ebm, "grade").categories() == ["low", "mid", "high"]
    assert sorted(term_view(edit_ebm, "group").categories()) == ["x", "y", "z"]
    with pytest.raises(SelectionError, match="'age' is continuous, so its bins have no categories"):
        term_view(edit_ebm, "age").categories()


def test_a_label_names_the_missing_bin_the_unseen_bin_an_interval_and_a_category(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    age = term_view(edit_ebm, "age")
    assert age.label(0) == "missing"
    assert age.label(age.unseen_position) == "unseen"
    left, right = age.bin_edges()[0]
    assert age.label(1) == f"{left:.6g} to {right:.6g}"
    assert term_view(edit_ebm, "grade").label(2) == "mid"


def test_is_real_rejects_the_missing_bin_and_the_unseen_bin(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "grade")
    assert view.is_real([1, 2, 3])
    assert not view.is_real([0])
    assert not view.is_real([view.unseen_position])


def test_the_text_of_a_group_of_bins_names_the_count_and_the_range(edit_ebm: ExplainableBoostingClassifier) -> None:
    age = term_view(edit_ebm, "age")
    left = f"{age.bin_edges()[2][0]:.6g}"
    right = f"{age.bin_edges()[4][1]:.6g}"
    assert age.describe_positions([3, 4, 5]) == f"3 bins of age ({left} to {right})"
    assert age.describe_positions([3]).startswith("1 bin of age (")
    assert age.describe_positions([0]) == "the missing bin of age"
    assert age.describe_positions([age.unseen_position]) == "the unseen bin of age"


def test_the_range_of_categories_lists_the_names_and_shortens_a_long_list(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    grade = term_view(edit_ebm, "grade")
    assert grade.range_text([1, 3]) == "low, high"
    age = term_view(edit_ebm, "age")
    assert age.range_text([0, 1]) == "missing, " + age.label(1)
    long = list(range(8))
    assert age.range_text(long) == f"missing to {age.label(7)}"


def test_interaction_names_lists_the_pair_terms_of_a_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    pair_names = [name for name in edit_ebm.term_names_ if " & " in name]
    assert pair_names
    for name in pair_names:
        for feature in name.split(" & "):
            assert name in interaction_names(edit_ebm, feature)
    assert interaction_names(edit_ebm, "age") == tuple(name for name in pair_names if "age" in name.split(" & "))
