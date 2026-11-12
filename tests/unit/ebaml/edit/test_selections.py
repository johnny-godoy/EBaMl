"""Tests for the selections of bins."""

import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import SelectionError
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, MissingBin, UnseenBin, ValueInterval, WholeGraph


def test_bins_count_from_the_first_real_bin_and_include_the_last(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "age")
    assert Bins(0, 0).resolve(view) == (1,)
    assert Bins(2, 4).resolve(view) == (3, 4, 5)
    assert Bins(0, view.real_bin_count - 1).resolve(view) == WholeGraph().resolve(view)


def test_bins_of_an_ordinal_feature_use_the_bin_numbers(edit_ebm: ExplainableBoostingClassifier) -> None:
    assert Bins(1, 2).resolve(term_view(edit_ebm, "grade")) == (2, 3)


@pytest.mark.parametrize(("first", "last"), [(-1, 3), (3, 2), (0, 10_000)])
def test_bins_outside_the_range_name_the_valid_numbers(
    edit_ebm: ExplainableBoostingClassifier,
    first: int,
    last: int,
) -> None:
    view = term_view(edit_ebm, "age")
    with pytest.raises(SelectionError, match=rf"The bin numbers go from 0 to {view.real_bin_count - 1}"):
        Bins(first, last).resolve(view)


def test_bins_refuse_a_nominal_feature(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(SelectionError, match="Use Categories"):
        Bins(0, 1).resolve(term_view(edit_ebm, "group"))


def test_categories_resolve_to_the_positions_of_the_names_in_increasing_order(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "group")
    known = view.categories()
    assert Categories([known[2], known[0]]).resolve(view) == (1, 3)
    assert Categories([known[1], known[1]]).names == (known[1],)


def test_categories_name_the_categories_that_exist_when_one_is_unknown(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(SelectionError, match=r"no category 'w'. The categories are"):
        Categories(["w"]).resolve(term_view(edit_ebm, "group"))


def test_categories_refuse_an_empty_list_and_a_feature_that_is_not_nominal(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    with pytest.raises(SelectionError, match="at least one category"):
        Categories([]).resolve(term_view(edit_ebm, "group"))
    with pytest.raises(SelectionError, match="Use Bins"):
        Categories(["low"]).resolve(term_view(edit_ebm, "grade"))


def test_the_missing_bin_and_the_unseen_bin_are_the_first_and_the_last_row(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "age")
    assert MissingBin().resolve(view) == (0,)
    assert UnseenBin().resolve(view) == (view.real_bin_count + 1,)


def test_the_whole_graph_holds_the_real_bins_only(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "group")
    assert WholeGraph().resolve(view) == (1, 2, 3)


def test_an_interval_selects_every_bin_that_overlaps_it(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "score")
    edges = view.bin_edges()
    inside = ValueInterval((edges[3][0] + edges[3][1]) / 2, (edges[5][0] + edges[5][1]) / 2)
    assert inside.resolve(view) == (4, 5, 6)


def test_an_interval_that_ends_on_a_bin_edge_leaves_out_the_bin_that_starts_there(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "score")
    edges = view.bin_edges()
    # the right edge of bin 3 is the left edge of bin 4: a bin holds its left edge, so bin 3 does not hold the edge
    assert ValueInterval(edges[3][1], edges[5][0]).resolve(view) == (5, 6)


def test_an_open_interval_reaches_the_ends_and_the_last_bin_holds_its_right_edge(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "score")
    assert ValueInterval().resolve(view) == WholeGraph().resolve(view)
    assert ValueInterval(low=view.bin_edges()[-1][1]).resolve(view) == (view.real_bin_count,)
    assert ValueInterval(high=view.bin_edges()[0][0]).resolve(view) == (1,)


def test_an_interval_outside_the_values_names_the_range_of_the_values(edit_ebm: ExplainableBoostingClassifier) -> None:
    view = term_view(edit_ebm, "score")
    with pytest.raises(
        SelectionError,
        match=r"No bin of 'score' overlaps the interval from 1000.0 to 2000.0. The values go from",
    ):
        ValueInterval(1000.0, 2000.0).resolve(view)


def test_an_interval_refuses_a_low_bound_above_the_high_bound_and_a_bound_that_is_not_a_number(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    view = term_view(edit_ebm, "score")
    with pytest.raises(SelectionError, match="low must not be above high"):
        ValueInterval(2.0, 1.0).resolve(view)
    with pytest.raises(SelectionError, match="needs numbers"):
        ValueInterval(float("nan"), 1.0).resolve(view)


def test_an_interval_refuses_a_feature_that_is_not_continuous(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(SelectionError, match="ValueInterval needs a continuous feature"):
        ValueInterval(0.0, 1.0).resolve(term_view(edit_ebm, "grade"))
