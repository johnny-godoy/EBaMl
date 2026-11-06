"""Tests for the search of a sufficient reason with several margins."""

import itertools
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ebaml.local.reason_search import ExactOrMinimal, Minimal, Selection, by_smallest_gain

MARGIN_COUNT = 3
FEATURE_COUNT = 5
GAIN: st.SearchStrategy[float] = st.integers(min_value=0, max_value=8).map(float)
WORST: st.SearchStrategy[float] = st.integers(min_value=-12, max_value=2).map(float)


@st.composite
def problems(draw: st.DrawFn) -> tuple[list[float], list[list[float]]]:
    """Draw the worst cases of some margins and the gains of some features.

    Returns
    -------
    tuple[list[float], list[list[float]]]
        The worst case of each margin, and for each margin the gain of each feature.

    """
    margin_count = draw(st.integers(min_value=2, max_value=MARGIN_COUNT))
    feature_count = draw(st.integers(min_value=1, max_value=FEATURE_COUNT))
    worst = [draw(WORST) for _ in range(margin_count)]
    gains = [[draw(GAIN) for _ in range(feature_count)] for _ in range(margin_count)]
    return worst, gains


def _margin(worst: Sequence[float], gains: Sequence[Sequence[float]], chosen: Sequence[int]) -> float:
    return min(w + sum(g[i] for i in chosen) for w, g in zip(worst, gains, strict=True))


def _smallest(worst: Sequence[float], gains: Sequence[Sequence[float]]) -> int | None:
    sizes = [
        size
        for size in range(len(gains[0]) + 1)
        for chosen in itertools.combinations(range(len(gains[0])), size)
        if _margin(worst, gains, chosen) > 0
    ]
    return min(sizes) if sizes else None


@given(problems())
def test_exact_search_returns_the_smallest_set(problem: tuple[list[float], list[list[float]]]) -> None:
    worst, gains = problem
    selection = ExactOrMinimal().select(worst, gains)
    smallest = _smallest(worst, gains)
    if smallest is None:
        assert selection is None
        return
    assert selection is not None
    assert len(selection.indices) == smallest
    assert selection.minimum
    assert selection.guaranteed_margin == _margin(worst, gains, selection.indices) > 0


@given(problems())
def test_minimal_search_returns_a_set_from_which_no_feature_can_be_removed(
    problem: tuple[list[float], list[list[float]]],
) -> None:
    worst, gains = problem
    selection = Minimal().select(worst, gains)
    smallest = _smallest(worst, gains)
    if smallest is None:
        assert selection is None
        return
    assert selection is not None
    assert _margin(worst, gains, selection.indices) > 0
    for feature in selection.indices:
        assert _margin(worst, gains, [other for other in selection.indices if other != feature]) <= 0
    # the flag may claim the minimum only when it holds
    if selection.minimum:
        assert len(selection.indices) == smallest
    assert len(selection.indices) >= smallest


def test_the_empty_set_is_chosen_when_the_margins_are_positive_already() -> None:
    assert Minimal().select([1.0, 2.0], [[5.0], [5.0]]) == Selection([], 1.0, minimum=True)
    assert ExactOrMinimal().select([1.0, 2.0], [[5.0], [5.0]]) == Selection([], 1.0, minimum=True)


def test_no_set_is_chosen_when_all_the_features_are_not_enough() -> None:
    assert Minimal().select([-5.0, -5.0], [[1.0, 1.0], [1.0, 1.0]]) is None
    assert ExactOrMinimal().select([-5.0, -5.0], [[1.0, 1.0], [1.0, 1.0]]) is None


WORST_MINIMAL_IS_LARGER = [-6.0, -2.0]
GAINS_MINIMAL_IS_LARGER = [[2.0, 2.0, 3.0, 5.0], [2.0, 2.0, 3.0, 1.0]]


def test_minimal_is_not_always_the_minimum() -> None:
    # features 2, 0 and 1 are enough and none can be removed, but features 2 and 3 are enough alone
    minimal = Minimal().select(WORST_MINIMAL_IS_LARGER, GAINS_MINIMAL_IS_LARGER)
    exact = ExactOrMinimal().select(WORST_MINIMAL_IS_LARGER, GAINS_MINIMAL_IS_LARGER)
    assert minimal == Selection([2, 0, 1], 1.0, minimum=False)
    assert exact == Selection([2, 3], 2.0, minimum=True)


def test_the_exact_search_falls_back_to_minimal_above_its_limit() -> None:
    assert ExactOrMinimal(max_features=3).select(WORST_MINIMAL_IS_LARGER, GAINS_MINIMAL_IS_LARGER) == Minimal().select(
        WORST_MINIMAL_IS_LARGER,
        GAINS_MINIMAL_IS_LARGER,
    )
    assert ExactOrMinimal(max_features=4).select(WORST_MINIMAL_IS_LARGER, GAINS_MINIMAL_IS_LARGER) == Selection(
        [2, 3],
        2.0,
        minimum=True,
    )


def test_features_are_ordered_by_their_smallest_gain() -> None:
    assert by_smallest_gain([[1.0, 4.0, 2.0], [3.0, 4.0, 0.5]]) == [1, 0, 2]


def test_the_limit_of_the_exact_search_is_not_negative() -> None:
    with pytest.raises(ValueError, match="max_features"):
        ExactOrMinimal(max_features=-1)
