"""Tests for the greedy construction of a sufficient set."""

import pytest
from hypothesis import given, settings

from ebaml.local import covering_greedy as greedy_module
from ebaml.local.covering_greedy import greedy_cover
from ebaml.local.covering_problem import Choice, CoveringProblem
from tests.unit.ebaml.local.conftest import (
    ELEMENT_COUNT,
    GREEDY_FINDS_NOTHING,
    GREEDY_IS_NOT_OPTIMAL,
    ProblemParts,
    oracle_size,
    problem_parts,
    set_cover_problem,
    set_covers,
    smallest_cover,
)


def naive_greedy(problem: CoveringProblem) -> list[Choice] | None:
    """Run the greedy with every gain computed again at each step, and the same removal pass.

    Returns
    -------
    list[Choice] | None
        The picks in the order of the search, or None if the greedy finds no set.

    """
    deficits = greedy_module._deficits(problem)
    covered = [0.0] * len(problem.worst)
    picked: list[Choice] = []
    while not problem.sufficient(picked):
        used = {choice.group for choice in picked}
        candidates = [
            (greedy_module._coverage_gain(gains, deficits, covered), Choice(group, alternative))
            for group, alternatives in enumerate(problem.groups)
            if group not in used
            for alternative, gains in enumerate(alternatives)
        ]
        best_gain = max((gain for gain, _ in candidates), default=0.0)
        if best_gain <= 0:
            return None
        pick = min(choice for gain, choice in candidates if gain == best_gain)
        picked.append(pick)
        gains = problem.groups[pick.group][pick.alternative]
        covered = [margin_covered + max(gain, 0.0) for margin_covered, gain in zip(covered, gains, strict=True)]
    for pick in reversed(list(picked)):
        without = [other for other in picked if other != pick]
        if problem.sufficient(without):
            picked = without
    return picked


@settings(max_examples=150, deadline=None)
@given(parts=problem_parts(max_groups=8))
def test_the_lazy_greedy_picks_what_the_plain_greedy_picks(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    assert greedy_cover(problem) == naive_greedy(problem)


@settings(max_examples=60, deadline=None)
@given(sets=set_covers())
def test_the_greedy_on_set_cover_is_chvatals_greedy_within_its_bound(sets: list[frozenset[int]]) -> None:
    problem = set_cover_problem(sets)
    smallest = smallest_cover(sets)
    picks = greedy_cover(problem)
    if smallest is None:
        assert picks is None
        return
    assert picks is not None
    # Chvátal's greedy: the set with the most uncovered elements, the lowest index on a tie, then the removal pass
    uncovered = set(range(ELEMENT_COUNT))
    chosen: list[int] = []
    while uncovered:
        best, _ = max(enumerate(sets), key=lambda indexed: (len(indexed[1] & uncovered), -indexed[0]))
        chosen.append(best)
        uncovered -= sets[best]
    for index in reversed(list(chosen)):
        without = [other for other in chosen if other != index]
        if set().union(*(sets[other] for other in without)) >= set(range(ELEMENT_COUNT)):
            chosen = without
    assert [pick.group for pick in picks] == chosen
    harmonic = sum(1 / count for count in range(1, max(len(members) for members in sets) + 1))
    assert len(picks) <= harmonic * smallest


@settings(max_examples=60, deadline=None)
@given(parts=problem_parts())
def test_the_removal_pass_leaves_no_pick_that_the_set_does_not_need(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    picks = greedy_cover(problem)
    if picks is None:
        return
    assert problem.sufficient(picks)
    assert len(picks) >= (oracle_size(problem) or 0)
    for pick in picks:
        assert not problem.sufficient([other for other in picks if other != pick])


@pytest.mark.parametrize("specialist_count", [2, 3, 5])
def test_the_greedy_picks_the_one_generalist_over_the_specialists(specialist_count: int) -> None:
    # Each specialist has a larger smallest gain (2) than the generalist (1.5), but all of them are needed for the
    # second margin. An order by the smallest gain picks every specialist, and the generalist alone is enough.
    worst = [-1.0, -(2.0 * specialist_count - 1)]
    groups = [[[1.5, 2.0 * specialist_count]], *[[[2.0, 2.0]] for _ in range(specialist_count)]]
    assert greedy_cover(CoveringProblem(worst, groups)) == [Choice(0, 0)]


def test_the_greedy_misses_the_smallest_set_of_the_example_of_the_paper() -> None:
    assert greedy_cover(GREEDY_IS_NOT_OPTIMAL) == [Choice(2, 0), Choice(0, 0), Choice(1, 0)]
    assert oracle_size(GREEDY_IS_NOT_OPTIMAL) == 2


def test_the_greedy_gives_up_when_a_negative_gain_breaks_a_margin() -> None:
    assert greedy_cover(GREEDY_FINDS_NOTHING) is None
    assert oracle_size(GREEDY_FINDS_NOTHING) == 2


def test_the_greedy_starts_from_the_given_picks_and_skips_the_excluded_ones() -> None:
    # without the greedy's own first pick {2}, the search must rebuild from {1} and finds the smallest set {1, 3}
    assert greedy_cover(GREEDY_IS_NOT_OPTIMAL, [Choice(1, 0)], excluded=[Choice(2, 0)]) == [Choice(1, 0), Choice(3, 0)]


def test_the_removal_pass_may_drop_a_starting_pick() -> None:
    problem = CoveringProblem([-1.0], [[[0.5]], [[3.0]]])
    assert greedy_cover(problem, [Choice(0, 0)]) == [Choice(1, 0)]


def test_the_greedy_returns_no_set_when_every_useful_alternative_is_excluded() -> None:
    assert greedy_cover(CoveringProblem([-1.0], [[[2.0]]]), excluded=[Choice(0, 0)]) is None
