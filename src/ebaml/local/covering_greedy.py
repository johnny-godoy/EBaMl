"""The greedy construction of a sufficient set for a covering problem.

The greedy needs no solver library. ``GreedyCover`` and ``LocalSearchCover`` of ``ebaml.local.covering_solvers`` call
it, and ``MipCover`` uses its set as the size limit of the solver.
"""

import heapq
from collections.abc import Collection, Sequence

from ebaml.local.covering_problem import MARGIN_EPSILON, Choice, CoveringProblem


def _coverage_gain(gains: Sequence[float], deficits: Sequence[float], covered: Sequence[float]) -> float:
    """Return how much an alternative raises the coverage of the set.

    The coverage of a margin is its covered amount divided by its deficit, capped at ``1``. The gain of a margin is
    computed directly, not as a difference of two coverages, so that it never grows when ``covered`` grows. The lazy
    greedy relies on this.

    Returns
    -------
    float
        The sum over the margins of the rise of the coverage. Only the positive part of a gain counts.

    """
    return sum(
        min(max(gain, 0.0), max(deficit - margin_covered, 0.0)) / deficit
        for gain, deficit, margin_covered in zip(gains, deficits, covered, strict=True)
        if deficit > 0
    )


def _deficits(problem: CoveringProblem) -> list[float]:
    """Return how much each margin lacks to reach ``ε`` times the size of its numbers.

    Returns
    -------
    list[float]
        One deficit for each margin. A margin that needs nothing has a deficit of zero or less.

    """
    return [MARGIN_EPSILON * scale - worst for worst, scale in zip(problem.worst, problem.scales, strict=True)]


def _covered_by(problem: CoveringProblem, choices: Sequence[Choice]) -> list[float]:
    """Return, for each margin, the sum of the positive parts of the gains of the choices.

    Returns
    -------
    list[float]
        One sum for each margin, added in the order of ``choices``.

    """
    covered = [0.0] * len(problem.worst)
    for choice in choices:
        gains = problem.groups[choice.group][choice.alternative]
        covered = [margin_covered + max(gain, 0.0) for margin_covered, gain in zip(covered, gains, strict=True)]
    return covered


def greedy_cover(
    problem: CoveringProblem,
    start: Sequence[Choice] = (),
    excluded: Collection[Choice] = (),
) -> list[Choice] | None:
    """Pick the alternative with the largest gain of coverage until the set is sufficient, then remove the extra ones.

    The coverage of a set is the sum over the margins of ``min(1, covered / deficit)``. It is monotone and submodular,
    so the gain of an alternative never grows when the set grows. The search keeps the last gain of each alternative in
    a heap, and computes a gain again only when it reaches the top (Minoux's lazy evaluation). The picks are the same
    as those of the plain greedy. A tie goes to the alternative with the lowest group, then the lowest position.

    After a pick, the other alternatives of its group are not available. When the set is sufficient, the search removes
    each pick that the set does not need, from the last pick to the first, the picks of ``start`` last.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.
    start: Sequence[Choice]
        Picks to begin with, at most one for each group. ``LocalSearchCover`` passes the part of a set that it keeps.
    excluded: Collection[Choice]
        Alternatives that the search does not pick. ``LocalSearchCover`` passes the picks that it removed, so that the
        search rebuilds the set in another way.

    Returns
    -------
    list[Choice] | None
        The picks, ``start`` first and then in the order of the search, or None if no alternative with a positive gain
        is left before the set is sufficient.

    Examples
    --------
    >>> greedy_cover(CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.5, 1.5]]]))
    [Choice(group=2, alternative=0)]

    """
    deficits = _deficits(problem)
    picked = list(start)
    covered = _covered_by(problem, picked)
    used_groups = {choice.group for choice in picked}
    # an entry is (minus the gain, group, alternative, the round in which the gain was computed)
    heap = [
        (-_coverage_gain(gains, deficits, covered), group, alternative, 0)
        for group, alternatives in enumerate(problem.groups)
        for alternative, gains in enumerate(alternatives)
        if Choice(group, alternative) not in excluded
    ]
    heapq.heapify(heap)
    round_number = 0
    while not problem.sufficient(picked):
        pick = _next_pick(problem, heap, used_groups, deficits, covered, round_number=round_number)
        if pick is None:
            return None
        picked.append(pick)
        used_groups.add(pick.group)
        covered = _covered_by(problem, picked)
        round_number += 1
    for pick in reversed(list(picked)):
        without = [other for other in picked if other != pick]
        if problem.sufficient(without):
            picked = without
    return picked


def _next_pick(  # ruff: ignore[too-many-arguments] -- the lazy greedy keeps its state in these values, and a class with one user would hide the loop
    problem: CoveringProblem,
    heap: list[tuple[float, int, int, int]],
    used_groups: set[int],  # a set, because the greedy adds a group after each pick
    deficits: Sequence[float],
    covered: Sequence[float],
    *,
    round_number: int,
) -> Choice | None:
    """Pop the alternative with the largest gain, computing the stale gains again on the way.

    Returns
    -------
    Choice | None
        The pick, or None if no alternative of an unused group has a positive gain.

    """
    while heap:
        minus_gain, group, alternative, computed_in = heapq.heappop(heap)
        if group in used_groups:
            continue
        if computed_in == round_number:
            return Choice(group, alternative) if minus_gain < 0 else None
        gain = _coverage_gain(problem.groups[group][alternative], deficits, covered)
        heapq.heappush(heap, (-gain, group, alternative, round_number))
    return None
