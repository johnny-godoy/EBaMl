"""Helpers for the tests of the covering problems: a brute-force oracle and the strategies of Hypothesis.

The worst cases that the strategies draw are half-integers and the gains are integers, so no margin of a drawn problem
is zero or tiny, and the tolerances of the solvers cannot change an answer.
"""

import itertools
from collections.abc import Sequence

from hypothesis import strategies as st

from ebaml.local.covering_problem import Choice, Cover, CoveringProblem

# the parts of a problem: the worst case of each margin, and groups[group][alternative][margin]
type ProblemParts = tuple[list[float], list[list[list[float]]]]

WORST: st.SearchStrategy[float] = st.integers(min_value=-12, max_value=2).map(lambda whole: whole + 0.5)
GAIN: st.SearchStrategy[float] = st.integers(min_value=0, max_value=8).map(float)
SIGNED_GAIN: st.SearchStrategy[float] = st.integers(min_value=-4, max_value=8).map(float)
ELEMENT_COUNT = 5


@st.composite
def problem_parts(
    draw: st.DrawFn,
    *,
    max_groups: int = 6,
    max_alternatives: int = 3,
    signed: bool = True,
) -> ProblemParts:
    """Draw the parts of a covering problem.

    Hypothesis draws lists, not the attrs class, because beartype samples with ``random`` when it checks a class.

    Returns
    -------
    ProblemParts
        The worst cases and the groups.

    """
    margin_count = draw(st.integers(min_value=1, max_value=3))
    group_count = draw(st.integers(min_value=1, max_value=max_groups))
    gain = SIGNED_GAIN if signed else GAIN
    worst = [draw(WORST) for _ in range(margin_count)]
    groups = [
        [
            [draw(gain) for _ in range(margin_count)]
            for _ in range(draw(st.integers(min_value=1, max_value=max_alternatives)))
        ]
        for _ in range(group_count)
    ]
    return worst, groups


def oracle_size(problem: CoveringProblem) -> int | None:
    """Try every choice of at most one alternative from each group, by size.

    Returns
    -------
    int | None
        The fewest picks of a sufficient set, or None if no set is sufficient.

    """
    group_count = len(problem.groups)
    for size in range(group_count + 1):
        for chosen_groups in itertools.combinations(range(group_count), size):
            alternative_counts = [len(problem.groups[group]) for group in chosen_groups]
            ranges = [range(alternative_count) for alternative_count in alternative_counts]
            for alternatives in itertools.product(*ranges):
                choices = list(itertools.starmap(Choice, zip(chosen_groups, alternatives, strict=True)))
                if problem.sufficient(choices):
                    return size
    return None


def check_result(problem: CoveringProblem, cover: Cover) -> None:
    """Check the invariants of every result: one pick per group at most, sufficient, and a true guaranteed margin."""
    groups = [choice.group for choice in cover.choices]
    assert len(groups) == len(set(groups))
    assert problem.sufficient(cover.choices)
    assert cover.guaranteed_margin == min(problem.margins(cover.choices))
    assert cover.guaranteed_margin > 0
    assert cover.lower_bound <= len(cover.choices)


@st.composite
def set_covers(draw: st.DrawFn) -> list[frozenset[int]]:
    """Draw a family of sets over the elements ``0`` to ``ELEMENT_COUNT - 1``.

    Returns
    -------
    list[frozenset[int]]
        The elements of each set.

    """
    set_count = draw(st.integers(min_value=1, max_value=7))
    return [draw(st.frozensets(st.integers(min_value=0, max_value=ELEMENT_COUNT - 1))) for _ in range(set_count)]


def set_cover_problem(sets: Sequence[frozenset[int]]) -> CoveringProblem:
    """Build the problem of the reduction of Theorem 8 of ``proofs/EBaMlProofs/LocalExplanations/Hardness.lean``.

    A margin for each element with the worst case -1/2, and a group for each set with the gain 1 for its elements. A
    sufficient set is then a cover.

    Returns
    -------
    CoveringProblem
        The problem.

    """
    return CoveringProblem(
        [-0.5] * ELEMENT_COUNT,
        [[[1.0 if element in members else 0.0 for element in range(ELEMENT_COUNT)]] for members in sets],
    )


def smallest_cover(sets: Sequence[frozenset[int]]) -> int | None:
    """Return the fewest sets that cover every element, by brute force.

    Returns
    -------
    int | None
        The size, or None if no cover exists.

    """
    covers = [
        size
        for size in range(len(sets) + 1)
        for chosen in itertools.combinations(sets, size)
        if set().union(*chosen) >= set(range(ELEMENT_COUNT))
    ]
    return min(covers) if covers else None


def random_problem(seed: int, feature_count: int, margin_count: int) -> CoveringProblem:
    """Draw a monotone problem with real gains, too large for the oracle.

    Returns
    -------
    CoveringProblem
        One group for each feature, and worst cases that need a part of the gains.

    """
    import numpy as np  # ruff: ignore[import-outside-top-level] -- only the large problems need numpy

    random_generator = np.random.default_rng(seed)
    gains = random_generator.uniform(0, 1, size=(feature_count, margin_count)).round(3)
    worst = -gains.sum(axis=0) * random_generator.uniform(0.2, 0.4, size=margin_count)
    return CoveringProblem(worst.tolist(), [[row.tolist()] for row in gains])


# The greedy picks {2}, then {0} and {1}, and its removal pass keeps all three. The LP bound is 2, and {1, 3} is the
# only set of size 2, so a solver must do better than the greedy.
GREEDY_IS_NOT_OPTIMAL: CoveringProblem = CoveringProblem(
    [-1.5, -2.5, -3.5],
    [[[2.0, 2.0, 0.0]], [[2.0, 0.0, 2.0]], [[1.0, 2.0, 3.0]], [[0.0, 3.0, 3.0]]],
)
# The greedy finds no set, because the first pick breaks the second margin. {0, 1} is the only set.
GREEDY_FINDS_NOTHING: CoveringProblem = CoveringProblem([-1.5, 0.5], [[[3.0, -5.0]], [[0.0, 5.0]]])
