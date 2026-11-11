"""The covering problem of the local explanations: its numbers, the check of a set, and the reduction.

Two facts need the smallest set of features that makes several margins positive at the same time:

- the sufficient reason of a decision with several margins, such as "inside an interval";
- the counterfactual of a chosen class of a multiclass classifier.

``CoveringProblem`` holds the numbers of both. A solver of ``ebaml.local.covering_solvers`` finds a small sufficient
set and returns a ``Cover``. This module imports no solver library, so its code runs in a plain unit test.
"""

from collections.abc import Sequence

import attrs

from ebaml.local.minimality import MinimalityStatus

# A solver cannot test a strict ">", so it asks each margin to reach this share of the size of the numbers of the
# margin. The greedy uses the same share for its deficits. The check in Python then uses ">" on the real numbers.
MARGIN_EPSILON = 1e-6


@attrs.frozen(order=True)
class Choice:
    """One picked alternative: the position of its group and its position in the group."""

    group: int
    alternative: int


@attrs.frozen
class CoveringProblem:
    """The numbers of a covering problem: margins, and groups of alternatives that raise them.

    A solution picks at most one alternative from each group, and it uses as few groups as it can. A set of picks is
    *sufficient* when, for every margin, the worst case of the margin plus the gains of the picked alternatives is
    positive.

    - The sufficient reason has one group for each feature, with one alternative: the feature keeps its value.
    - The counterfactual of a chosen class has one group for each feature, with one alternative for each plausible bin
      that is not the bin of the instance.

    Attributes
    ----------
    worst: Sequence[float]
        The worst case of each margin when no alternative is picked.
    groups: Sequence[Sequence[Sequence[float]]]
        ``groups[group][alternative][margin]`` is the gain of the alternative for the margin. A gain can be negative.

    Examples
    --------
    >>> problem = CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.5, 1.5]]])
    >>> problem.margins([Choice(2, 0)])
    [0.5, 0.5]

    """

    worst: Sequence[float]
    groups: Sequence[Sequence[Sequence[float]]]

    def margins(self, choices: Sequence[Choice]) -> list[float]:
        """Return each margin when the chosen alternatives are picked.

        The sum runs in the order of ``choices``, so the same choices give the same floats every time.

        Parameters
        ----------
        choices: Sequence[Choice]
            The picked alternatives.

        Returns
        -------
        list[float]
            The worst case of each margin plus the gains of the choices.

        """
        return [
            margin_worst + sum(self.groups[choice.group][choice.alternative][margin] for choice in choices)
            for margin, margin_worst in enumerate(self.worst)
        ]

    def sufficient(self, choices: Sequence[Choice]) -> bool:
        """Tell whether every margin is positive when the chosen alternatives are picked.

        Parameters
        ----------
        choices: Sequence[Choice]
            The picked alternatives.

        Returns
        -------
        bool
            True if every margin is above zero. A margin of exactly zero is not enough, as a logit of zero is not the
            positive class.

        """
        return all(margin > 0 for margin in self.margins(choices))

    @property
    def monotone(self) -> bool:
        """True if each group has one alternative and no gain is negative.

        Then a sufficient set exists if and only if the set of all groups is sufficient (``sufficient_mono`` in
        ``proofs/EBaMlProofs/LocalExplanations/Covering.lean``).
        """
        return all(
            len(alternatives) == 1 and all(gain >= 0 for gain in alternatives[0]) for alternatives in self.groups
        )

    @property
    def every_group(self) -> list[Choice]:
        """The first alternative of each group that has one."""
        return [Choice(group, 0) for group, alternatives in enumerate(self.groups) if alternatives]

    @property
    def scales(self) -> list[float]:
        """The size of the numbers of each margin: the largest absolute value of its worst case and its gains.

        A margin with only zeros has the scale ``1``, so that each scale can divide.
        """
        scales = []
        for margin, worst in enumerate(self.worst):
            size = max([abs(worst)] + [abs(gains[margin]) for alternatives in self.groups for gains in alternatives])
            scales.append(size if size > 0 else 1.0)
        return scales


@attrs.frozen
class Cover:
    """The alternatives that a solver picked.

    Attributes
    ----------
    choices: list[Choice]
        The picks, by the position of their group.
    guaranteed_margin: float
        The smallest margin when these alternatives are picked. It is positive.
    lower_bound: int
        No sufficient set has fewer picks than this. It is the LP relaxation of the problem, rounded up, when the solver
        has a backend for the LP, and ``1`` otherwise: the solver checks the empty set first. The LP bound depends on
        the tolerances of the LP solver.
    minimality: MinimalityStatus
        ``SOLVER_PROOF`` if the solver proved the set optimal, ``AT_LOWER_BOUND`` if its size equals ``lower_bound``,
        and ``UNPROVED`` otherwise.

    """

    choices: list[Choice]
    guaranteed_margin: float
    lower_bound: int
    minimality: MinimalityStatus


def make_cover(problem: CoveringProblem, choices: Sequence[Choice], lower_bound: int, *, proved: bool) -> Cover:
    """Build the result of a solver from a sufficient set of picks.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.
    choices: Sequence[Choice]
        A set of picks that the check in Python accepts.
    lower_bound: int
        The lower bound that the solver knows.
    proved: bool
        True if the solver proved that no smaller set is sufficient.

    Returns
    -------
    Cover
        The picks by group, their smallest margin, the bound, and ``minimality``. A size at the bound is
        ``AT_LOWER_BOUND``, whether or not the solver also proved the set.

    """
    ordered = sorted(choices)
    if len(ordered) <= lower_bound:
        minimality = MinimalityStatus.AT_LOWER_BOUND
    elif proved:
        minimality = MinimalityStatus.SOLVER_PROOF
    else:
        minimality = MinimalityStatus.UNPROVED
    return Cover(ordered, min(problem.margins(ordered)), lower_bound, minimality)


def reduce_problem(problem: CoveringProblem) -> list[list[int]]:
    """Remove the alternatives that no smallest set needs, and return the positions that stay.

    Two rules remove an alternative. Each keeps the smallest size and the existence of a sufficient set (Corollary 6 of
    ``proofs/papers/local-explanations/paper.typ``, proved on paper and not checked in Lean):

    - An alternative with no positive gain: a set without it has margins at least as large, and one pick fewer.
    - An alternative that another alternative of its own group dominates: the other one has a gain at least as large
      for every margin. A set swaps one for the other, with the same size. Of two equal alternatives, the one with the
      lower position stays.

    The rules do not compare alternatives of different groups. Two features with the same gains may both be needed,
    so the removal of one of them can make every set insufficient.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.

    Returns
    -------
    list[list[int]]
        For each group, the positions of the alternatives that stay, in order.

    Examples
    --------
    >>> reduce_problem(CoveringProblem([-1.0], [[[2.0], [1.0], [2.0]], [[0.0]], [[-1.0]]]))
    [[0], [], []]

    """
    kept: list[list[int]] = []
    for alternatives in problem.groups:
        useful = [index for index, gains in enumerate(alternatives) if any(gain > 0 for gain in gains)]
        kept.append([
            index
            for index in useful
            if not any(_dominates(alternatives[other], alternatives[index], other < index) for other in useful)
        ])
    return kept


def _dominates(stronger: Sequence[float], weaker: Sequence[float], stronger_first: bool) -> bool:  # ruff: ignore[boolean-type-hint-positional-argument] -- a private predicate, called with the comparison of the positions
    """Tell whether ``stronger`` makes ``weaker`` useless: a gain at least as large for every margin.

    Returns
    -------
    bool
        True if every gain of ``stronger`` is at least the gain of ``weaker``, and the two differ or ``stronger`` comes
        first.

    """
    at_least = all(strong >= weak for strong, weak in zip(stronger, weaker, strict=True))
    return at_least and (stronger_first or list(stronger) != list(weaker))
