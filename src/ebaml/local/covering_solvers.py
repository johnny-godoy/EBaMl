"""The solvers of the covering problems of the local explanations.

Sorting does not solve a covering problem (``ebaml.local.covering_problem``), and no fast exact algorithm exists unless
P = NP (``proofs/EBaMlProofs/LocalExplanations/Hardness.lean``). A ``CoveringSolver`` finds a small sufficient set.
There is one class for each algorithm:

- ``GreedyCover`` builds a set with the greedy algorithm of set cover. It needs no solver library.
- ``LocalSearchCover`` starts from the greedy set, removes part of it at random and builds it again, and keeps the
  smallest set that it finds. It needs no solver library either, and a seed makes it reproducible.
- ``MipCover`` searches all sets by branch and bound in a MIP solver of a ``MipBackend``. Its stopping rule
  (``Stop.OPTIMAL`` or ``Stop.FIRST``) and its time limit are parameters.

Every solver checks the empty set first, and reports a lower bound on the smallest size: the LP relaxation, rounded up,
when it has a backend for the LP, else ``1``. A result is marked as the smallest set (``Cover.minimality``) only if the
check in Python accepts it, and either the MIP solver proves it optimal or its size equals the lower bound. The solvers
work with floating-point tolerances, and Lean does not check the certificate of a solver, so the proof paper
(``proofs/papers/local-explanations/paper.typ``) states the limits.

To add an algorithm, write a class with the ``solve`` method of ``CoveringSolver``. To add a library, write a
``MipBackend`` (``ebaml.local.covering_backend``).
"""

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import attrs
import numpy as np

from ebaml.core.exceptions import SearchLimitError
from ebaml.local.covering_backend import MipBackend, ModelStatus, SolveSettings, Stop, build_model
from ebaml.local.covering_greedy import greedy_cover
from ebaml.local.covering_problem import (
    MARGIN_EPSILON,
    Choice,
    Cover,
    CoveringProblem,
    make_cover,
    reduce_problem,
)
from ebaml.local.ortools_backend import OrToolsBackend

# If the check in Python rejects a set of the solver, the search solves again with an ε this many times larger.
_EPSILON_STEP = 100.0
_EPSILON_TRIES = 3
# The LP value is a float: subtract this before the rounding up, so that 2.0000001 gives 2 and not 3.
_ROUNDING_TOLERANCE = 1e-6


@runtime_checkable
class CoveringSolver(Protocol):
    """An algorithm that finds a small sufficient set of a covering problem."""

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Find a small sufficient set of picks.

        Parameters
        ----------
        problem: CoveringProblem
            The problem.
        seed: int
            The seed of the algorithm, ``LocalOptions.seed`` for the local facts, so that the same problem gives the
            same set.

        Returns
        -------
        Cover | None
            The picks, or None if the solver proves that no set is sufficient.

        """
        ...


@attrs.frozen
class _Opening:
    """What the steps that every solver shares found.

    ``result`` is the answer when ``final`` is true. Otherwise ``kept`` is the reduced problem, ``lower_bound`` the
    bound, and ``greedy`` the set of the greedy, empty if the greedy found no set. A set that the greedy finds is never
    empty, because ``_open`` has checked the empty set.
    """

    final: bool
    result: Cover | None = None
    kept: list[list[int]] = attrs.field(factory=list)
    lower_bound: int = 1
    greedy: list[Choice] = attrs.field(factory=list)


def _lower_bound(problem: CoveringProblem, kept: Sequence[Sequence[int]], backend: MipBackend, seed: int) -> int | None:
    """Solve the LP relaxation, and round its value up.

    The LP asks each margin to be at least zero, not ``ε``, so every set that the check in Python accepts is a solution
    of the LP, and the bound holds for it (Lemma 17 of the paper). The bound still depends on the tolerances of the LP
    solver.

    Returns
    -------
    int | None
        The bound, at least ``1`` because the caller has checked the empty set, or None if the backend proves that the
        LP has no solution, so that no set is sufficient.

    """
    result = backend.solve(build_model(problem, kept, epsilon=0.0, integer=False), SolveSettings(seed=seed))
    if result.status is ModelStatus.INFEASIBLE:
        return None
    if result.status is not ModelStatus.OPTIMAL or result.objective is None:
        return 1
    return max(1, math.ceil(result.objective - _ROUNDING_TOLERANCE))


def _open(
    problem: CoveringProblem,
    bound_backend: MipBackend | None,
    seed: int,
    *,
    give_up: str | None = None,
) -> _Opening:
    """Run the steps that every solver shares: the empty set, the set of all groups, the bound and the greedy.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.
    bound_backend: MipBackend | None
        The backend of the LP of the lower bound, or None for the bound ``1``.
    seed: int
        The seed of the LP.
    give_up: str | None
        For an algorithm that needs the greedy set: the text of the error when the greedy finds no set.

    Returns
    -------
    _Opening
        The final answer when a step settles the problem, else the reduced problem, the bound and the greedy set.

    """
    if problem.sufficient([]):
        return _Opening(final=True, result=make_cover(problem, [], 0, proved=True))
    if problem.monotone and not problem.sufficient(problem.every_group):
        return _Opening(final=True)
    kept = reduce_problem(problem)
    lower_bound = 1 if bound_backend is None else _lower_bound(problem, kept, bound_backend, seed)
    if lower_bound is None:
        return _Opening(final=True)
    greedy = greedy_cover(problem) or []
    if greedy and len(greedy) <= lower_bound:
        return _Opening(final=True, result=make_cover(problem, greedy, lower_bound, proved=True))
    if not greedy and give_up is not None:
        return _Opening(final=True, result=_no_set_found(problem, lower_bound, give_up))
    return _Opening(final=False, kept=kept, lower_bound=lower_bound, greedy=greedy)


def _no_set_found(problem: CoveringProblem, lower_bound: int, message: str) -> Cover:
    """Answer a solver that found no set and no proof that none exists.

    A monotone problem has a sufficient set iff the set of all groups is sufficient (Lemma 6), and ``_open`` has checked
    that set, so the answer is all groups. For any other problem, ``None`` would claim that no set exists.

    Returns
    -------
    Cover
        All groups of a monotone problem, not marked as the smallest set unless they reach the bound.

    Raises
    ------
    SearchLimitError
        If the problem is not monotone.

    """
    if not problem.monotone:
        raise SearchLimitError(message)
    return make_cover(problem, problem.every_group, lower_bound, proved=False)


_GREEDY_GAVE_UP = "The greedy found no set of changes. Use MipCover, which proves whether a set exists."


def _default_backend() -> MipBackend:
    return OrToolsBackend()


@attrs.frozen
class GreedyCover:
    """Build a set with the greedy algorithm of set cover, then remove the picks that the set does not need.

    The greedy is the classic approximation algorithm of set cover (Chvátal), with the coverage of Wolsey for real
    gains (``ebaml.local.covering_greedy``). It takes at most one step for each group, so it needs no time limit.
    EBaMl has no proof of its ratio to the smallest size, so the lower bound is its guarantee. A group with several
    alternatives (the counterfactual of a chosen class) is outside the bound of Wolsey.

    Attributes
    ----------
    bound_backend: MipBackend | None
        The backend that solves the LP of the lower bound. The default is ``OrToolsBackend()``. ``None`` runs no LP:
        the bound is then ``1``, and only a set of one pick is marked as the smallest set.

    Examples
    --------
    >>> problem = CoveringProblem([-2.0], [[[1.0]], [[3.0]], [[2.0]]])
    >>> cover = GreedyCover(bound_backend=None).solve(problem, seed=0)
    >>> cover.choices, cover.lower_bound, str(cover.minimality)
    ([Choice(group=1, alternative=0)], 1, 'at_lower_bound')

    """

    bound_backend: MipBackend | None = attrs.field(factory=_default_backend)

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Find a set with the greedy (``CoveringSolver.solve``).

        Parameters
        ----------
        problem: CoveringProblem
            The problem.
        seed: int
            The seed of the LP. The greedy itself is deterministic.

        Returns
        -------
        Cover | None
            The set of the greedy, or None if the LP or the set of all groups proves that no set exists. If the greedy
            finds no set of a problem with negative gains, ``_no_set_found`` raises a ``SearchLimitError``.

        """
        opening = _open(problem, self.bound_backend, seed, give_up=_GREEDY_GAVE_UP)
        if opening.final:
            return opening.result
        return make_cover(problem, opening.greedy, opening.lower_bound, proved=False)


@attrs.frozen
class LocalSearchCover:
    """Start from the greedy set, remove part of it at random, build it again, and keep the smallest set.

    Each step removes ``destroy_share`` of the current picks (at least one), and runs the greedy from the picks that
    stay, without the removed ones. The step keeps the new set if it is not larger, so that the search moves across sets
    of the same size, and the search remembers the smallest set. This is the loop of ruin and recreate that OR-Tools
    recommends for set cover. It is not global: it proves nothing, and the lower bound is its guarantee. The same seed
    gives the same set.

    Attributes
    ----------
    iterations: int
        The number of steps. The default is 100. The search stops earlier when a set reaches the lower bound.
    destroy_share: float
        The share of the picks that each step removes, in ``(0, 1]``. The default is 0.1.
    seed: int | None
        The seed of the random removals. ``None`` (the default) uses the seed that ``solve`` gets.
    bound_backend: MipBackend | None
        The backend that solves the LP of the lower bound, as for ``GreedyCover``.

    Examples
    --------
    >>> problem = CoveringProblem(
    ...     [-1.5, -2.5, -3.5], [[[2.0, 2.0, 0.0]], [[2.0, 0.0, 2.0]], [[1.0, 2.0, 3.0]], [[0.0, 3.0, 3.0]]]
    ... )
    >>> cover = LocalSearchCover().solve(problem, seed=0)
    >>> [choice.group for choice in cover.choices], cover.minimality.is_smallest
    ([1, 3], True)

    """

    iterations: int = attrs.field(default=100, validator=attrs.validators.ge(0))
    destroy_share: float = attrs.field(default=0.1, validator=[attrs.validators.gt(0.0), attrs.validators.le(1.0)])
    seed: int | None = None
    bound_backend: MipBackend | None = attrs.field(factory=_default_backend)

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Find a set with the local search (``CoveringSolver.solve``).

        Parameters
        ----------
        problem: CoveringProblem
            The problem.
        seed: int
            The seed of the removals when ``self.seed`` is None, and the seed of the LP.

        Returns
        -------
        Cover | None
            The smallest set that the search found, or None if the LP or the set of all groups proves that no set
            exists. If the greedy finds no set to start from, ``_no_set_found`` raises a ``SearchLimitError``.

        """
        opening = _open(problem, self.bound_backend, seed, give_up=_GREEDY_GAVE_UP)
        if opening.final:
            return opening.result
        random_generator = np.random.default_rng(seed if self.seed is None else self.seed)
        current = best = opening.greedy
        for _ in range(self.iterations):
            if len(best) <= opening.lower_bound:
                break
            removed_count = max(1, round(self.destroy_share * len(current)))
            removed = {int(index) for index in random_generator.choice(len(current), removed_count, replace=False)}
            stay = [pick for index, pick in enumerate(current) if index not in removed]
            rebuilt = greedy_cover(problem, stay, excluded=[current[index] for index in removed])
            if rebuilt is not None and len(rebuilt) <= len(current):
                current = rebuilt
                best = min(best, current, key=len)
        return make_cover(problem, best, opening.lower_bound, proved=False)


@attrs.frozen
class MipCover:
    """Search all sets by branch and bound in a MIP solver.

    The search runs the greedy first, and asks the solver for a set smaller than the greedy set, so a proof that none
    exists proves the greedy set the smallest. With ``Stop.OPTIMAL`` the solver ends only when it proves the optimum,
    or at the time limit. With ``Stop.FIRST`` it ends at the first smaller set. If the time limit ends first, the search
    returns the best set that it found, with ``minimality`` ``UNPROVED`` unless the set reaches the lower bound.

    Attributes
    ----------
    backend: MipBackend
        The library and its solver. The default is ``OrToolsBackend()``, with HiGHS. It also solves the LP of the bound.
    stop: Stop
        When the solver may stop. The default is ``Stop.OPTIMAL``.
    time_limit: float
        The most seconds that each call of the solver runs. The default is 10.

    Examples
    --------
    >>> cover = MipCover().solve(CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]]]), seed=0)
    >>> [choice.group for choice in cover.choices], cover.minimality.is_smallest
    ([0, 1], True)

    """

    backend: MipBackend = attrs.field(factory=_default_backend)
    stop: Stop = Stop.OPTIMAL
    time_limit: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Find a set with the MIP solver (``CoveringSolver.solve``).

        Parameters
        ----------
        problem: CoveringProblem
            The problem.
        seed: int
            The seed of the solver.

        Returns
        -------
        Cover | None
            The best set that the search found, or None if the solver proves that no set exists. If the time limit
            ends with no set of a problem that is not monotone, ``_no_set_found`` raises a ``SearchLimitError``.

        """
        opening = _open(problem, self.backend, seed)
        if opening.final:
            return opening.result
        epsilon = MARGIN_EPSILON
        for _ in range(_EPSILON_TRIES):
            result = self.backend.solve(
                build_model(
                    problem,
                    opening.kept,
                    epsilon=epsilon,
                    max_picks=len(opening.greedy) - 1 if opening.greedy else None,
                ),
                SolveSettings(seed=seed, time_limit=self.time_limit, stop=self.stop),
            )
            if result.status is ModelStatus.INFEASIBLE:
                return self._infeasible(problem, opening)
            if result.status is ModelStatus.NO_SOLUTION:
                break
            if problem.sufficient(result.picks):
                proved = result.status is ModelStatus.OPTIMAL
                return make_cover(problem, result.picks, opening.lower_bound, proved=proved)
            epsilon *= _EPSILON_STEP
        if opening.greedy:
            return make_cover(problem, opening.greedy, opening.lower_bound, proved=False)
        return _no_set_found(problem, opening.lower_bound, self._limit_message())

    @staticmethod
    def _infeasible(problem: CoveringProblem, opening: _Opening) -> Cover | None:
        """Read a proof that the model has no solution.

        Returns
        -------
        Cover | None
            The greedy set, proved the smallest because the model asked for a smaller one, or None if there is no
            greedy set, because then the model asked for any set.

        """
        if not opening.greedy:
            return None
        return make_cover(problem, opening.greedy, opening.lower_bound, proved=True)

    def _limit_message(self) -> str:
        return (
            f"The MIP solver found no set of changes in its time limit of {self.time_limit} s. Give MipCover a larger "
            "time_limit, or use stop=Stop.FIRST, which ends at the first set."
        )
