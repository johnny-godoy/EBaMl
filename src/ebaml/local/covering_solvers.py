"""The covering problems of the local explanations, and the searches that solve them.

Two facts need the smallest set of features that makes several margins positive at the same time:

- the sufficient reason of a decision with several margins, such as "inside an interval";
- the counterfactual of a chosen class of a multiclass classifier.

Sorting does not solve these problems, and no fast exact algorithm exists unless P = NP
(``proofs/EBaMlProofs/LocalExplanations/Hardness.lean``). ``CoveringProblem`` holds the numbers of both problems.
A ``CoveringSearch`` solves it:

- ``Exact`` proves the smallest set with a MIP solver (HiGHS, SCIP or CP-SAT), unless its time limit ends first.
- ``Heuristic`` stops at the first set that the solver finds.
- ``Stochastic`` runs the local search of CP-SAT with a fixed seed.
- ``Greedy`` adds the alternative with the largest gain of coverage until the set is sufficient.

Every search reports a lower bound on the smallest size: the LP relaxation, rounded up. A result is marked as the
smallest set (``Cover.minimum``) only if the check in Python accepts it, and either the solver proves it optimal or its
size equals the lower bound. Every solver works with floating-point tolerances, and Lean does not check the
certificate of a solver, so the proof paper (``proofs/papers/local-explanations/paper.typ``) states the limits.

This module is the only module of ``ebaml`` that imports ``ortools``. To add a search, write a new subclass of
``CoveringSearch``.
"""

import abc
import datetime as dt
import enum
import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from typing import override

import attrs
from ortools.math_opt.python import mathopt
from ortools.sat import sat_parameters_pb2

from ebaml.core.exceptions import SearchLimitError

# The left side of each margin must reach this share of the size of the numbers of the margin. The solvers work with
# tolerances, so a strict ">" becomes ">= ε". The check in Python then uses ">" on the real numbers.
_FIRST_EPSILON = 1e-6
# If the check in Python rejects a set of the solver, the search solves again with an ε this many times larger.
_EPSILON_STEP = 100.0
_EPSILON_TRIES = 3
# The feasibility tolerance of the engines, far below ε, so that a solution of the solver almost always passes.
_FEASIBILITY_TOLERANCE = 1e-9
# The LP value is a float: subtract this before the rounding up, so that 2.0000001 gives 2 and not 3.
_ROUNDING_TOLERANCE = 1e-6
# A variable of a solution is picked when its value is above this, because the values of a binary variable are 0 or 1
# up to the tolerance.
_PICKED = 0.5


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

    def margins(self, choices: Sequence["Choice"]) -> list[float]:
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

    def sufficient(self, choices: Sequence["Choice"]) -> bool:
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
    def every_group(self) -> list["Choice"]:
        """The first alternative of each group that has one."""
        return [Choice(group, 0) for group, alternatives in enumerate(self.groups) if alternatives]


@attrs.frozen(order=True)
class Choice:
    """One picked alternative: the position of its group and its position in the group."""

    group: int
    alternative: int


@attrs.frozen
class Cover:
    """The alternatives that a search picked.

    Attributes
    ----------
    choices: list[Choice]
        The picks, by the position of their group.
    guaranteed_margin: float
        The smallest margin when these alternatives are picked. It is positive.
    lower_bound: int
        No sufficient set has fewer picks than this: the LP relaxation of the problem, rounded up. The bound depends on
        the tolerances of the LP solver.
    minimum: bool
        True if no smaller set is sufficient: the solver proved the set optimal, or its size equals ``lower_bound``.

    """

    choices: list[Choice]
    guaranteed_margin: float
    lower_bound: int
    minimum: bool


def _cover(problem: CoveringProblem, choices: Sequence[Choice], lower_bound: int, *, proved: bool) -> Cover:
    ordered = sorted(choices)
    return Cover(
        ordered,
        min(problem.margins(ordered)),
        lower_bound,
        minimum=proved or len(ordered) <= lower_bound,
    )


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
    scales = _scales(problem)
    return [_FIRST_EPSILON * scale - worst for worst, scale in zip(problem.worst, scales, strict=True)]


def greedy_cover(problem: CoveringProblem) -> list[Choice] | None:
    """Pick the alternative with the largest gain of coverage until the set is sufficient, then remove the extra ones.

    The coverage of a set is the sum over the margins of ``min(1, covered / deficit)``. It is monotone and submodular,
    so the gain of an alternative never grows when the set grows. The search keeps the last gain of each alternative in
    a heap, and computes a gain again only when it reaches the top (Minoux's lazy evaluation). The picks are the same
    as those of the plain greedy. A tie goes to the alternative with the lowest group, then the lowest position.

    After a pick, the other alternatives of its group are not available. When the set is sufficient, the search removes
    each pick that the set does not need, from the last pick to the first.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.

    Returns
    -------
    list[Choice] | None
        The picks in the order of the search, or None if no alternative with a positive gain is left before the set is
        sufficient.

    Examples
    --------
    >>> greedy_cover(CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.5, 1.5]]]))
    [Choice(group=2, alternative=0)]

    """
    deficits = _deficits(problem)
    covered = [0.0] * len(problem.worst)
    picked: list[Choice] = []
    used_groups: set[int] = set()
    # an entry is (minus the gain, group, alternative, the round in which the gain was computed)
    heap = [
        (-_coverage_gain(gains, deficits, covered), group, alternative, 0)
        for group, alternatives in enumerate(problem.groups)
        for alternative, gains in enumerate(alternatives)
    ]
    heapq.heapify(heap)
    while not problem.sufficient(picked):
        pick = _next_pick(problem, heap, used_groups, deficits, covered, round_number=len(picked))
        if pick is None:
            return None
        picked.append(pick)
        used_groups.add(pick.group)
        covered = [
            margin_covered + max(gain, 0.0)
            for margin_covered, gain in zip(covered, problem.groups[pick.group][pick.alternative], strict=True)
        ]
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


def _scales(problem: CoveringProblem) -> list[float]:
    """Return the size of the numbers of each margin: the largest absolute value of its worst case and its gains.

    Returns
    -------
    list[float]
        One positive scale for each margin.

    """
    scales = []
    for margin, worst in enumerate(problem.worst):
        size = max([abs(worst)] + [abs(gains[margin]) for alternatives in problem.groups for gains in alternatives])
        scales.append(size if size > 0 else 1.0)
    return scales


class Engine(enum.Enum):
    """The MIP solver that a search uses. The ``ortools`` wheel ships all three."""

    HIGHS = "highs"
    SCIP = "scip"
    CP_SAT = "cp_sat"


@attrs.frozen
class _SearchSettings:
    """The settings of a search that every engine reads. ``None`` leaves the default of the engine."""

    random_seed: int
    time_limit: float
    gap_tolerance: float | None = None
    solution_limit: int | None = None
    heuristics: mathopt.Emphasis | None = None
    cp_sat: sat_parameters_pb2.SatParameters = attrs.field(factory=sat_parameters_pb2.SatParameters)


def _common_parameters(settings: _SearchSettings, threads: int | None) -> mathopt.SolveParameters:
    return mathopt.SolveParameters(
        time_limit=dt.timedelta(seconds=settings.time_limit),
        threads=threads,
        random_seed=settings.random_seed,
        relative_gap_tolerance=settings.gap_tolerance,
        absolute_gap_tolerance=settings.gap_tolerance,
        solution_limit=settings.solution_limit,
        heuristics=settings.heuristics,
    )


def _highs_parameters(settings: _SearchSettings) -> mathopt.SolveParameters:
    # HiGHS sets its threads only through its global options, so MathOpt rejects `threads` for it
    parameters = _common_parameters(settings, threads=None)
    parameters.highs.double_options["mip_feasibility_tolerance"] = _FEASIBILITY_TOLERANCE
    parameters.highs.double_options["primal_feasibility_tolerance"] = _FEASIBILITY_TOLERANCE
    return parameters


def _scip_parameters(settings: _SearchSettings) -> mathopt.SolveParameters:
    parameters = _common_parameters(settings, threads=1)
    parameters.gscip.real_params["numerics/feastol"] = _FEASIBILITY_TOLERANCE
    return parameters


def _cp_sat_parameters(settings: _SearchSettings) -> mathopt.SolveParameters:
    # CP-SAT rejects the heuristics emphasis of MathOpt, and scales the real coefficients to integers to this precision
    parameters = _common_parameters(attrs.evolve(settings, heuristics=None), threads=1)
    parameters.cp_sat.MergeFrom(settings.cp_sat)
    parameters.cp_sat.MergeFrom(sat_parameters_pb2.SatParameters(mip_wanted_precision=_FEASIBILITY_TOLERANCE))
    return parameters


@attrs.frozen
class _EngineSettings:
    solver_type: mathopt.SolverType
    parameters: Callable[[_SearchSettings], mathopt.SolveParameters]


# The settings of each engine. A new engine is a new row.
_ENGINES: Mapping[Engine, _EngineSettings] = {
    Engine.HIGHS: _EngineSettings(mathopt.SolverType.HIGHS, _highs_parameters),
    Engine.SCIP: _EngineSettings(mathopt.SolverType.GSCIP, _scip_parameters),
    Engine.CP_SAT: _EngineSettings(mathopt.SolverType.CP_SAT, _cp_sat_parameters),
}

# The signature of `mathopt.solve`. A test passes a double that obeys its contract.
type MathOptSolve = Callable[..., mathopt.SolveResult]


@attrs.frozen
class _Model:
    """A MathOpt model of a problem, with one variable for each alternative that the reduction keeps."""

    model: mathopt.Model
    variables: dict[Choice, mathopt.Variable]

    def picks(self, result: mathopt.SolveResult) -> list[Choice]:
        """Return the alternatives that a solution of the model picks.

        Returns
        -------
        list[Choice]
            The picks, by group.

        """
        values = result.variable_values()
        return sorted(choice for choice, variable in self.variables.items() if values[variable] > _PICKED)


def _build_model(
    problem: CoveringProblem,
    kept: Sequence[Sequence[int]],
    *,
    integer: bool,
    epsilon: float,
) -> _Model:
    """Write the problem as a MathOpt model: binary picks, at most one per group, and each margin at least ``ε``.

    Each margin is divided by the size of its numbers, so that ``epsilon`` is relative and the coefficients are at most
    ``1``.

    Returns
    -------
    _Model
        The model and its variables.

    """
    model = mathopt.Model(name="covering")
    variables = {
        Choice(group, alternative): model.add_variable(lb=0.0, ub=1.0, is_integer=integer)
        for group, alternatives in enumerate(kept)
        for alternative in alternatives
    }
    for group, alternatives in enumerate(kept):
        if len(alternatives) > 1:
            model.add_linear_constraint(
                mathopt.fast_sum(variables[Choice(group, alternative)] for alternative in alternatives) <= 1,
            )
    for margin, (worst, scale) in enumerate(zip(problem.worst, _scales(problem), strict=True)):
        covered = mathopt.fast_sum(
            problem.groups[choice.group][choice.alternative][margin] / scale * variable
            for choice, variable in variables.items()
        )
        model.add_linear_constraint(covered >= epsilon - worst / scale)
    model.minimize(mathopt.fast_sum(variables.values()))
    return _Model(model, variables)


@attrs.frozen
class _Bound:
    """What the LP relaxation proves: a lower bound on the size, or that no set exists."""

    lower_bound: int
    infeasible: bool


def _lp_bound(problem: CoveringProblem, kept: Sequence[Sequence[int]], solve: MathOptSolve) -> _Bound:
    """Solve the LP relaxation with GLOP, and round its value up.

    The LP asks each margin to be at least zero, not ``ε``, so every set that the check in Python accepts is a solution
    of the LP, and the bound holds for it. The bound still depends on the tolerances of GLOP.

    Returns
    -------
    _Bound
        The bound, at least ``1`` because the caller has checked the empty set, or the proof that no set exists.

    """
    relaxation = _build_model(problem, kept, integer=False, epsilon=0.0)
    result = solve(relaxation.model, mathopt.SolverType.GLOP, params=mathopt.SolveParameters(threads=1))
    reason = result.termination.reason
    if reason == mathopt.TerminationReason.INFEASIBLE:
        return _Bound(lower_bound=1, infeasible=True)
    if reason != mathopt.TerminationReason.OPTIMAL:
        return _Bound(lower_bound=1, infeasible=False)
    return _Bound(lower_bound=max(1, math.ceil(result.objective_value() - _ROUNDING_TOLERANCE)), infeasible=False)


@attrs.frozen
class _Found:
    """What one search found.

    ``choices`` is a set that the check in Python accepts, or None. ``proved`` says that no smaller set exists.
    ``infeasible`` says that the solver proved that no set exists.
    """

    choices: list[Choice] | None
    proved: bool
    infeasible: bool = False


class CoveringSearch(abc.ABC):
    """A way to solve a covering problem.

    ``solve`` runs the steps that every search shares: the check of the empty set and of the set of all groups, the
    greedy, the lower bound and the fallback. A subclass fills in ``_improve``, which may find a smaller set than the
    greedy.
    """

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Find a small sufficient set of picks.

        Parameters
        ----------
        problem: CoveringProblem
            The problem.
        seed: int
            The seed of the solver, so that the same problem gives the same set. ``Stochastic`` may use its own.

        Returns
        -------
        Cover | None
            The picks, or None if the search proves that no set is sufficient. If the problem is not monotone and
            the search ends with no set and no proof that none exists, ``_settle`` raises a ``SearchLimitError``.

        """
        if problem.sufficient([]):
            return _cover(problem, [], 0, proved=True)
        if problem.monotone and not problem.sufficient(problem.every_group):
            return None
        kept = reduce_problem(problem)
        bound = _lp_bound(problem, kept, self._solve_function)
        if bound.infeasible:
            return None
        greedy = greedy_cover(problem)
        if greedy is not None and len(greedy) <= bound.lower_bound:
            return _cover(problem, greedy, bound.lower_bound, proved=True)
        return self._settle(problem, self._improve(problem, kept, greedy, seed=seed), bound.lower_bound)

    def _settle(self, problem: CoveringProblem, found: _Found, lower_bound: int) -> Cover | None:
        """Turn what the search found into the result, with the fallback when it found no set.

        Returns
        -------
        Cover | None
            The set of the search; else None if the solver proved that no set exists; else all groups of a monotone
            problem, which the caller has checked.

        Raises
        ------
        SearchLimitError
            If the problem is not monotone and the search ends with no set and no proof.

        """
        if found.choices is not None:
            return _cover(problem, found.choices, lower_bound, proved=found.proved)
        if found.infeasible:
            return None
        if problem.monotone:
            return _cover(problem, problem.every_group, lower_bound, proved=False)
        raise SearchLimitError(self._limit_message())

    @property
    @abc.abstractmethod
    def _solve_function(self) -> MathOptSolve:
        """The function that solves a MathOpt model: ``mathopt.solve``, or a test double."""

    @abc.abstractmethod
    def _improve(
        self,
        problem: CoveringProblem,
        kept: Sequence[Sequence[int]],
        greedy: list[Choice] | None,
        *,
        seed: int,
    ) -> _Found:
        """Return the best set that the search finds, given the set of the greedy."""

    @abc.abstractmethod
    def _limit_message(self) -> str:
        """Return the text of the error when the search ends with no set, with the fix."""


@attrs.frozen
class Greedy(CoveringSearch):
    """Add the alternative with the largest gain of coverage until the set is sufficient, then remove the extra picks.

    The search is the classic approximation algorithm of set cover (Chvátal), with the coverage of Wolsey for real
    gains. It takes at most one step for each group, so it needs no time limit. EBaMl has no proof of its ratio to the
    smallest size, so the lower bound in ``Cover.lower_bound`` is its guarantee. A group with several alternatives
    (the counterfactual of a chosen class) is outside the bound of Wolsey.

    Attributes
    ----------
    solve_function: MathOptSolve
        The function that solves the LP of the lower bound. The default is ``mathopt.solve``. A test passes a double.

    Examples
    --------
    >>> problem = CoveringProblem([-2.0], [[[1.0]], [[3.0]], [[2.0]]])
    >>> Greedy().solve(problem, seed=0)
    Cover(choices=[Choice(group=1, alternative=0)], guaranteed_margin=1.0, lower_bound=1, minimum=True)

    """

    solve_function: MathOptSolve = attrs.field(default=mathopt.solve, kw_only=True, repr=False)

    @property
    @override
    def _solve_function(self) -> MathOptSolve:
        return self.solve_function

    @override
    def _improve(
        self,
        problem: CoveringProblem,
        kept: Sequence[Sequence[int]],
        greedy: list[Choice] | None,
        *,
        seed: int,
    ) -> _Found:
        return _Found(greedy, proved=False)

    @override
    def _limit_message(self) -> str:
        return "The greedy search found no set of changes. Use the Exact search, which proves whether a set exists."


@attrs.frozen
class _SolverSearch(CoveringSearch):
    """A search that runs a MIP solver after the greedy, with a time limit."""

    time_limit: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0), kw_only=True)
    solve_function: MathOptSolve = attrs.field(default=mathopt.solve, kw_only=True, repr=False)

    @property
    @override
    def _solve_function(self) -> MathOptSolve:
        return self.solve_function

    @abc.abstractmethod
    def _settings(self, seed: int) -> _SearchSettings:
        """Return the settings of the solver."""

    @property
    @abc.abstractmethod
    def _engine(self) -> Engine:
        """The engine that the search runs."""

    @override
    def _improve(
        self,
        problem: CoveringProblem,
        kept: Sequence[Sequence[int]],
        greedy: list[Choice] | None,
        *,
        seed: int,
    ) -> _Found:
        epsilon = _FIRST_EPSILON
        for _ in range(_EPSILON_TRIES):
            found = self._run(problem, kept, greedy, seed=seed, epsilon=epsilon)
            if found is not None:
                return found
            epsilon *= _EPSILON_STEP
        return _Found(greedy, proved=False)

    def _run(
        self,
        problem: CoveringProblem,
        kept: Sequence[Sequence[int]],
        greedy: list[Choice] | None,
        *,
        seed: int,
        epsilon: float,
    ) -> _Found | None:
        """Solve the model once with one ``ε``.

        If the greedy found a set, the model asks for a smaller one, so a proof that none exists proves the greedy set.
        This constraint takes the place of ``SolveParameters.cutoff_limit``, which the Python API of OR-Tools 9.15
        cannot read back.

        Returns
        -------
        _Found | None
            What the solver found, or None if the check in Python rejects the set of the solver.

        """
        built = _build_model(problem, kept, integer=True, epsilon=epsilon)
        if greedy is not None:
            built.model.add_linear_constraint(mathopt.fast_sum(built.variables.values()) <= len(greedy) - 1)
        parameters = _ENGINES[self._engine].parameters(self._settings(seed))
        result = self.solve_function(built.model, _ENGINES[self._engine].solver_type, params=parameters)
        reason = result.termination.reason
        if reason == mathopt.TerminationReason.INFEASIBLE:
            return _Found(greedy, proved=greedy is not None, infeasible=greedy is None)
        if not result.has_primal_feasible_solution():
            return _Found(greedy, proved=False)
        picks = built.picks(result)
        if not problem.sufficient(picks):
            return None
        return _Found(picks, proved=reason == mathopt.TerminationReason.OPTIMAL)

    @override
    def _limit_message(self) -> str:
        return (
            f"The {self._engine.value} solver found no set of changes in its time limit of {self.time_limit} s. "
            "Give the search a larger time_limit, or use the Heuristic search, which stops at the first set."
        )


@attrs.frozen
class _EngineSearch(_SolverSearch):
    """A solver search that runs the engine that the user chooses."""

    engine: Engine = Engine.HIGHS

    @property
    @override
    def _engine(self) -> Engine:
        return self.engine


@attrs.frozen
class Exact(_EngineSearch):
    """Prove the smallest set with a MIP solver, unless the time limit ends first.

    The gap tolerances are zero, so the solver ends only when it proves the optimum. If the time limit ends first, the
    search returns the best set that it found, with ``minimum`` false unless the set reaches the lower bound.

    Attributes
    ----------
    engine: Engine
        The solver. The default is HiGHS.
    time_limit: float
        The most seconds that the solver runs. The default is 10.
    solve_function: MathOptSolve
        The function that solves the models. The default is ``mathopt.solve``. A test passes a double.

    Examples
    --------
    >>> cover = Exact().solve(CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]]]), seed=0)
    >>> [choice.group for choice in cover.choices], cover.minimum
    ([0, 1], True)

    """

    @override
    def _settings(self, seed: int) -> _SearchSettings:
        return _SearchSettings(random_seed=seed, time_limit=self.time_limit, gap_tolerance=0.0)


@attrs.frozen
class Heuristic(_EngineSearch):
    """Stop at the first set that the MIP solver finds, with the heuristics of the solver turned up.

    The solver looks only for a set smaller than the set of the greedy. The result is marked as the smallest set only
    if its size equals the lower bound, or if the solver proves that the set of the greedy is the smallest. The
    attributes ``engine``, ``time_limit`` and ``solve_function`` mean what they mean for ``Exact``.

    """

    @override
    def _settings(self, seed: int) -> _SearchSettings:
        return _SearchSettings(
            random_seed=seed,
            time_limit=self.time_limit,
            solution_limit=1,
            heuristics=mathopt.Emphasis.HIGH,
        )


@attrs.frozen
class Stochastic(_SolverSearch):
    """Run CP-SAT with its violation local search, one worker and a fixed seed.

    The same seed gives the same set on each run. The search ends at the optimum, with the default gap tolerances of
    CP-SAT, or at the time limit.

    Attributes
    ----------
    seed: int | None
        The seed of the search. ``None`` (the default) uses the seed that ``solve`` gets, ``LocalOptions.seed`` for
        the local facts, so that the facts stay reproducible.
    time_limit: float
        The most seconds that the solver runs. The default is 10.
    solve_function: MathOptSolve
        The function that solves the models. The default is ``mathopt.solve``. A test passes a double.

    """

    seed: int | None = None

    @property
    @override
    def _engine(self) -> Engine:
        return Engine.CP_SAT

    @override
    def _settings(self, seed: int) -> _SearchSettings:
        return _SearchSettings(
            random_seed=seed if self.seed is None else self.seed,
            time_limit=self.time_limit,
            cp_sat=sat_parameters_pb2.SatParameters(use_feasibility_jump=True, num_violation_ls=1),
        )
