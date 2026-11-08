"""Tests for the covering problems and their searches.

The solvers are third-party code that EBaMl does not prove, so these tests are the main guard of the claim "smallest
set". A brute-force oracle is the reference on small problems. The worst cases are half-integers and the gains are
integers, so no margin of a drawn problem is zero or tiny, and the tolerances of the solvers cannot change an answer.
"""

import datetime as dt
import itertools
import math
from collections.abc import Callable, Sequence

import attrs
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from ortools.math_opt.python import mathopt

from ebaml.core.exceptions import SearchLimitError
from ebaml.local import covering_solvers as covering
from ebaml.local.covering_solvers import (
    Choice,
    Cover,
    CoveringProblem,
    CoveringSearch,
    Engine,
    Exact,
    Greedy,
    Heuristic,
    Stochastic,
    greedy_cover,
    reduce_problem,
)
from ebaml.local.reason_search import ExactOrMinimal

# the parts of a problem: the worst case of each margin, and groups[group][alternative][margin]
type ProblemParts = tuple[list[float], list[list[list[float]]]]

WORST: st.SearchStrategy[float] = st.integers(min_value=-12, max_value=2).map(lambda whole: whole + 0.5)
GAIN: st.SearchStrategy[float] = st.integers(min_value=0, max_value=8).map(float)
SIGNED_GAIN: st.SearchStrategy[float] = st.integers(min_value=-4, max_value=8).map(float)

STRATEGIES: list[CoveringSearch] = [
    Exact(),
    Exact(Engine.SCIP),
    Exact(Engine.CP_SAT),
    Heuristic(),
    Heuristic(Engine.SCIP),
    Heuristic(Engine.CP_SAT),
    Stochastic(),
    Greedy(),
]
EXACT_ENGINES: list[CoveringSearch] = [Exact(engine) for engine in Engine]
ZERO_GAP = 0


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
            ranges = [range(len(problem.groups[group])) for group in chosen_groups]
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


def solve_or_limit(strategy: CoveringSearch, problem: CoveringProblem) -> Cover | SearchLimitError | None:
    """Solve, and return the error of the limit in place of raising it.

    Returns
    -------
    Cover | None | SearchLimitError
        The result of the search, or its error.

    """
    try:
        return strategy.solve(problem, seed=0)
    except SearchLimitError as error:
        return error


@settings(max_examples=60, deadline=None)
@given(parts=problem_parts())
@example(parts=([-0.5], [[[1.0]]]))  # one margin, one group
@example(parts=([1.5, 0.5], [[[-1.0, 0.0]]]))  # the empty set is sufficient
@example(parts=([-5.5, -5.5], [[[1.0, 1.0]], [[1.0, 1.0]]]))  # no set is sufficient
@example(parts=([-1.5, -1.5], [[[2.0, 0.0]], [[0.0, 2.0]], [[2.0, 2.0]]]))  # ties between sets of size 1
@example(parts=([-1.5, 0.5], [[[3.0, -5.0]], [[0.0, 5.0]]]))  # a negative gain: the greedy finds no set
@pytest.mark.parametrize("strategy", STRATEGIES, ids=repr)
def test_every_strategy_returns_a_sufficient_set_no_smaller_than_the_oracle(
    strategy: CoveringSearch,
    parts: ProblemParts,
) -> None:
    problem = CoveringProblem(*parts)
    smallest = oracle_size(problem)
    result = solve_or_limit(strategy, problem)
    if isinstance(result, SearchLimitError):
        # only the greedy, which proves nothing, may give up, and only on a problem with negative gains
        assert isinstance(strategy, Greedy)
        assert not problem.monotone
        return
    if smallest is None:
        assert result is None
        return
    assert result is not None
    check_result(problem, result)
    assert len(result.choices) >= smallest
    assert result.lower_bound <= smallest
    if result.minimum:
        assert len(result.choices) == smallest
    if isinstance(strategy, Exact):
        assert len(result.choices) == smallest
        assert result.minimum


@settings(max_examples=40, deadline=None)
@given(parts=problem_parts())
def test_a_strategy_returns_none_only_when_the_oracle_finds_no_set(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    if Exact().solve(problem, seed=0) is None:
        assert oracle_size(problem) is None


def _random_problem(seed: int, feature_count: int, margin_count: int) -> CoveringProblem:
    random_generator = np.random.default_rng(seed)
    gains = random_generator.uniform(0, 1, size=(feature_count, margin_count)).round(3)
    worst = -gains.sum(axis=0) * random_generator.uniform(0.2, 0.4, size=margin_count)
    return CoveringProblem(worst.tolist(), [[row.tolist()] for row in gains])


@pytest.mark.parametrize(("seed", "feature_count", "margin_count"), [(0, 60, 3), (1, 100, 6), (2, 150, 9)])
def test_the_engines_agree_on_the_size_of_large_problems(seed: int, feature_count: int, margin_count: int) -> None:
    problem = _random_problem(seed, feature_count, margin_count)
    results = [strategy.solve(problem, seed=0) for strategy in EXACT_ENGINES]
    sizes = set()
    for result in results:
        assert result is not None
        check_result(problem, result)
        assert result.minimum
        sizes.add(len(result.choices))
    assert len(sizes) == 1


def _exact_size(problem: CoveringProblem) -> int | None:
    result = Exact().solve(problem, seed=0)
    return None if result is None else len(result.choices)


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts(), power=st.integers(min_value=-6, max_value=6))
def test_the_size_survives_a_scaling_of_all_numbers_by_a_power_of_ten(parts: ProblemParts, power: int) -> None:
    worst, groups = parts
    factor = 10.0**power
    scaled = CoveringProblem(
        [value * factor for value in worst],
        [[[gain * factor for gain in gains] for gains in alternatives] for alternatives in groups],
    )
    assert _exact_size(scaled) == _exact_size(CoveringProblem(worst, groups))


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts())
def test_the_size_survives_a_permutation_of_the_groups_and_the_margins(parts: ProblemParts) -> None:
    worst, groups = parts
    reversed_groups = CoveringProblem(worst, list(reversed(groups)))
    reversed_margins = CoveringProblem(
        list(reversed(worst)),
        [[list(reversed(gains)) for gains in alternatives] for alternatives in groups],
    )
    size = _exact_size(CoveringProblem(worst, groups))
    assert _exact_size(reversed_groups) == size
    assert _exact_size(reversed_margins) == size


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts())
def test_the_size_survives_a_useless_group_a_dominated_alternative_and_a_copied_margin(parts: ProblemParts) -> None:
    worst, groups = parts
    size = _exact_size(CoveringProblem(worst, groups))
    zero_group = CoveringProblem(worst, [*groups, [[0.0] * len(worst)]])
    dominated = CoveringProblem(worst, [[*groups[0], [gain - 1.0 for gain in groups[0][0]]], *groups[1:]])
    copied_margin = CoveringProblem(
        [*worst, worst[0]],
        [[[*gains, gains[0]] for gains in alternatives] for alternatives in groups],
    )
    assert _exact_size(zero_group) == size
    assert _exact_size(dominated) == size
    assert _exact_size(copied_margin) == size


def test_a_margin_of_exactly_zero_is_not_sufficient() -> None:
    # The one set of size 1 leaves the second margin at exactly zero, as a logit of zero is not the positive class.
    # The set of size 2 is the smallest.
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 1.0]], [[0.0, 2.0]]])
    assert not problem.sufficient([Choice(0, 0)])
    assert oracle_size(problem) == 2
    for strategy in STRATEGIES:
        result = strategy.solve(problem, seed=0)
        assert result is not None
        assert len(result.choices) == 2
        assert result.guaranteed_margin > 0


def test_a_tiny_positive_margin_is_sufficient_but_below_epsilon() -> None:
    # The set {0} keeps a margin of 1e-12, which the check in Python accepts and the solvers, with ε relative to the
    # numbers, do not. The greedy finds the set, and the solver, which looks for a smaller one, proves that none exists.
    problem = CoveringProblem([-1.0], [[[1.0 + 1e-12]], [[0.5]], [[0.5]]])
    assert problem.sufficient([Choice(0, 0)])
    for strategy in STRATEGIES:
        result = strategy.solve(problem, seed=0)
        assert result is not None
        check_result(problem, result)
        assert len(result.choices) == 1


def test_a_set_of_the_solver_that_fails_the_check_is_solved_again_with_a_larger_epsilon() -> None:
    calls: list[float] = []
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]], [[1.0, 1.0]]])

    def wrong_then_right(
        opt_model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters | None = None,
        model_params: mathopt.ModelSolveParameters | None = None,
    ) -> mathopt.SolveResult:
        if solver_type == mathopt.SolverType.GLOP:
            return _not_solved(mathopt.TerminationReason.NO_SOLUTION_FOUND)
        calls.append(solver_type.value)
        if len(calls) == 1:
            # the set {2} has margins of exactly zero, which a solver with a loose tolerance may return
            values = {variable: float(index == 2) for index, variable in enumerate(opt_model.variables())}
            return _solved(values, mathopt.TerminationReason.OPTIMAL)
        return mathopt.solve(opt_model, solver_type, params=params, model_params=model_params)

    result = Exact(solve_function=wrong_then_right).solve(problem, seed=0)
    assert result is not None
    check_result(problem, result)
    assert len(calls) == 2


def test_every_epsilon_failing_the_check_leaves_the_set_of_the_greedy() -> None:
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]], [[1.0, 1.0]]])
    values_of: Callable[[mathopt.Model], dict[mathopt.Variable, float]] = lambda opt_model: {  # ruff: ignore[lambda-assignment] -- a short double that the test passes to the fake
        variable: float(index == 2) for index, variable in enumerate(opt_model.variables())
    }
    fake = RecordingSolve(
        lp=_not_solved(mathopt.TerminationReason.NO_SOLUTION_FOUND),
        mip=lambda opt_model: _solved(values_of(opt_model), mathopt.TerminationReason.OPTIMAL),
    )
    result = Exact(solve_function=fake).solve(problem, seed=0)
    assert result is not None
    assert result.choices == greedy_cover(problem)
    assert not result.minimum
    assert len(fake.calls) == 3


def _not_solved(reason: mathopt.TerminationReason) -> mathopt.SolveResult:
    return mathopt.SolveResult(termination=mathopt.Termination(reason=reason, limit=mathopt.Limit.TIME))


def _solved(values: dict[mathopt.Variable, float], reason: mathopt.TerminationReason) -> mathopt.SolveResult:
    solution = mathopt.Solution(
        primal_solution=mathopt.PrimalSolution(
            variable_values=values,
            objective_value=sum(values.values()),
            feasibility_status=mathopt.SolutionStatus.FEASIBLE,
        ),
    )
    limit = None if reason == mathopt.TerminationReason.OPTIMAL else mathopt.Limit.TIME
    return mathopt.SolveResult(termination=mathopt.Termination(reason=reason, limit=limit), solutions=[solution])


@attrs.define
class RecordingSolve:
    """A double of ``mathopt.solve`` that records each call and can replace the result of the LP or of the MIP.

    It obeys the contract of ``mathopt.solve``: it takes a model and a solver type, and returns a ``SolveResult`` whose
    termination fits its solutions.
    """

    lp: mathopt.SolveResult | None = None
    mip: Callable[[mathopt.Model], mathopt.SolveResult] | None = None
    calls: list[tuple[mathopt.SolverType, mathopt.SolveParameters]] = attrs.field(factory=list)

    def __call__(
        self,
        opt_model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters | None = None,
        model_params: mathopt.ModelSolveParameters | None = None,
    ) -> mathopt.SolveResult:
        """Record the call, and return the replaced result or the result of the real solver.

        Returns
        -------
        mathopt.SolveResult
            The result.

        """
        if solver_type == mathopt.SolverType.GLOP:
            if self.lp is not None:
                return self.lp
            return mathopt.solve(opt_model, solver_type, params=params, model_params=model_params)
        assert params is not None
        self.calls.append((solver_type, params))
        if self.mip is not None:
            return self.mip(opt_model)
        return mathopt.solve(opt_model, solver_type, params=params, model_params=model_params)


# The greedy picks {2}, then {0} and {1}, and its removal pass keeps all three. The LP bound is 2, and {1, 3} is the
# only set of size 2, so the MIP runs.
GREEDY_IS_NOT_OPTIMAL: CoveringProblem = CoveringProblem(
    [-1.5, -2.5, -3.5],
    [[[2.0, 2.0, 0.0]], [[2.0, 0.0, 2.0]], [[1.0, 2.0, 3.0]], [[0.0, 3.0, 3.0]]],
)
# The greedy finds no set, because the first pick breaks the second margin. {0, 1} is the only set.
GREEDY_FINDS_NOTHING: CoveringProblem = CoveringProblem([-1.5, 0.5], [[[3.0, -5.0]], [[0.0, 5.0]]])


def test_the_problems_of_the_termination_tests_need_the_solver() -> None:
    assert greedy_cover(GREEDY_IS_NOT_OPTIMAL) is not None
    assert len(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or []) == 3
    assert oracle_size(GREEDY_IS_NOT_OPTIMAL) == 2
    assert greedy_cover(GREEDY_FINDS_NOTHING) is None
    assert oracle_size(GREEDY_FINDS_NOTHING) == 2


@pytest.mark.parametrize("strategy_class", [Exact, Heuristic, Stochastic])
def test_an_optimal_solver_proves_the_smallest_set(strategy_class: type[Exact | Heuristic | Stochastic]) -> None:
    result = strategy_class(solve_function=RecordingSolve()).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [1, 3]
    assert result.minimum


def test_a_feasible_set_at_the_time_limit_is_returned_without_the_claim() -> None:
    def feasible_at_limit(opt_model: mathopt.Model) -> mathopt.SolveResult:
        # the double ignores the constraint on the size, as a solver with a weak heuristic may find the same set
        values = dict.fromkeys(opt_model.variables(), 1.0)
        values[next(iter(opt_model.variables()))] = 1.0
        return _solved(values, mathopt.TerminationReason.FEASIBLE)

    lp = _not_solved(mathopt.TerminationReason.NO_SOLUTION_FOUND)
    result = Exact(solve_function=RecordingSolve(lp=lp, mip=feasible_at_limit)).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    check_result(GREEDY_IS_NOT_OPTIMAL, result)
    assert not result.minimum
    assert result.lower_bound == 1


def test_no_solution_at_the_time_limit_keeps_the_set_of_the_greedy() -> None:
    fake = RecordingSolve(mip=lambda _: _not_solved(mathopt.TerminationReason.NO_SOLUTION_FOUND))
    result = Exact(solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert result.choices == sorted(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or [])
    assert not result.minimum


def test_an_infeasible_model_with_the_size_limit_proves_the_set_of_the_greedy() -> None:
    fake = RecordingSolve(mip=lambda _: _not_solved(mathopt.TerminationReason.INFEASIBLE))
    result = Exact(solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert len(result.choices) == 3
    assert result.minimum


def test_an_infeasible_model_without_a_greedy_set_proves_that_no_set_exists() -> None:
    fake = RecordingSolve(mip=lambda _: _not_solved(mathopt.TerminationReason.INFEASIBLE))
    assert Exact(solve_function=fake).solve(GREEDY_FINDS_NOTHING, seed=0) is None


def test_no_solution_at_the_time_limit_and_no_greedy_set_raises_with_the_fix() -> None:
    fake = RecordingSolve(mip=lambda _: _not_solved(mathopt.TerminationReason.NO_SOLUTION_FOUND))
    with pytest.raises(SearchLimitError, match=r"time limit of 2\.5 s\. Give the search a larger time_limit"):
        Exact(time_limit=2.5, solve_function=fake).solve(GREEDY_FINDS_NOTHING, seed=0)


def test_the_greedy_that_finds_no_set_raises_and_names_the_exact_search() -> None:
    with pytest.raises(SearchLimitError, match="Use the Exact search"):
        Greedy().solve(GREEDY_FINDS_NOTHING, seed=0)


def test_the_exact_search_finds_the_set_that_the_greedy_misses() -> None:
    result = Exact().solve(GREEDY_FINDS_NOTHING, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [0, 1]
    assert result.minimum


def test_an_infeasible_lp_proves_that_no_set_exists_for_every_strategy() -> None:
    lp = _not_solved(mathopt.TerminationReason.INFEASIBLE)
    for strategy in (Exact(solve_function=RecordingSolve(lp=lp)), Greedy(solve_function=RecordingSolve(lp=lp))):
        assert strategy.solve(GREEDY_FINDS_NOTHING, seed=0) is None


def test_a_monotone_problem_with_no_set_found_falls_back_to_every_group() -> None:
    # The greedy always finds a set of a monotone problem whose full set is sufficient, so only a direct call reaches
    # the fallback. It returns every group, which the caller has checked, without the claim.
    problem = CoveringProblem([-1.5], [[[1.0]], [[1.0]], [[0.0]]])
    cover = Exact()._settle(problem, covering._Found(None, proved=False), lower_bound=1)
    assert cover == Cover([Choice(0, 0), Choice(1, 0), Choice(2, 0)], 0.5, 1, minimum=False)


def test_a_monotone_problem_whose_full_set_fails_has_no_set() -> None:
    fake = RecordingSolve()
    assert Exact(solve_function=fake).solve(CoveringProblem([-5.5], [[[1.0]], [[2.0]]]), seed=0) is None
    assert fake.calls == []


def test_the_heuristic_with_the_real_solver_stops_at_a_valid_set() -> None:
    # solution_limit=1 is a limit that does not depend on the speed of the machine
    problem = _random_problem(3, 80, 5)
    result = Heuristic().solve(problem, seed=0)
    assert result is not None
    check_result(problem, result)


@pytest.mark.parametrize(
    ("strategy", "solver_type", "threads"),
    [
        (Exact(time_limit=3.0), mathopt.SolverType.HIGHS, None),
        (Exact(Engine.SCIP, time_limit=3.0), mathopt.SolverType.GSCIP, 1),
        (Exact(Engine.CP_SAT, time_limit=3.0), mathopt.SolverType.CP_SAT, 1),
    ],
    ids=repr,
)
def test_the_exact_search_asks_for_zero_gaps_one_thread_the_seed_and_the_limit(
    strategy: Exact,
    solver_type: mathopt.SolverType,
    threads: int | None,
) -> None:
    fake = RecordingSolve()
    attrs.evolve(strategy, solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=7)
    [(called_type, params)] = fake.calls
    assert called_type == solver_type
    assert params.relative_gap_tolerance == params.absolute_gap_tolerance == ZERO_GAP
    assert params.solution_limit is None
    assert params.threads == threads
    assert params.random_seed == 7
    assert params.time_limit == dt.timedelta(seconds=3)


def test_each_engine_gets_its_feasibility_tolerance() -> None:
    fakes = {engine: RecordingSolve() for engine in Engine}
    for engine, fake in fakes.items():
        Exact(engine, solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    highs = fakes[Engine.HIGHS].calls[0][1].highs.double_options
    tolerances = [
        highs["mip_feasibility_tolerance"],
        highs["primal_feasibility_tolerance"],
        fakes[Engine.SCIP].calls[0][1].gscip.real_params["numerics/feastol"],
        fakes[Engine.CP_SAT].calls[0][1].cp_sat.mip_wanted_precision,
    ]
    assert tolerances == [pytest.approx(1e-9, rel=0.0)] * len(tolerances)


@pytest.mark.parametrize(
    ("engine", "heuristics"),
    [(Engine.HIGHS, mathopt.Emphasis.HIGH), (Engine.SCIP, mathopt.Emphasis.HIGH), (Engine.CP_SAT, None)],
)
def test_the_heuristic_search_stops_at_the_first_solution(engine: Engine, heuristics: mathopt.Emphasis | None) -> None:
    fake = RecordingSolve()
    Heuristic(engine, time_limit=4.0, solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    [(_, params)] = fake.calls
    assert params.solution_limit == 1
    assert params.heuristics == heuristics
    assert params.relative_gap_tolerance is None
    assert params.time_limit == dt.timedelta(seconds=4)


def test_the_stochastic_search_runs_the_local_search_of_cp_sat_with_its_seed() -> None:
    fake = RecordingSolve()
    Stochastic(seed=11, time_limit=5.0, solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    [(solver_type, params)] = fake.calls
    assert solver_type == mathopt.SolverType.CP_SAT
    assert params.random_seed == 11
    assert params.threads == 1
    assert params.cp_sat.use_feasibility_jump
    assert params.cp_sat.num_violation_ls == 1
    assert params.relative_gap_tolerance is None
    assert params.time_limit == dt.timedelta(seconds=5)


def test_the_stochastic_search_without_a_seed_uses_the_seed_of_the_call() -> None:
    fake = RecordingSolve()
    Stochastic(solve_function=fake).solve(GREEDY_IS_NOT_OPTIMAL, seed=3)
    assert fake.calls[0][1].random_seed == 3


def test_the_lp_runs_on_glop_with_one_thread() -> None:
    seen: list[tuple[mathopt.SolverType, mathopt.SolveParameters | None]] = []

    def recording(
        opt_model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters | None = None,
        model_params: mathopt.ModelSolveParameters | None = None,
    ) -> mathopt.SolveResult:
        seen.append((solver_type, params))
        return mathopt.solve(opt_model, solver_type, params=params, model_params=model_params)

    Greedy(solve_function=recording).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    [(solver_type, params)] = seen
    assert solver_type == mathopt.SolverType.GLOP
    assert params is not None
    assert params.threads == 1


@pytest.mark.parametrize("strategy", STRATEGIES, ids=repr)
def test_each_strategy_gives_the_same_set_twice(strategy: CoveringSearch) -> None:
    problem = _random_problem(4, 40, 4)
    assert strategy.solve(problem, seed=5) == strategy.solve(problem, seed=5)


def test_the_time_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="time_limit"):
        Exact(time_limit=0.0)


ELEMENT_COUNT = 5


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


def _set_cover_problem(sets: Sequence[frozenset[int]]) -> CoveringProblem:
    # Theorem 8 of proofs/EBaMlProofs/LocalExplanations/Hardness.lean: a margin for each element with the worst case
    # -1/2, and a group for each set with the gain 1 for its elements. A sufficient set is then a cover.
    return CoveringProblem(
        [-0.5] * ELEMENT_COUNT,
        [[[1.0 if element in members else 0.0 for element in range(ELEMENT_COUNT)]] for members in sets],
    )


def _smallest_cover(sets: Sequence[frozenset[int]]) -> int | None:
    covers = [
        size
        for size in range(len(sets) + 1)
        for chosen in itertools.combinations(sets, size)
        if set().union(*chosen) >= set(range(ELEMENT_COUNT))
    ]
    return min(covers) if covers else None


@settings(max_examples=40, deadline=None)
@given(sets=set_covers())
@pytest.mark.parametrize("strategy", EXACT_ENGINES, ids=repr)
def test_the_exact_search_solves_set_cover_through_the_reduction_of_the_proof(
    strategy: CoveringSearch,
    sets: list[frozenset[int]],
) -> None:
    result = strategy.solve(_set_cover_problem(sets), seed=0)
    smallest = _smallest_cover(sets)
    if smallest is None:
        assert result is None
    else:
        assert result is not None
        assert len(result.choices) == smallest
        assert result.minimum


@settings(max_examples=60, deadline=None)
@given(parts=problem_parts(max_alternatives=1, signed=False))
def test_the_exact_search_agrees_with_the_search_by_size(parts: ProblemParts) -> None:
    worst, groups = parts
    gains = [[alternatives[0][margin] for alternatives in groups] for margin in range(len(worst))]
    by_size = ExactOrMinimal().select(worst, gains)
    result = Exact().solve(CoveringProblem(worst, groups), seed=0)
    assert (by_size is None) == (result is None)
    if by_size is not None and result is not None:
        assert len(result.choices) == len(by_size.indices)


def naive_greedy(problem: CoveringProblem) -> list[Choice] | None:
    """Run the greedy with every gain computed again at each step, and the same removal pass.

    Returns
    -------
    list[Choice] | None
        The picks in the order of the search, or None if the greedy finds no set.

    """
    deficits = covering._deficits(problem)
    covered = [0.0] * len(problem.worst)
    picked: list[Choice] = []
    while not problem.sufficient(picked):
        used = {choice.group for choice in picked}
        candidates = [
            (covering._coverage_gain(gains, deficits, covered), Choice(group, alternative))
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
    problem = _set_cover_problem(sets)
    smallest = _smallest_cover(sets)
    picks = greedy_cover(problem)
    if smallest is None:
        assert picks is None
        return
    assert picks is not None
    # Chvátal's greedy: the set with the most uncovered elements, the lowest index on a tie, then the removal pass
    uncovered = set(range(ELEMENT_COUNT))
    chosen: list[int] = []
    while uncovered:
        best = max(range(len(sets)), key=lambda index: (len(sets[index] & uncovered), -index))
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
    for pick in picks:
        assert not problem.sufficient([other for other in picks if other != pick])


@settings(max_examples=60, deadline=None)
@given(parts=problem_parts())
def test_the_lower_bound_never_exceeds_the_oracle(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    smallest = oracle_size(problem)
    result = solve_or_limit(Greedy(), problem)
    if isinstance(result, Cover):
        assert smallest is not None
        assert result.lower_bound <= smallest
        if len(result.choices) == result.lower_bound:
            assert smallest == result.lower_bound


@pytest.mark.parametrize("specialist_count", [2, 3, 5])
def test_the_greedy_picks_the_one_generalist_over_the_specialists(specialist_count: int) -> None:
    # Each specialist has a larger smallest gain (2) than the generalist (1.5), but all of them are needed for the
    # second margin. An order by the smallest gain picks every specialist, and the generalist alone is enough.
    worst = [-1.0, -(2.0 * specialist_count - 1)]
    groups = [[[1.5, 2.0 * specialist_count]], *[[[2.0, 2.0]] for _ in range(specialist_count)]]
    result = Greedy().solve(CoveringProblem(worst, groups), seed=0)
    assert result is not None
    assert result.choices == [Choice(0, 0)]
    assert result.minimum


def test_the_size_limit_of_the_greedy_does_not_change_the_size_of_the_exact_search() -> None:
    kept = reduce_problem(GREEDY_IS_NOT_OPTIMAL)
    with_limit = Exact()._run(GREEDY_IS_NOT_OPTIMAL, kept, greedy_cover(GREEDY_IS_NOT_OPTIMAL), seed=0, epsilon=1e-6)
    without_limit = Exact()._run(GREEDY_IS_NOT_OPTIMAL, kept, None, seed=0, epsilon=1e-6)
    assert with_limit is not None
    assert without_limit is not None
    assert len(with_limit.choices or []) == len(without_limit.choices or []) == 2


def _reduced(problem: CoveringProblem) -> CoveringProblem:
    kept = reduce_problem(problem)
    return CoveringProblem(
        problem.worst,
        [[problem.groups[group][index] for index in indices] for group, indices in enumerate(kept)],
    )


@settings(max_examples=100, deadline=None)
@given(parts=problem_parts(max_groups=5))
def test_the_reduction_keeps_the_smallest_size(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    assert oracle_size(_reduced(problem)) == oracle_size(problem)


def test_the_reduction_keeps_the_only_alternative_that_meets_a_margin() -> None:
    problem = CoveringProblem([-1.5, -1.5], [[[2.0, 0.0], [1.0, 0.0]], [[0.0, 2.0]]])
    assert reduce_problem(problem) == [[0], [0]]


def test_the_reduction_keeps_the_first_of_two_equal_alternatives_of_a_group() -> None:
    assert reduce_problem(CoveringProblem([-0.5], [[[1.0], [1.0], [0.5]]])) == [[0]]


def test_the_reduction_keeps_two_groups_with_equal_gains() -> None:
    # Both features are needed: a removal of one, as a dominance across groups would do, leaves no sufficient set.
    problem = CoveringProblem([-1.5], [[[1.0]], [[1.0]]])
    assert reduce_problem(problem) == [[0], [0]]
    assert oracle_size(problem) == 2


def test_the_reduction_removes_the_alternatives_with_no_positive_gain() -> None:
    assert reduce_problem(CoveringProblem([-0.5, -0.5], [[[0.0, 0.0], [-1.0, 0.0]], [[2.0, -1.0]]])) == [[], [0]]


def test_the_monotone_flag_needs_one_alternative_and_no_negative_gain() -> None:
    assert CoveringProblem([-1.0], [[[1.0]], [[0.0]]]).monotone
    assert not CoveringProblem([-1.0], [[[1.0], [2.0]]]).monotone
    assert not CoveringProblem([-1.0], [[[-1.0]]]).monotone


def test_a_group_with_no_alternative_is_never_picked() -> None:
    problem = CoveringProblem([-0.5], [[], [[1.0]]])
    for strategy in STRATEGIES:
        result = strategy.solve(problem, seed=0)
        assert result is not None
        assert result.choices == [Choice(1, 0)]
    assert problem.every_group == [Choice(1, 0)]


def test_the_lower_bound_of_a_fractional_lp_is_rounded_up() -> None:
    # each margin needs a gain of 1 from two of the three groups, so the LP puts 1/2 on each group: a value of 1.5
    sets = [frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})]
    problem = CoveringProblem(
        [-1.0] * 3,
        [[[1.0 if element in members else 0.0 for element in range(3)]] for members in sets],
    )
    bound = covering._lp_bound(problem, reduce_problem(problem), mathopt.solve)
    assert bound == covering._Bound(lower_bound=2, infeasible=False)
    assert math.isclose(2, bound.lower_bound)
