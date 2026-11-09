"""Tests for the OR-Tools backend: the model it writes, the parameters of each solver, and how it reads an answer."""

import datetime as dt

import attrs
import pytest
from ortools.math_opt.python import mathopt

from ebaml.local.covering_backend import CoveringModel, ModelResult, ModelStatus, SolveSettings, Stop
from ebaml.local.covering_problem import Choice
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver

# Two margins that need one gain each: the alternative 2 covers both alone, so the optimum is one pick.
MODEL: CoveringModel = CoveringModel(
    [Choice(0, 0), Choice(1, 0), Choice(2, 0)],
    [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    [0.5, 0.5],
    [],
    max_picks=None,
    integer=True,
)
TOLERANCE = 1e-9


@attrs.define
class RecordingSolve:
    """A double of ``mathopt.solve`` that records each call and can replace its result.

    It obeys the contract of ``mathopt.solve``: it takes a model and a solver type, and returns a ``SolveResult`` whose
    termination fits its solutions.
    """

    result: mathopt.SolveResult | None = None
    calls: list[tuple[mathopt.SolverType, mathopt.SolveParameters]] = attrs.field(factory=list)

    def __call__(
        self,
        opt_model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters | None = None,
    ) -> mathopt.SolveResult:
        """Record the call, and return the replaced result or the result of the real solver.

        Returns
        -------
        mathopt.SolveResult
            The result.

        """
        assert params is not None
        self.calls.append((solver_type, params))
        if self.result is not None:
            return self.result
        return mathopt.solve(opt_model, solver_type, params=params)


def _ended(reason: mathopt.TerminationReason) -> mathopt.SolveResult:
    return mathopt.SolveResult(termination=mathopt.Termination(reason=reason, limit=mathopt.Limit.TIME))


@pytest.mark.parametrize("solver", list(OrToolsSolver))
def test_each_solver_proves_the_optimum_of_the_model(solver: OrToolsSolver) -> None:
    result = OrToolsBackend(solver).solve(MODEL, SolveSettings(seed=0, time_limit=5.0))
    assert result == ModelResult(ModelStatus.OPTIMAL, [Choice(2, 0)], 1.0)


def test_the_lp_runs_on_glop_with_one_thread_and_gives_its_value() -> None:
    fake = RecordingSolve()
    # each margin needs 1 from two alternatives of three, so the LP puts 1/2 on each: a value of 1.5
    model = CoveringModel(
        [Choice(0, 0), Choice(1, 0), Choice(2, 0)],
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        [1.0, 1.0, 1.0],
        [],
        max_picks=None,
        integer=False,
    )
    result = OrToolsBackend(solve_function=fake).solve(model, SolveSettings(seed=0))
    [(solver_type, params)] = fake.calls
    assert solver_type == mathopt.SolverType.GLOP
    assert params.threads == 1
    assert result.status is ModelStatus.OPTIMAL
    assert result.objective == pytest.approx(1.5)


def test_the_model_keeps_one_alternative_of_a_group_and_the_limit_of_picks() -> None:
    # the two alternatives of group 0 would cover both margins together, but at most one of them may be picked
    grouped = CoveringModel(
        [Choice(0, 0), Choice(0, 1)],
        [[1.0, 0.0], [0.0, 1.0]],
        [0.5, 0.5],
        [[0, 1]],
        max_picks=None,
        integer=True,
    )
    assert OrToolsBackend().solve(grouped, SolveSettings(seed=0)).status is ModelStatus.INFEASIBLE
    limited = attrs.evolve(MODEL, max_picks=0)
    assert OrToolsBackend().solve(limited, SolveSettings(seed=0)).status is ModelStatus.INFEASIBLE


@pytest.mark.parametrize(
    ("solver", "solver_type", "threads"),
    [
        (OrToolsSolver.HIGHS, mathopt.SolverType.HIGHS, None),
        (OrToolsSolver.SCIP, mathopt.SolverType.GSCIP, 1),
        (OrToolsSolver.CP_SAT, mathopt.SolverType.CP_SAT, 1),
    ],
)
def test_a_proved_optimum_asks_for_zero_gaps_one_thread_the_seed_and_the_limit(
    solver: OrToolsSolver,
    solver_type: mathopt.SolverType,
    threads: int | None,
) -> None:
    fake = RecordingSolve()
    OrToolsBackend(solver, solve_function=fake).solve(MODEL, SolveSettings(seed=7, time_limit=3.0, stop=Stop.OPTIMAL))
    [(called_type, params)] = fake.calls
    assert called_type == solver_type
    assert params.relative_gap_tolerance == params.absolute_gap_tolerance == pytest.approx(0.0, abs=0.0)
    assert params.solution_limit is None
    assert params.heuristics is None
    assert params.threads == threads
    assert params.random_seed == 7
    assert params.time_limit == dt.timedelta(seconds=3)


@pytest.mark.parametrize(
    ("solver", "heuristics"),
    [
        (OrToolsSolver.HIGHS, mathopt.Emphasis.HIGH),
        (OrToolsSolver.SCIP, mathopt.Emphasis.HIGH),
        (OrToolsSolver.CP_SAT, None),
    ],
)
def test_the_first_solution_asks_for_one_solution_and_the_heuristics(
    solver: OrToolsSolver,
    heuristics: mathopt.Emphasis | None,
) -> None:
    fake = RecordingSolve()
    OrToolsBackend(solver, solve_function=fake).solve(MODEL, SolveSettings(seed=0, stop=Stop.FIRST))
    [(_, params)] = fake.calls
    assert params.solution_limit == 1
    assert params.heuristics == heuristics
    assert params.relative_gap_tolerance is None
    assert params.time_limit is None


def test_each_solver_gets_its_feasibility_tolerance() -> None:
    fakes = {solver: RecordingSolve() for solver in OrToolsSolver}
    for solver, fake in fakes.items():
        OrToolsBackend(solver, solve_function=fake).solve(MODEL, SolveSettings(seed=0))
    highs = fakes[OrToolsSolver.HIGHS].calls[0][1].highs.double_options
    tolerances = [
        highs["mip_feasibility_tolerance"],
        highs["primal_feasibility_tolerance"],
        fakes[OrToolsSolver.SCIP].calls[0][1].gscip.real_params["numerics/feastol"],
        fakes[OrToolsSolver.CP_SAT].calls[0][1].cp_sat.mip_wanted_precision,
    ]
    assert tolerances == [pytest.approx(TOLERANCE, rel=0.0)] * len(tolerances)


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        (mathopt.TerminationReason.INFEASIBLE, ModelStatus.INFEASIBLE),
        (mathopt.TerminationReason.NO_SOLUTION_FOUND, ModelStatus.NO_SOLUTION),
        (mathopt.TerminationReason.NUMERICAL_ERROR, ModelStatus.NO_SOLUTION),
    ],
)
def test_an_end_with_no_solution_has_no_picks(reason: mathopt.TerminationReason, status: ModelStatus) -> None:
    backend = OrToolsBackend(solve_function=RecordingSolve(result=_ended(reason)))
    assert backend.solve(MODEL, SolveSettings(seed=0)) == ModelResult(status)


def test_a_solution_at_the_time_limit_is_feasible_with_its_picks() -> None:
    def feasible_at_limit(
        opt_model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters | None = None,
    ) -> mathopt.SolveResult:
        solved = mathopt.solve(opt_model, solver_type, params=params)
        solved.termination = mathopt.Termination(reason=mathopt.TerminationReason.FEASIBLE, limit=mathopt.Limit.TIME)
        return solved

    result = OrToolsBackend(solve_function=feasible_at_limit).solve(MODEL, SolveSettings(seed=0))
    assert result == ModelResult(ModelStatus.FEASIBLE, [Choice(2, 0)], 1.0)
