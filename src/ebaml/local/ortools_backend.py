"""The OR-Tools backend of the covering solvers: MathOpt with HiGHS, SCIP or CP-SAT, and GLOP for the LP.

``OrToolsBackend`` implements ``MipBackend`` of ``ebaml.local.covering_backend``. It writes a ``CoveringModel`` as a
MathOpt model, sets the parameters of the chosen solver, and reads the answer back as a ``ModelResult``. This module is
the only module of ``ebaml`` that imports ``ortools``. Reach HiGHS only through MathOpt: a process that also imports
``highspy`` fails, because both packages ship their own HiGHS library.
"""

import datetime as dt
import enum
from collections.abc import Callable, Mapping

import attrs
from ortools.math_opt.python import mathopt
from ortools.sat import sat_parameters_pb2

from ebaml.local.covering_backend import CoveringModel, ModelResult, ModelStatus, SolveSettings, Stop

# The feasibility tolerance of the solvers, far below the ε of the models, so that a solution almost always passes the
# check in Python.
_FEASIBILITY_TOLERANCE = 1e-9
# A variable of a solution is picked when its value is above this: a binary value is 0 or 1 up to the tolerance.
_PICKED = 0.5


class OrToolsSolver(enum.Enum):
    """The solver of OR-Tools that runs the integer models. The ``ortools`` wheel ships all three."""

    HIGHS = "highs"
    SCIP = "scip"
    CP_SAT = "cp_sat"


def _common_parameters(settings: SolveSettings, threads: int | None, *, heuristics: bool) -> mathopt.SolveParameters:
    first = settings.stop is Stop.FIRST
    gap = None if first else 0.0
    return mathopt.SolveParameters(
        time_limit=None if settings.time_limit is None else dt.timedelta(seconds=settings.time_limit),
        threads=threads,
        random_seed=settings.seed,
        relative_gap_tolerance=gap,
        absolute_gap_tolerance=gap,
        solution_limit=1 if first else None,
        heuristics=mathopt.Emphasis.HIGH if first and heuristics else None,
    )


def _highs_parameters(settings: SolveSettings) -> mathopt.SolveParameters:
    # HiGHS sets its threads only through its global options, so MathOpt rejects `threads` for it
    parameters = _common_parameters(settings, threads=None, heuristics=True)
    parameters.highs.double_options["mip_feasibility_tolerance"] = _FEASIBILITY_TOLERANCE
    parameters.highs.double_options["primal_feasibility_tolerance"] = _FEASIBILITY_TOLERANCE
    return parameters


def _scip_parameters(settings: SolveSettings) -> mathopt.SolveParameters:
    parameters = _common_parameters(settings, threads=1, heuristics=True)
    parameters.gscip.real_params["numerics/feastol"] = _FEASIBILITY_TOLERANCE
    return parameters


def _cp_sat_parameters(settings: SolveSettings) -> mathopt.SolveParameters:
    # CP-SAT rejects the heuristics emphasis of MathOpt, and scales the real coefficients to integers to this precision
    parameters = _common_parameters(settings, threads=1, heuristics=False)
    parameters.cp_sat.MergeFrom(sat_parameters_pb2.SatParameters(mip_wanted_precision=_FEASIBILITY_TOLERANCE))
    return parameters


@attrs.frozen
class _SolverSettings:
    solver_type: mathopt.SolverType
    parameters: Callable[[SolveSettings], mathopt.SolveParameters]


# The settings of each solver. A new solver is a new row.
_SOLVERS: Mapping[OrToolsSolver, _SolverSettings] = {
    OrToolsSolver.HIGHS: _SolverSettings(mathopt.SolverType.HIGHS, _highs_parameters),
    OrToolsSolver.SCIP: _SolverSettings(mathopt.SolverType.GSCIP, _scip_parameters),
    OrToolsSolver.CP_SAT: _SolverSettings(mathopt.SolverType.CP_SAT, _cp_sat_parameters),
}

# The signature of `mathopt.solve`. A test passes a double that obeys its contract.
type MathOptSolve = Callable[..., mathopt.SolveResult]


@attrs.frozen
class _MathOptModel:
    """A MathOpt model and the variable of each choice, in the order of ``CoveringModel.choices``."""

    model: mathopt.Model
    variables: list[mathopt.Variable]


def _mathopt_model(model: CoveringModel) -> _MathOptModel:
    """Write the covering model as a MathOpt model.

    Returns
    -------
    _MathOptModel
        The model and its variables.

    """
    built = mathopt.Model(name="covering")
    variables = [built.add_variable(lb=0.0, ub=1.0, is_integer=model.integer) for _ in model.choices]
    for members in model.group_members:
        built.add_linear_constraint(mathopt.fast_sum(variables[member] for member in members) <= 1)
    for row, need in zip(model.margin_rows, model.margin_needs, strict=True):
        built.add_linear_constraint(
            mathopt.fast_sum(coefficient * variable for coefficient, variable in zip(row, variables, strict=True))
            >= need,
        )
    if model.max_picks is not None:
        built.add_linear_constraint(mathopt.fast_sum(variables) <= model.max_picks)
    built.minimize(mathopt.fast_sum(variables))
    return _MathOptModel(built, variables)


# The status of each end of a solve that has a meaning of its own. Every other end counts as a solve with no proof.
_STATUS: Mapping[mathopt.TerminationReason, ModelStatus] = {
    mathopt.TerminationReason.OPTIMAL: ModelStatus.OPTIMAL,
    mathopt.TerminationReason.INFEASIBLE: ModelStatus.INFEASIBLE,
}


@attrs.frozen
class OrToolsBackend:
    """Solve the covering models with OR-Tools MathOpt.

    The integer model runs on the chosen solver with one thread (HiGHS sets its own) and the seed of the call, so the
    same model gives the same answer. The LP relaxation runs on GLOP. ``Stop.OPTIMAL`` sets the gap tolerances to zero.
    ``Stop.FIRST`` stops at the first solution, with the heuristics of HiGHS and SCIP turned up.

    Attributes
    ----------
    solver: OrToolsSolver
        The solver of the integer models. The default is HiGHS.
    solve_function: MathOptSolve
        The function that solves a MathOpt model. The default is ``mathopt.solve``. A test passes a double.

    Examples
    --------
    >>> from ebaml.local.covering_backend import CoveringModel, SolveSettings
    >>> model = CoveringModel([], [[]], [0.0], [], max_picks=None, integer=True)
    >>> OrToolsBackend().solve(model, SolveSettings(seed=0)).status
    <ModelStatus.OPTIMAL: 'optimal'>

    """

    solver: OrToolsSolver = OrToolsSolver.HIGHS
    solve_function: MathOptSolve = attrs.field(default=mathopt.solve, kw_only=True, repr=False)

    def solve(self, model: CoveringModel, settings: SolveSettings) -> ModelResult:
        """Solve the model, and read the answer back.

        Parameters
        ----------
        model: CoveringModel
            The model, integer or LP.
        settings: SolveSettings
            The seed, the time limit and the stopping rule.

        Returns
        -------
        ModelResult
            ``OPTIMAL`` or ``INFEASIBLE`` when the solver proves it, ``FEASIBLE`` for a solution with no proof, and
            ``NO_SOLUTION`` otherwise.

        """
        built = _mathopt_model(model)
        if model.integer:
            solver = _SOLVERS[self.solver]
            result = self.solve_function(built.model, solver.solver_type, params=solver.parameters(settings))
        else:
            result = self.solve_function(
                built.model,
                mathopt.SolverType.GLOP,
                params=mathopt.SolveParameters(threads=1),
            )
        status = _STATUS.get(result.termination.reason, ModelStatus.FEASIBLE)
        if status is ModelStatus.INFEASIBLE:
            return ModelResult(status)
        if not result.has_primal_feasible_solution():
            return ModelResult(ModelStatus.NO_SOLUTION)
        values = result.variable_values()
        picks = [
            choice
            for choice, variable in zip(model.choices, built.variables, strict=True)
            if values[variable] > _PICKED
        ]
        return ModelResult(status, sorted(picks), result.objective_value())
