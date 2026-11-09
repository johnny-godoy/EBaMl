"""The contract between a covering solver and the optimization library that solves its linear models.

``MipCover`` and the lower bound of every solver write the covering problem as a ``CoveringModel``: plain numbers, with
no type of a library. A ``MipBackend`` solves the model and returns a ``ModelResult``. ``ebaml.local.ortools_backend``
holds the backend of OR-Tools. To add a library, write a class with the ``solve`` method of ``MipBackend``.
"""

import enum
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import attrs

from ebaml.local.covering_problem import Choice, CoveringProblem


class Stop(enum.Enum):
    """When the solver of an integer model may stop."""

    OPTIMAL = "optimal"
    """Stop only when the solver proves the optimum, with the gap tolerances zero, or at the time limit."""
    FIRST = "first"
    """Stop at the first set that the solver finds, or at the time limit."""


@attrs.frozen
class SolveSettings:
    """The settings of one call of a backend.

    Attributes
    ----------
    seed: int
        The seed of the solver, so that the same model gives the same answer.
    time_limit: float | None
        The most seconds that the solver runs. ``None`` for the LP, which has no limit.
    stop: Stop
        When the solver of an integer model may stop. The LP ignores it.

    """

    seed: int
    time_limit: float | None = None
    stop: Stop = Stop.OPTIMAL


@attrs.frozen
class CoveringModel:
    """A covering problem as a linear model: minimize the number of picks.

    Each variable is one alternative of the problem, with a value from 0 to 1. A backend solves it as an integer model
    (each value 0 or 1) or as its LP relaxation.

    Attributes
    ----------
    choices: list[Choice]
        The alternative of each variable.
    margin_rows: list[list[float]]
        For each margin, the coefficient of each variable: its gain divided by the scale of the margin.
    margin_needs: list[float]
        For each margin, the least value of the sum of its row.
    group_members: list[list[int]]
        The variables of each group with more than one alternative. At most one of them is picked.
    max_picks: int | None
        The most picks, or ``None`` for no limit.
    integer: bool
        True for the integer model, False for the LP relaxation.

    """

    choices: list[Choice]
    margin_rows: list[list[float]]
    margin_needs: list[float]
    group_members: list[list[int]]
    max_picks: int | None
    integer: bool


class ModelStatus(enum.Enum):
    """How the solve of a model ended."""

    OPTIMAL = "optimal"
    """The solver proved that its answer is optimal."""
    FEASIBLE = "feasible"
    """The solver found a solution but did not prove it optimal."""
    INFEASIBLE = "infeasible"
    """The solver proved that the model has no solution."""
    NO_SOLUTION = "no_solution"
    """The solver stopped with no solution and no proof, for example at the time limit."""


@attrs.frozen
class ModelResult:
    """The answer of a backend.

    Attributes
    ----------
    status: ModelStatus
        How the solve ended.
    picks: list[Choice]
        The alternatives whose variable is picked in the solution, by group. Empty without a solution.
    objective: float | None
        The value of the solution: the number of picks, or the LP value. ``None`` without a solution.

    """

    status: ModelStatus
    picks: list[Choice] = attrs.field(factory=list)
    objective: float | None = None


@runtime_checkable
class MipBackend(Protocol):
    """A library that solves the linear models of the covering solvers."""

    def solve(self, model: CoveringModel, settings: SolveSettings) -> ModelResult:
        """Solve the model.

        Parameters
        ----------
        model: CoveringModel
            The model, integer or LP.
        settings: SolveSettings
            The seed, the time limit and the stopping rule.

        Returns
        -------
        ModelResult
            The status, the picks of the solution and its value.

        """
        ...


def build_model(
    problem: CoveringProblem,
    kept: Sequence[Sequence[int]],
    *,
    epsilon: float,
    integer: bool = True,
    max_picks: int | None = None,
) -> CoveringModel:
    """Write the problem as a linear model: picks of the kept alternatives, each margin at least ``epsilon``.

    Each margin is divided by the size of its numbers, so that ``epsilon`` is relative and the coefficients are at most
    ``1``. The division does not change the set of solutions.

    Parameters
    ----------
    problem: CoveringProblem
        The problem.
    kept: Sequence[Sequence[int]]
        For each group, the positions of the alternatives that the model offers (``reduce_problem``).
    epsilon: float
        The least value of each scaled margin. The LP of the lower bound uses ``0``.
    integer: bool
        True for the integer model, False for the LP relaxation.
    max_picks: int | None
        The most picks, or ``None`` for no limit.

    Returns
    -------
    CoveringModel
        The model.

    Examples
    --------
    >>> model = build_model(CoveringProblem([-2.0], [[[1.0]], [[4.0]]]), [[0], [0]], epsilon=0.0)
    >>> model.margin_rows, model.margin_needs
    ([[0.25, 1.0]], [0.5])

    """
    choices = [Choice(group, alternative) for group, alternatives in enumerate(kept) for alternative in alternatives]
    position = {choice: index for index, choice in enumerate(choices)}
    margin_rows = [
        [problem.groups[choice.group][choice.alternative][margin] / scale for choice in choices]
        for margin, scale in enumerate(problem.scales)
    ]
    margin_needs = [epsilon - worst / scale for worst, scale in zip(problem.worst, problem.scales, strict=True)]
    group_members = [
        [position[Choice(group, alternative)] for alternative in alternatives]
        for group, alternatives in enumerate(kept)
        if len(alternatives) > 1
    ]
    return CoveringModel(choices, margin_rows, margin_needs, group_members, max_picks, integer)
