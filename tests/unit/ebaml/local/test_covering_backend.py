"""Tests for the contract between the covering solvers and a backend: the linear model of a problem."""

import pytest

from ebaml.local.covering_backend import (
    CoveringModel,
    MipBackend,
    ModelResult,
    ModelStatus,
    SolveSettings,
    Stop,
    build_model,
)
from ebaml.local.covering_problem import Choice, CoveringProblem
from ebaml.local.ortools_backend import OrToolsBackend


def test_the_model_scales_each_margin_by_the_size_of_its_numbers() -> None:
    problem = CoveringProblem([-2.0, 1.0], [[[1.0, 0.0]], [[4.0, -2.0]]])
    model = build_model(problem, [[0], [0]], epsilon=0.1)
    assert model.choices == [Choice(0, 0), Choice(1, 0)]
    assert model.margin_rows == [[0.25, 1.0], [0.0, -1.0]]
    assert model.margin_needs == [pytest.approx(0.1 + 0.5), pytest.approx(0.1 - 0.5)]
    assert model.group_members == []
    assert model.max_picks is None
    assert model.integer


def test_the_model_offers_only_the_kept_alternatives_and_limits_a_group_to_one() -> None:
    problem = CoveringProblem([-1.0], [[[1.0], [2.0], [3.0]], [[1.0]]])
    model = build_model(problem, [[1, 2], []], epsilon=0.0, integer=False, max_picks=3)
    assert model.choices == [Choice(0, 1), Choice(0, 2)]
    assert model.group_members == [[0, 1]]
    assert model.max_picks == 3
    assert not model.integer


def test_the_settings_default_to_a_proved_optimum_and_no_limit() -> None:
    assert SolveSettings(seed=3) == SolveSettings(seed=3, time_limit=None, stop=Stop.OPTIMAL)


def test_a_result_without_a_solution_has_no_picks_and_no_value() -> None:
    assert ModelResult(ModelStatus.NO_SOLUTION) == ModelResult(ModelStatus.NO_SOLUTION, [], None)


def test_the_ortools_backend_satisfies_the_contract() -> None:
    assert isinstance(OrToolsBackend(), MipBackend)
    model = CoveringModel([Choice(0, 0)], [[1.0]], [0.5], [], max_picks=None, integer=True)
    assert OrToolsBackend().solve(model, SolveSettings(seed=0)) == ModelResult(ModelStatus.OPTIMAL, [Choice(0, 0)], 1.0)
