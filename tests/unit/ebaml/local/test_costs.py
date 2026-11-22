"""Tests of the cost strategies of a recourse plan."""

import math

import numpy as np
import pytest

from ebaml.core.exceptions import RecourseOptionError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.local.costs import CostStrategy, DataCost, ScaledCost, UnitCost, UserCost, check_costs
from ebaml.local.fact_features import Feature

# a continuous feature with the bin centers 5, 15, 25 and 35, the instance in the second bin
_EDGES = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]


def _continuous(weights: list[float], name: str = "x") -> Feature:
    graph = EBMGraph(name, "continuous", _EDGES, [0.0] * 4, [0.0] * 4)
    return Feature(name, graph, np.array(weights), 1, 15.0, np.ones(4, dtype=bool))


def _nominal(weights: list[float], name: str = "f") -> Feature:
    graph = EBMGraph(name, "nominal", ["a", "b", "c"], [0.0] * 3, [0.0] * 3)
    return Feature(name, graph, np.array(weights), 0, "a", np.ones(3, dtype=bool))


@pytest.mark.parametrize("strategy", [UnitCost(), DataCost(), UserCost(), ScaledCost(UnitCost())], ids=repr)
def test_each_strategy_keeps_the_contract(strategy: CostStrategy) -> None:
    assert isinstance(strategy, CostStrategy)
    costs = strategy.costs([_continuous([1.0] * 4), _nominal([1.0] * 3)])
    assert [len(feature_costs) for feature_costs in costs] == [4, 3]
    assert strategy.describe()


def test_unit_cost_gives_every_bin_the_cost_one() -> None:
    assert UnitCost().costs([_nominal([1.0, 1.0, 1.0])]) == [[1.0, 1.0, 1.0]]


def test_data_cost_of_a_continuous_feature_is_its_distance_over_the_median_absolute_deviation() -> None:
    # the weights put the median at 15 and the median absolute deviation at 10
    (costs,) = DataCost().costs([_continuous([1.0, 2.0, 1.0, 0.5])])
    assert costs == pytest.approx([1.0, 0.0, 1.0, 2.0])


def test_data_cost_of_a_feature_with_all_its_weight_in_one_bin_uses_the_mean_deviation() -> None:
    # the median deviation is 0, so the scale is the mean absolute deviation of the centers from 15, which is 0
    # as well, and then 1
    (costs,) = DataCost().costs([_continuous([0.0, 1.0, 0.0, 0.0])])
    assert costs == pytest.approx([10.0, 0.0, 10.0, 20.0])


def test_data_cost_of_a_level_is_one_minus_its_share() -> None:
    (costs,) = DataCost().costs([_nominal([2.0, 1.0, 1.0])])
    assert costs == pytest.approx([0.5, 0.75, 0.75])


def test_data_cost_puts_the_levels_on_the_scale_of_the_continuous_features() -> None:
    continuous, nominal = DataCost().costs([_continuous([1.0, 2.0, 1.0, 0.5]), _nominal([2.0, 1.0, 1.0])])
    # the alternatives of the continuous feature cost 1, 1 and 2 (mean 4/3), and the levels cost 0.75 each
    assert np.mean([nominal[1], nominal[2]]) == pytest.approx(np.mean([continuous[0], continuous[2], continuous[3]]))


def test_user_cost_takes_a_bin_then_a_feature_then_the_default() -> None:
    strategy = UserCost(per_feature={"f": 3.0}, per_bin={"f": {"b": 5.0}}, default=2.0)
    assert strategy.costs([_nominal([1.0] * 3), _nominal([1.0] * 3, name="g")]) == [[3.0, 5.0, 3.0], [2.0] * 3]


def test_scaled_cost_multiplies_the_base_costs_of_some_features() -> None:
    strategy = ScaledCost(UnitCost(), {"f": 4.0})
    assert strategy.costs([_nominal([1.0] * 3), _nominal([1.0] * 3, name="g")]) == [[4.0] * 3, [1.0] * 3]
    assert strategy.describe() == "every change costs the same, times a difficulty for some inputs"


@pytest.mark.parametrize("cost", [0.0, -1.0, math.inf, math.nan])
def test_a_cost_of_a_reachable_bin_must_be_finite_and_positive(cost: float) -> None:
    with pytest.raises(RecourseOptionError, match=r"change of 'f' to b is"):
        check_costs([_nominal([1.0] * 3)], [[1.0, cost, 1.0]])


def test_the_cost_of_the_bin_of_the_instance_and_of_an_unreachable_bin_is_not_checked() -> None:
    feature = _nominal([1.0] * 3)
    reachable = Feature(
        feature.name,
        feature.graph,
        feature.weights,
        0,
        "a",
        feature.plausible,
        reachable=np.array([True, True, False]),
    )
    check_costs([reachable], [[0.0, 1.0, math.inf]])
