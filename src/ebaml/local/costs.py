"""How hard each change of a recourse plan is: the cost strategies of ``RecourseOptions.costs``.

A strategy gives a cost to each bin of each feature. The counterfactual then looks for the cheapest plan: the plan
whose changes have the smallest sum of costs. ``UnitCost`` gives every change the cost 1, so the cheapest plan is the
plan with the fewest changes. The paper ``proofs/papers/recourse/paper.typ`` proves that an optimum of the integer
program is a cheapest plan, and that sorting finds it for unit costs only.

The ideas come from GAM Coach (Wang et al., CHI 2023, arXiv:2302.14165). Every cost is finite and strictly
positive, so a free change cannot pad a plan. ``check_costs`` checks it.
"""

import math
import typing
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import FeatureType, Interval
from ebaml.core.exceptions import RecourseOptionError
from ebaml.local.fact_features import Feature, bin_label


@runtime_checkable
class CostStrategy(Protocol):
    """A way to give a cost to each change of a recourse plan."""

    def costs(self, features: Sequence[Feature]) -> list[list[float]]:
        """Give a cost to each bin of each feature.

        Parameters
        ----------
        features: Sequence[Feature]
            The features of the instance. A strategy may use all of them, for example to put the costs of two kinds of
            features on one scale.

        Returns
        -------
        list[list[float]]
            ``costs[feature][bin]``: the cost of the change of the feature to the bin. The cost of the bin of the
            instance is never used.

        """
        ...

    def describe(self) -> str:
        """Put the strategy in words, for the facts.

        Returns
        -------
        str
            One sentence.

        """
        ...


@attrs.frozen
class UnitCost:
    """Every change costs 1, so the cheapest plan is the plan with the fewest changes.

    Examples
    --------
    >>> from ebaml import UnitCost
    >>> UnitCost().describe()
    'every change costs the same'

    """

    def costs(self, features: Sequence[Feature]) -> list[list[float]]:  # ruff: ignore[no-self-use] -- the Protocol needs a method
        """Give each bin the cost 1.

        Parameters
        ----------
        features: Sequence[Feature]
            The features.

        Returns
        -------
        list[list[float]]
            ``1.0`` for each bin.

        """
        return [[1.0] * len(feature.weights) for feature in features]

    def describe(self) -> str:  # ruff: ignore[no-self-use] -- the Protocol needs a method
        """Put the strategy in words.

        Returns
        -------
        str
            The words.

        """
        return "every change costs the same"


def _weighted_median(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, float(cumulative[-1]) / 2))
    return float(values[order][position])


def _continuous_costs(feature: Feature) -> list[float]:
    # a continuous feature has an interval for each bin
    bin_ranges = typing.cast("Sequence[Interval]", feature.graph.x_vals)
    centers = np.array([(float(low) + float(high)) / 2 for low, high in bin_ranges], dtype=float)
    weights = np.asarray(feature.weights, dtype=float)
    median = _weighted_median(centers, weights)
    deviation = _weighted_median(np.abs(centers - median), weights)
    if deviation <= 0:
        # all the training weight sits in one bin; the mean absolute deviation is then the scale, or 1 if it is 0 too
        deviation = float(np.average(np.abs(centers - median), weights=weights)) or 1.0
    distances = np.abs(centers - centers[feature.bin_index]) / deviation
    return typing.cast("list[float]", distances.tolist())


def _categorical_costs(feature: Feature) -> list[float]:
    weights = np.asarray(feature.weights, dtype=float)
    shares = weights / weights.sum()
    return typing.cast("list[float]", (1 - shares).tolist())


@attrs.frozen
class DataCost:
    """Costs from the training data, as GAM Coach computes them.

    A change of a continuous feature costs the distance from the center of the bin of the instance to the center of
    the new bin, divided by the median absolute deviation of the feature. A change of a nominal or ordinal feature to a
    level costs ``1 - share`` of the level, so a rare level is hard to reach. The categorical costs are then scaled so
    that their mean equals the mean of the continuous costs. The median and the deviation are estimates from the binned
    training data (``bin_weights_``) that the EBM holds, not from the raw data.

    Examples
    --------
    >>> from ebaml import DataCost
    >>> DataCost().describe()
    'each change costs its distance in the training data'

    """

    def costs(self, features: Sequence[Feature]) -> list[list[float]]:  # ruff: ignore[no-self-use] -- the Protocol needs a method
        """Give each bin its cost from the training data.

        Parameters
        ----------
        features: Sequence[Feature]
            The features.

        Returns
        -------
        list[list[float]]
            The cost of each bin of each feature.

        """
        continuous = [feature.graph.feature_type is FeatureType.CONTINUOUS for feature in features]
        costs = [
            _continuous_costs(feature) if is_continuous else _categorical_costs(feature)
            for feature, is_continuous in zip(features, continuous, strict=True)
        ]
        continuous_mean = _mean_of_alternatives(features, costs, continuous, keep=True)
        categorical_mean = _mean_of_alternatives(features, costs, continuous, keep=False)
        if continuous_mean is None or categorical_mean is None:
            return costs
        scale = continuous_mean / categorical_mean
        return [
            feature_costs if is_continuous else [cost * scale for cost in feature_costs]
            for feature_costs, is_continuous in zip(costs, continuous, strict=True)
        ]

    def describe(self) -> str:  # ruff: ignore[no-self-use] -- the Protocol needs a method
        """Put the strategy in words.

        Returns
        -------
        str
            The words.

        """
        return "each change costs its distance in the training data"


def _mean_of_alternatives(
    features: Sequence[Feature],
    costs: Sequence[Sequence[float]],
    continuous: Sequence[bool],
    *,
    keep: bool,
) -> float | None:
    """Return the mean cost of the bins other than the bin of the instance, over the features of one kind.

    Returns
    -------
    float | None
        The mean, or ``None`` if no feature of the kind has such a bin.

    """
    values = [
        cost
        for feature, feature_costs, is_continuous in zip(features, costs, continuous, strict=True)
        if is_continuous is keep
        for bin_index, cost in enumerate(feature_costs)
        if bin_index != feature.bin_index
    ]
    return float(np.mean(values)) if values else None


# nosemgrep: ebaml-return-attrs-not-container -- the keys are feature names and bin texts, which are data
def _copy_bins(per_bin: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {name: dict(costs) for name, costs in per_bin.items()}


@attrs.frozen
class UserCost:
    """Costs from the caller: one cost for each feature, or one for each bin of a feature.

    Attributes
    ----------
    per_feature: Mapping[str, float]
        The cost of any change of a feature, by name.
    per_bin: Mapping[str, Mapping[str, float]]
        The cost of the change of a feature to a bin, by name and by the text of the bin (``[10, 20)`` or a category).
        It wins over ``per_feature``.
    default: float
        The cost of a change that neither field names.

    Examples
    --------
    >>> from ebaml import UserCost
    >>> UserCost(per_feature={"income": 3.0}).describe()
    'each change costs what the caller gave'

    """

    per_feature: Mapping[str, float] = attrs.field(factory=dict, converter=dict)
    per_bin: Mapping[str, Mapping[str, float]] = attrs.field(
        factory=dict,
        converter=_copy_bins,
    )
    default: float = 1.0

    def costs(self, features: Sequence[Feature]) -> list[list[float]]:
        """Give each bin the cost of the caller.

        Parameters
        ----------
        features: Sequence[Feature]
            The features.

        Returns
        -------
        list[list[float]]
            The cost of each bin of each feature.

        """
        return [self._feature_costs(feature) for feature in features]

    def _feature_costs(self, feature: Feature) -> list[float]:
        base = self.per_feature.get(feature.name, self.default)
        bins = self.per_bin.get(feature.name, {})
        return [bins.get(bin_label(x_val), base) for x_val in feature.graph.x_vals]

    def describe(self) -> str:  # ruff: ignore[no-self-use] -- the Protocol needs a method
        """Put the strategy in words.

        Returns
        -------
        str
            The words.

        """
        return "each change costs what the caller gave"


@attrs.frozen
class ScaledCost:
    """A difficulty multiplier for some features, on top of another strategy.

    GAM Coach uses it for a feature that is "easy" or "hard" to change. A feature that cannot change is
    ``RecourseOptions.immutable``, not an infinite multiplier.

    Attributes
    ----------
    base: CostStrategy
        The strategy whose costs the multipliers scale.
    multipliers: Mapping[str, float]
        The multiplier of each feature, by name. A feature with no entry keeps the cost of ``base``.

    Examples
    --------
    >>> from ebaml import ScaledCost, UnitCost
    >>> ScaledCost(UnitCost(), {"income": 2.0}).describe()
    'every change costs the same, times a difficulty for some inputs'

    """

    base: CostStrategy
    multipliers: Mapping[str, float] = attrs.field(factory=dict, converter=dict)

    def costs(self, features: Sequence[Feature]) -> list[list[float]]:
        """Give each bin the cost of ``base`` times the multiplier of its feature.

        Parameters
        ----------
        features: Sequence[Feature]
            The features.

        Returns
        -------
        list[list[float]]
            The cost of each bin of each feature.

        """
        return [
            [cost * self.multipliers.get(feature.name, 1.0) for cost in feature_costs]
            for feature, feature_costs in zip(features, self.base.costs(features), strict=True)
        ]

    def describe(self) -> str:
        """Put the strategy in words.

        Returns
        -------
        str
            The words of ``base`` and of the multipliers.

        """
        return f"{self.base.describe()}, times a difficulty for some inputs"


def check_costs(features: Sequence[Feature], costs: Sequence[Sequence[float]]) -> None:
    """Reject a cost of a reachable bin that is not finite or not strictly positive.

    Parameters
    ----------
    features: Sequence[Feature]
        The features, with their reachable bins.
    costs: Sequence[Sequence[float]]
        The cost of each bin of each feature.

    Raises
    ------
    RecourseOptionError
        If a cost of a bin that the counterfactual can reach is not finite or not above zero.

    """
    for feature, feature_costs in zip(features, costs, strict=True):
        for bin_index in np.flatnonzero(feature.reachable).tolist():
            cost = feature_costs[bin_index]
            if bin_index != feature.bin_index and not (math.isfinite(cost) and cost > 0):
                label = bin_label(feature.graph.x_vals[bin_index])
                msg = f"The cost of the change of {feature.name!r} to {label} is {cost}. A cost must be finite and > 0."
                raise RecourseOptionError(msg)
