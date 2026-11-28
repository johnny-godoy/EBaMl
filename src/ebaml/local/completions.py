"""The completions of a partially known instance: the order that reveals features, and the share that agree."""

from collections.abc import Sequence

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.local.fact_features import Feature
from ebaml.local.fact_types import CompletionStep, LocalOptions
from ebaml.local.sufficient_reasons import Bound, fixing_order, gain_table


def sample_completions(
    features: Sequence[Feature],
    order: Sequence[Feature],
    bounds: Sequence[Bound],
    options: LocalOptions,
) -> list[CompletionStep]:
    """Count completions, estimated by sampling the unknown features from their training bin weights.

    The estimate has its own assumptions. The unknown features are independent. Each one takes every bin, also a bin
    that ``min_support`` leaves out of the other two explanations, with the share of the training weight of the bin.
    The scores are the fitted ones, so ``confidence_level`` does not change a share. One generator with the seed of
    the options draws the bins once, and every step uses the same draws.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    order : Sequence[Feature]
        The features in the order they are revealed.
    bounds : Sequence[Bound]
        The margins of the decision. A completion keeps the decision if every margin holds.
    options : LocalOptions
        Uses ``n_samples``, ``seed`` and ``max_steps``.

    Returns
    -------
    list[CompletionStep]
        The share of completions with the same prediction after revealing 0, 1, 2, ... features.

    """
    rng = np.random.default_rng(options.seed)
    sampled_bins = {
        f.name: rng.choice(len(f.weights), size=options.n_samples, p=f.weights / f.weights.sum()) for f in features
    }
    # the score of the bin of the instance and of each sampled bin, once for each margin; every step reads them
    draws = [[_draw(f.against(bound.columns), sampled_bins[f.name]) for f in features] for bound in bounds]
    steps = []
    for k in range(min(options.max_steps, len(order)) + 1):
        known = {f.name for f in order[:k]}
        holds = np.ones(options.n_samples, dtype=bool)
        for bound, margin_draws in zip(bounds, draws, strict=True):
            # the same additions in the same order at each step, so the shares do not depend on the order of the steps
            total = sum(draw.current if draw.name in known else draw.sampled for draw in margin_draws)
            holds &= (bound.intercept + total > 0) == (bound.sign == 1)
        steps.append(CompletionStep([f.name for f in order[:k]], float(np.mean(holds))))
    return steps


@attrs.frozen
class _Draw:
    """The scores of one feature for one margin: the bin of the instance, and the sampled bins."""

    name: str
    current: float
    sampled: npt.NDArray[np.float64]


def _draw(view: Feature, bins: npt.NDArray[np.intp]) -> _Draw:
    """Read the scores of a feature for one margin.

    Parameters
    ----------
    view : Feature
        The feature as the margin sees it.
    bins : npt.NDArray[np.intp]
        The sampled bins of the feature.

    Returns
    -------
    _Draw
        The score of the bin of the instance, and the score of each sampled bin.

    """
    scores = view.scores
    return _Draw(view.name, float(scores[view.bin_index]), scores[bins])


def _by_smallest_gain(gains: Sequence[Sequence[float]]) -> list[int]:
    """Order the features by the smallest gain over the margins, largest first.

    A feature that helps one margin and does nothing for another one is no help for the claim, so the smallest gain is
    the measure.

    Parameters
    ----------
    gains : Sequence[Sequence[float]]
        For each margin, the gain of each feature.

    Returns
    -------
    list[int]
        The positions of the features.

    """
    feature_count = len(gains[0])
    return sorted(range(feature_count), key=lambda feature: -min(margin_gains[feature] for margin_gains in gains))


def reveal_order(features: Sequence[Feature], bounds: Sequence[Bound], z: float) -> list[Feature]:
    """Order the features by how much knowing their value guarantees, largest first.

    Parameters
    ----------
    features : Sequence[Feature]
        The features to order.
    bounds : Sequence[Bound]
        The margins of the decision. A feature ranks by its smallest gain over these margins.
    z : float
        The number of standard deviations that each score moves against the claim.

    Returns
    -------
    list[Feature]
        The features in the order that the count-completions steps reveal them.

    """
    if len(bounds) == 1:
        views = [f.against(bounds[0].columns) for f in features]
        by_name = {f.name: f for f in features}
        return [by_name[step.feature.name] for step in fixing_order(views, bounds[0].sign, z)]
    return [features[index] for index in _by_smallest_gain(gain_table(features, bounds, z).gains)]
