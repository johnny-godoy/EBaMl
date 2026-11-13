"""The sufficient reason of a prediction: the features that fix the decision whatever the other inputs are."""

import operator
from collections.abc import Sequence

import attrs
import numpy as np

from ebaml.local.covering_problem import CoveringProblem
from ebaml.local.covering_solvers import CoveringSolver
from ebaml.local.fact_features import Feature, robust_scores
from ebaml.local.fact_types import LocalOptions, SufficientReason
from ebaml.local.minimality import MinimalityStatus


@attrs.frozen
class _Gain:
    """What knowing the value of one feature guarantees."""

    gain: float
    feature: Feature


def fixing_order(features: Sequence[Feature], sign: int, z: float) -> list[_Gain]:
    """Features by how much knowing their value guarantees, largest first.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    list[_Gain]
        The guaranteed gain and the feature, largest gain first.

    """
    gains = []
    for feature in features:
        guaranteed = robust_scores(feature, sign, z, towards_outcome=True)
        gains.append(_Gain(float(guaranteed[feature.bin_index] - guaranteed.min()), feature))
    return sorted(gains, key=operator.attrgetter("gain"), reverse=True)


def _sufficient_reason(features: Sequence[Feature], intercept: float, sign: int, z: float) -> SufficientReason | None:
    """Minimum sufficient reason: fix the features that guarantee the most until no completion can flip the outcome.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    intercept : float
        The intercept of the model, moved by the threshold of the decision (zero for a classifier).
    sign : int
        ``1`` if the decision says that the score is high, ``-1`` if it says that the score is low.
    z : float
        The number of standard deviations that each score moves against the claim.

    Returns
    -------
    SufficientReason | None
        The features to fix, or None if even all of them do not guarantee the outcome.

    """
    worst = sign * intercept + sum(float(robust_scores(f, sign, z, towards_outcome=True).min()) for f in features)
    order = fixing_order(features, sign, z)
    # the guaranteed margin after fixing the first 0, 1, 2, ... features; the outcome is certain at the first above zero
    margins = worst + np.concatenate(([0.0], np.cumsum([step.gain for step in order])))
    enough = np.flatnonzero(margins > 0)
    if not enough.size:
        return None
    count = int(enough[0])
    return SufficientReason(
        features=[step.feature.name for step in order[:count]],
        guaranteed_margin=float(margins[count]),
        minimality=MinimalityStatus.SORTING_PROOF,  # sorting finds the smallest set of one margin, as the paper proves
        lower_bound=count,
    )


@attrs.frozen
class Bound:
    """One margin of the decision: ``sign * (intercept + the scores)`` is positive while the margin holds.

    ``columns`` names the two outputs that a margin of a multiclass classifier compares, and ``None`` otherwise.
    """

    intercept: float
    sign: int
    columns: tuple[int, int] | None = None


@attrs.frozen
class GainTable:
    """The numbers of the covering problem of a decision with several margins."""

    worst: list[float]
    gains: list[list[float]]


def gain_table(features: Sequence[Feature], bounds: Sequence[Bound], z: float) -> GainTable:
    """Read, for each margin, its worst case and how much each feature raises it when fixed.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    bounds : Sequence[Bound]
        The margins of the decision.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    GainTable
        The worst cases and the gains.

    """
    worst: list[float] = []
    gains: list[list[float]] = []
    for bound in bounds:
        guaranteed = [robust_scores(f.against(bound.columns), bound.sign, z, towards_outcome=True) for f in features]
        worst.append(bound.sign * bound.intercept + sum(float(scores.min()) for scores in guaranteed))
        gains.append([
            float(scores[f.bin_index] - scores.min()) for f, scores in zip(features, guaranteed, strict=True)
        ])
    return GainTable(worst, gains)


def _several_margins_reason(
    features: Sequence[Feature],
    table: GainTable,
    search: CoveringSolver,
    seed: int,
) -> SufficientReason | None:
    """Choose a sufficient reason for a decision with several margins, with the search of the user.

    Each feature is a group with one alternative: the feature keeps its value.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    table : GainTable
        The worst cases and the gains.
    search : CoveringSolver
        How to choose the features.
    seed : int
        The seed of the solver.

    Returns
    -------
    SufficientReason | None
        The reason, or None if even all the features do not guarantee the decision.

    """
    groups = [
        [[margin_gains[index] for margin_gains in table.gains]]
        # nosemgrep: ebaml-no-range-len -- the index addresses the feature in the gains of every margin
        for index in range(len(features))
    ]
    cover = search.solve(CoveringProblem(table.worst, groups), seed=seed)
    if cover is None:
        return None
    return SufficientReason(
        features=[features[choice.group].name for choice in cover.choices],
        guaranteed_margin=cover.guaranteed_margin,
        minimality=cover.minimality,
        lower_bound=cover.lower_bound,
    )


def find_reason(
    features: Sequence[Feature],
    bounds: Sequence[Bound],
    z: float,
    options: LocalOptions,
) -> SufficientReason | None:
    """Choose the sufficient reason of a decision.

    One margin has a smallest set that sorting finds exactly. Several margins use the search of the user.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    bounds : Sequence[Bound]
        The margins of the decision.
    z : float
        Standard deviations to move the scores by.
    options : LocalOptions
        Uses ``search`` and ``seed`` when the decision has several margins.

    Returns
    -------
    SufficientReason | None
        The reason, or None if even all the features do not guarantee the decision.

    """
    if len(bounds) == 1:
        views = [f.against(bounds[0].columns) for f in features]
        return _sufficient_reason(views, bounds[0].intercept, bounds[0].sign, z)
    return _several_margins_reason(features, gain_table(features, bounds, z), options.search, options.seed)
