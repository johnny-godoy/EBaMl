"""The counterfactual of a prediction: the fewest changes that flip the decision, or reach a target class."""

import operator
from collections.abc import Callable, Sequence

import attrs
import numpy as np

from ebaml.local.covering_problem import CoveringProblem
from ebaml.local.decisions import Margin
from ebaml.local.fact_features import Feature, bin_label, robust_scores
from ebaml.local.fact_types import Change, Counterfactual, LocalOptions
from ebaml.local.local_kinds import LocalKind, Outcome, PredictionFrame
from ebaml.local.minimality import MinimalityStatus


def _counterfactual(
    features: Sequence[Feature],
    intercept: float,
    sign: int,
    z: float,
    *,
    after: Callable[[float, Sequence[tuple[Feature, int]]], Outcome],
) -> Counterfactual | None:
    """Minimum change required: the features with the largest possible drop in favour of the outcome, until it flips.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    intercept : float
        The intercept of the model minus the threshold of the decision.
    sign : int
        ``1`` if the decision says that the score is high, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.
    after : Callable[[float, Sequence[tuple[Feature, int]]], Outcome]
        The decision and the output of the model after the changes, from the margin after the changes and from each
        changed feature with the bin that it takes.

    Returns
    -------
    Counterfactual | None
        The changes, or None if no changes flip the prediction.

    """
    margin = sign * intercept + sum(sign * f.current for f in features)
    drops = []
    for feature in features:
        pessimistic = robust_scores(feature, sign, z, towards_outcome=False)
        best = int(np.argmin(pessimistic))
        drops.append((sign * feature.current - float(pessimistic[best]), best, feature))
    ranked = sorted(drops, key=operator.itemgetter(0), reverse=True)
    # the margin left after the first 1, 2, ... changes; the prediction flips at the first one below zero
    flips = np.flatnonzero(margin - np.cumsum([drop for drop, _, _ in ranked]) < 0)
    if not flips.size:
        return None
    changes = ranked[: flips[0] + 1]
    new_margin = margin + sign * sum(float(f.scores[best]) - f.current for _, best, f in changes)
    outcome = after(new_margin, [(f, best) for _, best, f in changes])
    return Counterfactual(
        changes=[
            Change(f.name, f.label, bin_label(f.graph.x_vals[best]), float(f.scores[best]) - f.current)
            for _, best, f in changes
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimality=MinimalityStatus.SORTING_PROOF,  # sorting finds the smallest set of one margin, as the paper proves
        lower_bound=len(changes),
    )


def _smallest_counterfactual(
    features: Sequence[Feature],
    frame: PredictionFrame,
    scores: Sequence[float],
    intercepts: Sequence[float],
    z: float,
) -> Counterfactual | None:
    """Find the fewest changes after which one margin of the decision fails.

    The decision fails when one margin fails, so the answer is the smallest of the answers for each margin. Each is a
    problem with one margin, which sorting solves exactly.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    frame : PredictionFrame
        The decision, with its margins.
    scores : Sequence[float]
        The score of each output of the model for the instance.
    intercepts : Sequence[float]
        The intercept of each output of the model.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    Counterfactual | None
        The counterfactual with the fewest changes, the first one on a tie, or None if no margin can fail.

    """
    found = []
    for margin in frame.side.margins:

        def after(
            _new_margin: float,
            replaced: Sequence[tuple[Feature, int]],
            margin: Margin = margin,
        ) -> Outcome:
            new_scores = np.array(scores, dtype=float)
            for feature, bin_index in replaced:
                new_scores += feature.outputs(bin_index) - feature.outputs(feature.bin_index)
            return frame.after(margin, new_scores.tolist())

        counterfactual = _counterfactual(
            [feature.against(margin.columns) for feature in features],
            margin.offset(intercepts),
            margin.sign,
            z,
            after=after,
        )
        if counterfactual is not None:
            found.append(counterfactual)
    return min(found, key=lambda counterfactual: len(counterfactual.changes), default=None)


@attrs.frozen
class _Alternatives:
    """The bins that one feature can move to, and how much each bin raises each margin of a target class."""

    bins: list[int]
    gains: list[list[float]]


def _alternatives(views: Sequence[Feature], z: float) -> _Alternatives:
    """List the plausible bins of a feature, other than the bin of the instance, with their gains for each margin.

    The scores of the new bin move by ``z`` deviations against the target class, as in ``_counterfactual``. A gain can
    be negative: a bin that lowers one margin can still be part of the smallest set.

    Parameters
    ----------
    views : Sequence[Feature]
        The feature as seen by each margin of the target class.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    _Alternatives
        The bins and, for each bin, its gain for each margin.

    """
    first = views[0]
    bins = [int(bin_index) for bin_index in np.flatnonzero(first.plausible) if bin_index != first.bin_index]
    gains = [
        [float(view.scores[bin_index] - z * view.stds[bin_index] - view.current) for view in views]
        for bin_index in bins
    ]
    return _Alternatives(bins, gains)


@attrs.frozen
class Totals:
    """The score and the intercept of each output of the model for the instance, and the confidence factor ``z``."""

    scores: list[float]
    intercepts: list[float]
    z: float


def _target_counterfactual(
    features: Sequence[Feature],
    frame: PredictionFrame,
    margins: Sequence[Margin],
    totals: Totals,
    *,
    options: LocalOptions,
) -> Counterfactual | None:
    """Find the fewest changes after which every margin of the target class holds.

    Each feature is a group, and each of its plausible bins other than the bin of the instance is an alternative. A set
    of changes moves each changed feature to one bin. ``options.search`` solves this covering problem over every choice
    of bins.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    frame : PredictionFrame
        The prediction.
    margins : Sequence[Margin]
        The margins of the target class against every other class.
    totals : Totals
        The scores and the intercepts of the instance, and the standard deviations to move the scores by.
    options : LocalOptions
        Uses ``search`` and ``seed``.

    Returns
    -------
    Counterfactual | None
        The changes, or None if the search proves that no choice of bins makes every margin hold.

    """
    target, _ = margins[0].columns or (0, 0)
    views = [[feature.against(margin.columns) for feature in features] for margin in margins]
    alternatives = [
        _alternatives([margin_views[index] for margin_views in views], totals.z)
        # nosemgrep: ebaml-no-range-len -- the index addresses the feature in the views of every margin
        for index in range(len(features))
    ]
    worst = [
        margin.offset(totals.intercepts) + sum(view.current for view in margin_views)
        for margin, margin_views in zip(margins, views, strict=True)
    ]
    problem = CoveringProblem(worst, [feature_alternatives.gains for feature_alternatives in alternatives])
    cover = options.search.solve(problem, seed=options.seed)
    if cover is None:
        return None
    changed = [
        (features[choice.group], alternatives[choice.group].bins[choice.alternative]) for choice in cover.choices
    ]
    new_scores = np.array(totals.scores, dtype=float)
    for feature, bin_index in changed:
        new_scores += feature.outputs(bin_index) - feature.outputs(feature.bin_index)
    outcome = frame.after(margins[0], new_scores.tolist())
    return Counterfactual(
        changes=[
            Change(
                feature.name,
                feature.label,
                bin_label(feature.graph.x_vals[bin_index]),
                float(
                    (feature.outputs(bin_index) - feature.outputs(feature.bin_index))[target]
                    - (feature.outputs(bin_index) - feature.outputs(feature.bin_index))[frame.column],
                ),
            )
            for feature, bin_index in changed
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimality=cover.minimality,
        lower_bound=cover.lower_bound,
    )


def choose_counterfactual(
    features: Sequence[Feature],
    frame: PredictionFrame,
    totals: Totals,
    *,
    kind: LocalKind,
    options: LocalOptions,
) -> Counterfactual | None:
    """Find the counterfactual that the options ask for: the target class if there is one, else the smallest.

    Parameters
    ----------
    features : Sequence[Feature]
        The features.
    frame : PredictionFrame
        The prediction.
    totals : Totals
        The scores and the intercepts of the instance, and the standard deviations to move the scores by.
    kind : LocalKind
        The kind of the model.
    options : LocalOptions
        Uses ``target_class``, ``search`` and ``seed``.

    Returns
    -------
    Counterfactual | None
        The counterfactual, or None if no change gives it.

    """
    if options.target_class is None:
        return _smallest_counterfactual(features, frame, totals.scores, totals.intercepts, totals.z)
    margins = kind.target_margins(options.target_class, frame)
    return _target_counterfactual(features, frame, margins, totals, options=options)
