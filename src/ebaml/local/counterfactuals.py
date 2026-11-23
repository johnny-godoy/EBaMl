"""The counterfactual of a prediction: the fewest changes that flip the decision, or reach a target class."""

import operator
from collections.abc import Callable, Sequence

import attrs
import numpy as np

from ebaml.local.costs import check_costs
from ebaml.local.covering_problem import Cover, CoveringProblem
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
            Change(f.name, f.label, bin_label(f.graph.x_vals[best]), float(f.scores[best]) - f.current, best)
            for _, best, f in changes
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimality=MinimalityStatus.SORTING_PROOF,  # sorting finds the smallest set of one margin, as the paper proves
        cost=float(len(changes)),
        cost_lower_bound=float(len(changes)),
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
    """List the bins that a feature can reach, other than the bin of the instance, with their gains for each margin.

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
    bins = [int(bin_index) for bin_index in np.flatnonzero(first.reachable) if bin_index != first.bin_index]
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
    search: "_Search",
) -> Counterfactual | None:
    """Find the fewest changes, or the cheapest plan, after which every margin of the target class holds.

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
    search : _Search
        Uses ``search`` and ``seed`` of the options, the costs of the bins when they are not all equal, and the cap.

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
    problem = search.problem(worst, alternatives)
    cover = search.options.search.solve(problem, seed=search.options.seed)
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
                bin_index,
            )
            for feature, bin_index in changed
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimality=cover.minimality,
        cost=problem.cost(cover.choices),
        cost_lower_bound=_cost_lower_bound(problem, cover),
    )


@attrs.frozen
class _Limits:
    """The costs of the bins of each feature, ``None`` when every reachable change costs the same, and the cap.

    ``unit`` is the cost of one change when every reachable change costs the same, and ``1`` otherwise.
    """

    costs: list[list[float]] | None
    max_changes: int | None
    unit: float = 1.0


@attrs.frozen
class _Search:
    """The options of the search and the limits that the recourse options give."""

    options: LocalOptions
    limits: _Limits

    def problem(self, worst: Sequence[float], alternatives: Sequence["_Alternatives"]) -> CoveringProblem:
        """Build the covering problem of the alternatives, with their costs and the cap.

        Returns
        -------
        CoveringProblem
            The problem.

        """
        costs = self.limits.costs
        return CoveringProblem(
            worst,
            [feature_alternatives.gains for feature_alternatives in alternatives],
            costs=None if costs is None else _alternative_costs(alternatives, costs),
            max_changes=self.limits.max_changes,
        )


def _alternative_costs(alternatives: Sequence[_Alternatives], costs: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [costs[index][bin_index] for bin_index in feature_alternatives.bins]
        for index, feature_alternatives in enumerate(alternatives)
    ]


def _flip_alternatives(views: Sequence[Feature], sign: int, z: float) -> list[_Alternatives]:
    """List the reachable bins of each feature with how much each one lowers the robust margin of one margin.

    The gain of a bin is the signed score of the instance minus the pessimistic signed score of the bin, as the drop
    of ``_counterfactual`` and the gain of Section 4 of ``proofs/papers/recourse/paper.typ``.

    Returns
    -------
    list[_Alternatives]
        The alternatives of each feature, with one gain each.

    """
    alternatives = []
    for view in views:
        pessimistic = robust_scores(view, sign, z, towards_outcome=False)
        bins = [int(bin_index) for bin_index in np.flatnonzero(view.reachable) if bin_index != view.bin_index]
        alternatives.append(
            _Alternatives(bins, [[sign * view.current - float(pessimistic[bin_index])] for bin_index in bins]),
        )
    return alternatives


def _cheapest_flip(
    features: Sequence[Feature],
    frame: PredictionFrame,
    totals: Totals,
    search: _Search,
) -> Counterfactual | None:
    """Find the cheapest plan after which one margin of the decision fails, with ``options.search``.

    Each margin is a covering problem with one margin. The answer is the cheapest of the plans of the margins.

    Returns
    -------
    Counterfactual | None
        The cheapest plan, or None if the search proves that no plan within the cap flips a margin.

    """
    found = []
    for margin in frame.side.margins:
        views = [feature.against(margin.columns) for feature in features]
        alternatives = _flip_alternatives(views, margin.sign, totals.z)
        robust = margin.sign * margin.offset(totals.intercepts) + sum(margin.sign * view.current for view in views)
        problem = search.problem([-robust], alternatives)
        cover = search.options.search.solve(problem, seed=search.options.seed)
        if cover is None:
            continue
        changed = [
            (views[choice.group], alternatives[choice.group].bins[choice.alternative]) for choice in cover.choices
        ]
        new_scores = np.array(totals.scores, dtype=float)
        for view, bin_index in changed:
            new_scores += view.outputs(bin_index) - view.outputs(view.bin_index)
        outcome = frame.after(margin, new_scores.tolist())
        found.append(
            Counterfactual(
                changes=[
                    Change(
                        view.name,
                        view.label,
                        bin_label(view.graph.x_vals[bin_index]),
                        float(view.scores[bin_index]) - view.current,
                        bin_index,
                    )
                    for view, bin_index in changed
                ],
                new_decision=outcome.decision,
                new_output=outcome.output,
                minimality=cover.minimality,
                cost=problem.cost(cover.choices),
                cost_lower_bound=_cost_lower_bound(problem, cover),
            ),
        )
    return min(found, key=lambda counterfactual: counterfactual.cost, default=None)


def _limits(features: Sequence[Feature], options: LocalOptions) -> _Limits:
    """Read the costs and the cap of the recourse options, and check the costs.

    Returns
    -------
    _Limits
        The limits.

    """
    costs = options.recourse.costs.costs(features)
    check_costs(features, costs)
    reachable = {
        feature_costs[bin_index]
        for feature, feature_costs in zip(features, costs, strict=True)
        for bin_index in np.flatnonzero(feature.reachable).tolist()
        if bin_index != feature.bin_index
    }
    if len(reachable) <= 1:
        return _Limits(None, options.recourse.max_changes, next(iter(reachable), 1.0))
    return _Limits(costs, options.recourse.max_changes)


def _cost_lower_bound(problem: CoveringProblem, cover: Cover) -> float:
    """Return a cost that no plan beats: the cost of an optimal plan, else the size bound times the cheapest pick.

    Every plan has at least ``cover.lower_bound`` picks, and each pick costs at least the cheapest alternative.

    Returns
    -------
    float
        The bound.

    """
    if cover.minimality.is_optimal:
        return problem.cost(cover.choices)
    cheapest = min((cost for costs in problem.costs for cost in costs), default=1.0) if problem.costs else 1.0
    return cover.lower_bound * cheapest


def _exclude(features: Sequence[Feature], plan: Counterfactual) -> list[Feature]:
    """Remove from the reachable bins each pair of a feature and a bin that a plan uses, as the code of GAM Coach does.

    Returns
    -------
    list[Feature]
        The features, whose reachable bins leave out the bins of the plan.

    """
    used: dict[str, list[int]] = {}
    for change in plan.changes:
        used.setdefault(change.feature, []).append(change.to_bin)
    excluded = []
    for feature in features:
        if feature.name not in used:
            excluded.append(feature)
            continue
        reachable = feature.reachable.copy()
        reachable[used[feature.name]] = False
        excluded.append(attrs.evolve(feature, reachable=reachable))
    return excluded


def _one_plan(
    features: Sequence[Feature],
    frame: PredictionFrame,
    totals: Totals,
    kind: LocalKind,
    options: LocalOptions,
) -> Counterfactual | None:
    """Find one plan: the target class if there is one, else a flip, with equal costs or not.

    Returns
    -------
    Counterfactual | None
        The plan, or None if no plan exists under the options.

    """
    limits = _limits(features, options)
    if options.target_class is not None:
        margins = kind.target_margins(options.target_class, frame)
        found = _target_counterfactual(features, frame, margins, totals, _Search(options, limits))
    elif limits.costs is None:
        found = _smallest_counterfactual(features, frame, totals.scores, totals.intercepts, totals.z)
        # every plan has at least the changes that sorting finds (Theorem 4 of the recourse paper), so a cap below
        # them proves that no plan exists
        if found is not None and limits.max_changes is not None and len(found.changes) > limits.max_changes:
            found = None
    else:
        found = _cheapest_flip(features, frame, totals, _Search(options, limits))
    if found is None:
        return None
    return attrs.evolve(found, cost=found.cost * limits.unit, cost_lower_bound=found.cost_lower_bound * limits.unit)


def choose_counterfactuals(
    features: Sequence[Feature],
    frame: PredictionFrame,
    totals: Totals,
    *,
    kind: LocalKind,
    options: LocalOptions,
) -> list[Counterfactual]:
    """Find the plans that the options ask for: the target class if there is one, else a flip.

    With equal costs, a plan is the fewest changes, which sorting finds exactly. With other costs, it is the cheapest
    plan, which ``options.search`` finds. ``RecourseOptions.max_changes`` caps the number of changes. Each plan after
    the first is the cheapest plan that uses no pair of a feature and a bin of an earlier plan (Theorem 3 of
    ``proofs/papers/recourse/paper.typ``).

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
        Uses ``target_class``, ``search``, ``seed`` and ``recourse``.

    Returns
    -------
    list[Counterfactual]
        Up to ``RecourseOptions.plans`` plans, in the order found, so their costs do not go down. The list stops at the
        first search that proves that no further plan exists.

    """
    plans: list[Counterfactual] = []
    current = list(features)
    for _ in range(options.recourse.plans):
        plan = _one_plan(current, frame, totals, kind, options)
        if plan is None:
            break
        plans.append(plan)
        current = _exclude(current, plan)
    return plans
