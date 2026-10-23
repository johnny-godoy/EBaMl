"""Deterministic local explanations of one EBM prediction.

The three explanations are the explainability queries of Barceló, Monet, Pérez and Subercaseaux, "Model
Interpretability through the Lens of Computational Complexity" (arXiv:2010.12265):

- *Minimum change required* (counterfactual): the fewest inputs to change so that the prediction flips.
- *Minimum sufficient reason*: the fewest inputs that fix the prediction whatever the other inputs are.
- *Count completions*: the share of completions of a partially known input that get the same prediction.

An EBM without interactions is a sum of one score per feature, so the first two are solved exactly by sorting the
features by how much they move the score, and the third by sampling the unknown features from their training
distribution with a fixed seed. Nothing here calls an LLM: the facts are rendered as JSON for the prompts in
``baml_src/local_explanations.baml``.

The confidence of the model enters through ``confidence_level``: every score is moved by ``z`` standard deviations in
the direction that makes the claim harder to prove, so a counterfactual flips the prediction, and a sufficient reason
keeps it, even if the scores sit at the edge of their confidence interval.
"""

import math
import operator
import statistics
from collections.abc import Mapping
from typing import Any

import attrs
import numpy as np
import orjson

from ebaml.custom_types import Audience, ExplainableBoostingModel
from ebaml.graphs import EBMGraph

_LIKELIHOOD_WORDS = (
    (0.95, "almost certain"),
    (0.8, "very likely"),
    (0.6, "likely"),
    (0.4, "about as likely as not"),
    (0.2, "unlikely"),
    (0.05, "very unlikely"),
)
_N_CLASSES = 2
_STRENGTH_WORDS = ((0.5, "strongly"), (0.2, "moderately"))


@attrs.frozen
class _Feature:
    """The graph of a feature plus the bin the instance falls in."""

    name: str
    graph: EBMGraph
    weights: np.ndarray
    bin_index: int
    value: object
    plausible: np.ndarray  # bins with enough training support to be offered as alternatives

    @property
    def current(self) -> float:
        """Score of the bin of the instance."""
        return float(self.graph.scores[self.bin_index])

    @property
    def label(self) -> str:
        """The value of the instance, as the user gave it."""
        return f"{self.value:.4g}" if isinstance(self.value, float) else str(self.value)


@attrs.frozen
class LocalOptions:
    """Options of the local facts.

    Attributes
    ----------
    confidence_level: float
        Confidence level of the claims. ``0`` ignores the uncertainty of the scores.
    max_steps: int
        The most features to reveal one by one in the count-completions steps.
    n_samples: int
        Samples used to estimate the share of completions.
    seed: int
        Seed of the sampling, so that the facts are reproducible.
    min_support: float
        Share of the training data a value needs to be offered as an alternative in the counterfactual and as a
        possible value of an unknown feature in the sufficient reason: a category needs this share, and the lowest and
        highest ``min_support`` of a continuous feature are trimmed. It keeps rare values, with extreme scores, out of
        the explanations.

    """

    confidence_level: float = 0.95
    max_steps: int = 5
    n_samples: int = 10_000
    seed: int = 0
    min_support: float = 0.01


@attrs.frozen
class Prediction:
    """The prediction to explain, with the confidence interval of its probability."""

    label: object
    probability: float
    lower: float
    upper: float
    logit: float
    logit_std: float


@attrs.frozen
class Contribution:
    """The effect of one input on the prediction."""

    feature: str
    value: object
    score: float
    std: float


@attrs.frozen
class Change:
    """One input to change in a counterfactual."""

    feature: str
    from_value: object
    to_value: str
    score_change: float


@attrs.frozen
class Counterfactual:
    """The smallest set of changes that flips the prediction at the requested confidence."""

    changes: list[Change]
    new_probability: float  # probability of the class the prediction flips to


@attrs.frozen
class SufficientReason:
    """A smallest set of inputs that fixes the prediction whatever the other inputs are."""

    features: list[str]
    guaranteed_margin: float


@attrs.frozen
class CompletionStep:
    """The share of completions with the same prediction when only ``fixed`` features are known."""

    fixed: list[str]
    share: float


@attrs.frozen
class LocalFacts:
    """Everything computed about one prediction."""

    prediction: Prediction
    contributions: list[Contribution]
    counterfactual: Counterfactual | None
    sufficient_reason: SufficientReason | None
    completions: list[CompletionStep]
    confidence_level: float


def _bin_label(x_val: object) -> str:
    """Render a bin: ``[lo, hi)`` for continuous features, the category otherwise.

    Parameters
    ----------
    x_val: object
        An interval ``(lo, hi)`` or a category.

    Returns
    -------
    str
        The text of the bin.

    """
    if isinstance(x_val, tuple):
        return f"[{x_val[0]:.4g}, {x_val[1]:.4g})"
    return str(x_val)


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def _bin_index(ebm: ExplainableBoostingModel, feature_index: int, graph: EBMGraph, value: object) -> int:
    """Find the bin of ``value`` in the graph of a feature.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature.
    graph: EBMGraph
        The graph of the feature.
    value: object
        The value of the instance.

    Returns
    -------
    int
        The index of the bin in the graph.

    Raises
    ------
    ValueError
        If the value is missing or the category was not seen in training.

    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        msg = f"Feature {graph.feature_name!r} is missing; local explanations need every input."
        raise ValueError(msg)
    if graph.feature_type == "continuous":
        cuts = ebm.bins_[feature_index][0]
        return int(np.searchsorted(cuts, float(value), side="right"))
    if value not in graph.x_vals:
        msg = f"Unknown category {value!r} for feature {graph.feature_name!r}."
        raise ValueError(msg)
    return graph.x_vals.index(value)


def _plausible_bins(weights: np.ndarray, *, continuous: bool, min_support: float) -> np.ndarray:
    """Bins with enough training support to be offered as alternatives.

    A category needs ``min_support`` of the training data. The bins of a continuous feature are thin slices, so instead
    the lowest and highest ``min_support`` of the data are trimmed off.

    Parameters
    ----------
    weights: np.ndarray
        Training weight of each bin.
    continuous: bool
        Whether the feature is continuous.
    min_support: float
        Share of the training data a value needs.

    Returns
    -------
    np.ndarray
        For each bin, whether it can be offered as an alternative.

    """
    share = weights / weights.sum()
    if not continuous:
        return share >= min_support
    upper_edge = np.cumsum(share)
    lower_edge = upper_edge - share
    return (upper_edge > min_support) & (lower_edge < 1 - min_support)


def _features(ebm: ExplainableBoostingModel, instance: Mapping[str, Any], min_support: float) -> list[_Feature]:
    """Read the graph, the training weights and the bin of the instance of every feature.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    instance: Mapping[str, Any]
        The value of every feature.
    min_support: float
        See ``LocalOptions.min_support``.

    Returns
    -------
    list[_Feature]
        One entry per feature.

    Raises
    ------
    ValueError
        If the model is not a binary classifier without interactions, or the instance is incomplete.

    """
    if len(ebm.term_features_) != len(ebm.feature_names_in_) or any(len(t) != 1 for t in ebm.term_features_):
        msg = "Local explanations need an EBM without interactions (fit with interactions=0)."
        raise ValueError(msg)
    if len(ebm.classes_) != _N_CLASSES:
        msg = "Local explanations need a binary classifier."
        raise ValueError(msg)
    features = []
    for index, name in enumerate(ebm.feature_names_in_):
        if name not in instance:
            msg = f"The instance has no value for feature {name!r}."
            raise ValueError(msg)
        graph = EBMGraph.from_ebm(ebm, index)
        weights = np.asarray(ebm.bin_weights_[index][1:-1], dtype=float)
        bin_index = _bin_index(ebm, index, graph, instance[name])
        plausible = _plausible_bins(weights, continuous=graph.feature_type == "continuous", min_support=min_support)
        plausible[bin_index] = True
        features.append(_Feature(name, graph, weights, bin_index, instance[name], plausible))
    return features


def _robust_scores(feature: _Feature, sign: int, z: float, *, towards_outcome: bool) -> np.ndarray:
    """Scores of every bin in the direction of the outcome (``sign``), moved by ``z`` standard deviations.

    With ``towards_outcome`` the scores are lowered, so they are what is guaranteed in favour of the outcome; without
    it they are raised, so they are the most the outcome can still be argued against. Bins without enough support are
    ``inf``, so that taking the minimum skips them.

    Parameters
    ----------
    feature : _Feature
        The feature.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.
    towards_outcome : bool
        Lower the scores (what is guaranteed) instead of raising them.

    Returns
    -------
    np.ndarray
        The score of each bin.

    """
    scores = sign * np.asarray(feature.graph.scores, dtype=float)
    stds = z * np.asarray(feature.graph.stds, dtype=float)
    return np.where(feature.plausible, scores - stds if towards_outcome else scores + stds, np.inf)


def _counterfactual(features: list[_Feature], intercept: float, sign: int, z: float) -> Counterfactual | None:
    """Minimum change required: the features with the largest possible drop in favour of the outcome, until it flips.

    Parameters
    ----------
    features : list[_Feature]
        The features.
    intercept : float
        The intercept of the model.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    Counterfactual | None
        The changes, or None if no changes flip the prediction.

    """
    margin = sign * intercept + sum(sign * f.current for f in features)
    drops = []
    for feature in features:
        pessimistic = _robust_scores(feature, sign, z, towards_outcome=False)
        best = int(np.argmin(pessimistic))
        drops.append((sign * feature.current - float(pessimistic[best]), best, feature))
    changes = []
    for drop, best, feature in sorted(drops, key=operator.itemgetter(0), reverse=True):
        if margin < 0 or drop <= 0:
            break
        margin -= drop
        changes.append((feature, best, drop))
    if margin >= 0:
        return None
    new_logit = intercept + sum(f.current for f in features)
    for feature, best, _ in changes:
        new_logit += float(feature.graph.scores[best]) - feature.current
    new_probability = _sigmoid(new_logit) if sign < 0 else 1 - _sigmoid(new_logit)
    return Counterfactual(
        changes=[
            Change(f.name, f.label, _bin_label(f.graph.x_vals[best]), float(f.graph.scores[best]) - f.current)
            for f, best, _ in changes
        ],
        new_probability=new_probability,
    )


def _fixing_order(features: list[_Feature], sign: int, z: float) -> list[tuple[float, _Feature]]:
    """Features by how much knowing their value guarantees, largest first.

    Parameters
    ----------
    features : list[_Feature]
        The features.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    list[tuple[float, _Feature]]
        The guaranteed gain and the feature, largest gain first.

    """
    gains = []
    for feature in features:
        guaranteed = _robust_scores(feature, sign, z, towards_outcome=True)
        gains.append((float(guaranteed[feature.bin_index] - guaranteed.min()), feature))
    return sorted(gains, key=operator.itemgetter(0), reverse=True)


def _sufficient_reason(features: list[_Feature], intercept: float, sign: int, z: float) -> SufficientReason | None:
    """Minimum sufficient reason: fix the features that guarantee the most until no completion can flip the outcome.

    Parameters
    ----------
    features : list[_Feature]
        The features.
    intercept : float
        The intercept of the model.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    SufficientReason | None
        The features to fix, or None if even all of them do not guarantee the outcome.

    """
    worst = sign * intercept + sum(float(_robust_scores(f, sign, z, towards_outcome=True).min()) for f in features)
    chosen = []
    for gain, feature in _fixing_order(features, sign, z):
        if worst > 0:
            break
        worst += gain
        chosen.append(feature.name)
    if worst <= 0:
        return None
    return SufficientReason(features=chosen, guaranteed_margin=worst)


def _completions(
    features: list[_Feature],
    order: list[_Feature],
    intercept: float,
    sign: int,
    options: LocalOptions,
) -> list[CompletionStep]:
    """Count completions, estimated by sampling the unknown features from their training bin weights.

    Parameters
    ----------
    features : list[_Feature]
        The features.
    order : list[_Feature]
        The features in the order they are revealed.
    intercept : float
        The intercept of the model.
    sign : int
        ``1`` if the outcome is the positive class, ``-1`` otherwise.
    options : LocalOptions
        Uses ``n_samples``, ``seed`` and ``max_steps``.

    Returns
    -------
    list[CompletionStep]
        The share of completions with the same prediction after revealing 0, 1, 2, ... features.

    """
    rng = np.random.default_rng(options.seed)
    sampled = {}
    for f in features:
        bins = rng.choice(len(f.weights), size=options.n_samples, p=f.weights / f.weights.sum())
        sampled[f.name] = np.asarray(f.graph.scores, dtype=float)[bins]
    current = {f.name: f.current for f in features}
    steps = []
    for k in range(min(options.max_steps, len(order)) + 1):
        known = {f.name for f in order[:k]}
        logit = intercept + sum(current[n] if n in known else sampled[n] for n in current)
        steps.append(CompletionStep([f.name for f in order[:k]], float(np.mean(sign * logit > 0))))
    return steps


def compute_local_facts(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, Any],
    options: LocalOptions | None = None,
) -> LocalFacts:
    """Compute the three local explanations of one prediction of a binary EBM without interactions.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    instance: Mapping[str, Any]
        The value of every feature, by feature name.
    options: LocalOptions, optional
        Confidence level, sampling and support options.

    Returns
    -------
    LocalFacts
        The prediction, the contributions and the three explanations.

    Raises
    ------
    ValueError
        If the model is not a binary classifier without interactions, or the instance is incomplete.

    """
    options = options or LocalOptions()
    if not 0 <= options.confidence_level < 1:
        msg = f"confidence_level must be in [0, 1). Got {options.confidence_level}."
        raise ValueError(msg)
    z = statistics.NormalDist().inv_cdf((1 + options.confidence_level) / 2)
    features = _features(ebm, instance, options.min_support)
    intercept = float(np.ravel(ebm.intercept_)[0])
    logit = intercept + sum(f.current for f in features)
    logit_std = math.sqrt(sum(float(f.graph.stds[f.bin_index]) ** 2 for f in features))
    sign = 1 if logit > 0 else -1
    probability = _sigmoid(logit)
    interval = sorted((_sigmoid(logit - z * logit_std), _sigmoid(logit + z * logit_std)))
    prediction = Prediction(
        label=ebm.classes_[1 if sign > 0 else 0].item(),
        probability=probability if sign > 0 else 1 - probability,
        lower=interval[0] if sign > 0 else 1 - interval[1],
        upper=interval[1] if sign > 0 else 1 - interval[0],
        logit=logit,
        logit_std=logit_std,
    )
    order = [f for _, f in _fixing_order(features, sign, z)]
    return LocalFacts(
        prediction=prediction,
        contributions=sorted(
            (Contribution(f.name, f.label, f.current, float(f.graph.stds[f.bin_index])) for f in features),
            key=lambda c: abs(c.score),
            reverse=True,
        ),
        counterfactual=_counterfactual(features, intercept, sign, z),
        sufficient_reason=_sufficient_reason(features, intercept, sign, z),
        completions=_completions(features, order, intercept, sign, options),
        confidence_level=options.confidence_level,
    )


def _likelihood(probability: float) -> str:
    for threshold, word in _LIKELIHOOD_WORDS:
        if probability >= threshold:
            return word
    return "almost impossible"


def _strength(score: float) -> str:
    for threshold, word in _STRENGTH_WORDS:
        if abs(score) >= threshold:
            return word
    return "slightly"


def _dump(data: dict[str, Any]) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()


def _round(value: float) -> float:
    return round(value, 3)


def facts_to_texts(facts: LocalFacts, audience: Audience = Audience.LAYPERSON) -> dict[str, str]:
    """Render the facts as one JSON text per explanation, for the prompts.

    For a layperson the model numbers (scores, probabilities, intervals) are replaced by words, so the LLM has none to
    repeat. A data scientist gets the numbers.

    Parameters
    ----------
    facts: LocalFacts
        The facts of the prediction.
    audience: Audience, default=Audience.LAYPERSON
        Who will read the explanation.

    Returns
    -------
    dict[str, str]
        The keys ``prediction``, ``counterfactual``, ``sufficient_reason`` and ``completions``.

    """
    expert = audience == Audience.DATA_SCIENTIST
    p = facts.prediction
    prediction: dict[str, Any] = {"predicted_class": p.label}
    contributions: list[dict[str, Any]] = []
    for c in facts.contributions:
        entry: dict[str, Any] = {"feature": c.feature, "value": c.value}
        if expert:
            entry |= {"score": _round(c.score), "std": _round(c.std)}
        else:
            direction = "towards" if c.score * (1 if p.logit > 0 else -1) > 0 else "against"
            entry["effect"] = f"{_strength(c.score)} {direction} this prediction"
        contributions.append(entry)
    prediction["contributions_largest_first"] = contributions
    if expert:
        prediction |= {
            "probability": _round(p.probability),
            "confidence_interval": [_round(p.lower), _round(p.upper)],
            "confidence_level": facts.confidence_level,
            "logit": _round(p.logit),
        }
    else:
        prediction["likelihood"] = _likelihood(p.probability)

    cf = facts.counterfactual
    counterfactual: dict[str, Any] = {"flip_possible": cf is not None}
    if cf is not None:
        counterfactual["changes"] = [
            {"feature": c.feature, "from": c.from_value, "to": c.to_value}
            | ({"score_change": _round(c.score_change)} if expert else {})
            for c in cf.changes
        ]
        counterfactual["minimum_number_of_changes"] = len(cf.changes)
        if expert:
            counterfactual["flip_holds_at_confidence"] = facts.confidence_level
        else:
            counterfactual["flip_holds_even_if_the_model_is_somewhat_off"] = facts.confidence_level > 0
        if expert:
            counterfactual["probability_of_new_prediction"] = _round(cf.new_probability)
        else:
            counterfactual["likelihood_of_new_prediction"] = _likelihood(cf.new_probability)

    all_names = [c.feature for c in facts.contributions]
    sr = facts.sufficient_reason
    sufficient: dict[str, Any] = {"exists": sr is not None}
    if sr is not None:
        sufficient["features"] = sr.features
        sufficient["features_not_needed"] = [f for f in all_names if f not in sr.features]
        if expert:
            sufficient["guaranteed_margin_logit"] = _round(sr.guaranteed_margin)

    completions = [
        {
            "known_features": s.fixed,
            "share_with_same_prediction": _round(s.share) if expert else _likelihood(s.share),
        }
        for s in facts.completions
    ]
    return {
        "prediction": _dump(prediction),
        "counterfactual": _dump(counterfactual),
        "sufficient_reason": _dump(sufficient),
        "completions": _dump({"steps": completions}),
    }
