"""Deterministic local explanations of one EBM prediction.

The three explanations are the explainability queries of Barceló, Monet, Pérez and Subercaseaux, "Model
Interpretability through the Lens of Computational Complexity" (arXiv:2010.12265):

- *Minimum change required* (counterfactual): the fewest inputs to change so that the prediction flips.
- *Minimum sufficient reason*: the fewest inputs that fix the prediction whatever the other inputs are.
- *Count completions*: the share of completions of a partially known input that get the same prediction.

An EBM without interactions is a sum of one score per feature. The decision of a prediction is a margin: a sum of one
term per feature, positive while the decision holds (``ebaml.local.local_kinds``). The first two explanations are solved
exactly by sorting the features by how much they move the margin, and the third by sampling the unknown features from
their training distribution with a fixed seed. Nothing here calls an LLM, and nothing here renders text:
``ebaml.local.fact_text`` puts the facts in words for the prompts in ``baml_src/local_explanations.baml``.
The proof that the first two are exact for a binary classifier is in ``proofs/papers/local-explanations/paper.typ``.
A regressor uses the same algorithms on the margin of its decision. The proof of that reduction is not written yet.

The confidence of the model enters through ``confidence_level``: every score is moved by ``z`` standard deviations in
the direction that makes the claim harder to prove, so a counterfactual flips the prediction, and a sufficient reason
keeps it, even if the scores sit at the edge of their confidence interval.
"""

import math
import operator
import statistics
import typing
from collections.abc import Callable, Mapping, Sequence

import attrs
import numpy as np
import numpy.typing as npt
from interpret.utils._clean_x import _densify_categorical  # ruff: ignore[import-private-name] -- interpret has no public function that writes a value as the text of its category

from ebaml.core.custom_types import PANDAS_NA_TYPE, ClassLabel, ExplainableBoostingModel, FeatureValue
from ebaml.core.exceptions import InvalidInstanceError, OptionRangeError, UnsupportedModelError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.model_kinds import model_kind
from ebaml.graphs.terms import main_effect_features, main_term
from ebaml.local.decisions import Decision
from ebaml.local.local_kinds import LocalKind, OutputKind, local_kind


@attrs.frozen
class _Feature:
    """The graph of a feature plus the bin the instance falls in."""

    name: str
    graph: EBMGraph
    weights: np.ndarray
    bin_index: int
    value: FeatureValue
    plausible: np.ndarray  # bins with enough training support to be offered as alternatives

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """Score of every bin. A binary classifier and a regressor have one score per bin."""
        return np.asarray(self.graph.scores, dtype=float)

    @property
    def stds(self) -> npt.NDArray[np.float64]:
        """Standard deviation of the score of every bin."""
        return np.asarray(self.graph.stds, dtype=float)

    @property
    def current(self) -> float:
        """Score of the bin of the instance."""
        return float(self.scores[self.bin_index])

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
    decision: Decision | None
        The question about the predicted value of a regressor that the explanations keep, such as ``Threshold(10)`` or
        ``Interval(10, 20)``. ``None`` asks whether the value is above the intercept, the average prediction. A
        classifier has its predicted class as the decision, so it accepts ``None`` only.

    """

    confidence_level: float = 0.95
    max_steps: int = 5
    n_samples: int = 10_000
    seed: int = 0
    min_support: float = 0.01
    decision: Decision | None = None


@attrs.frozen
class Prediction:
    """The prediction to explain, with the confidence interval of its output.

    The fields fit every model kind. ``docs/local-explanations-proof.md`` defines them for each kind.

    Attributes
    ----------
    decision: ClassLabel
        What the explanations keep or change: the predicted class of a classifier, or the answer of a regressor to its
        ``Decision``, such as ``"above 10"``.
    output: float
        The output of the model for the decision: the probability of the predicted class, or the predicted value.
    lower: float
        The lower end of the confidence interval of ``output``.
    upper: float
        The upper end of the confidence interval of ``output``.
    score: float
        The sum of the intercept and the scores, on the scale of the link: the log-odds of the second class, or the
        predicted value.
    score_std: float
        The standard deviation of ``score``.
    margin: float
        The margin of the decision, positive while the decision holds: the log-odds of the predicted class, or the
        distance of the predicted value from the threshold, on the side of the decision. The margin means the same for
        every kind, and ``score`` keeps the scale of the link.
    threshold: float
        The score at which the decision stops to hold: ``0`` for the log-odds of a classifier. The score is above it
        when the decision says that the score is high.
    output_kind: OutputKind
        What ``output`` is.

    """

    decision: ClassLabel
    output: float
    lower: float
    upper: float
    score: float
    score_std: float
    margin: float
    threshold: float = 0.0
    output_kind: OutputKind = OutputKind.PROBABILITY


@attrs.frozen
class Contribution:
    """The effect of one input on the prediction."""

    feature: str
    value: str
    score: float
    std: float


@attrs.frozen
class Change:
    """One input to change in a counterfactual."""

    feature: str
    from_value: str
    to_value: str
    score_change: float


@attrs.frozen
class Counterfactual:
    """The smallest set of changes after which the decision does not hold, at the requested confidence.

    Attributes
    ----------
    changes: list[Change]
        The changes.
    new_decision: ClassLabel
        The decision after the changes: the other class of a binary classifier, or the opposite answer of a regressor.
    new_output: float
        The output of the model for the new decision: the probability of the other class, or the predicted value.

    """

    changes: list[Change]
    new_decision: ClassLabel
    new_output: float


@attrs.frozen
class SufficientReason:
    """A set of inputs that fixes the decision whatever the other inputs are.

    Attributes
    ----------
    features: list[str]
        The inputs.
    guaranteed_margin: float
        The worst case of the margin of the decision when these inputs keep their values.
    minimum: bool
        True if no smaller set fixes the decision. False if the set is only minimal: no input of the set can be
        removed. The set of a binary classifier is always the minimum.

    """

    features: list[str]
    guaranteed_margin: float
    minimum: bool


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


def _bin_label(x_val: tuple[float, float] | str) -> str:
    """Render a bin: ``[lo, hi)`` for continuous features, the category otherwise.

    Parameters
    ----------
    x_val: tuple[float, float] | str
        An interval ``(lo, hi)`` or a category.

    Returns
    -------
    str
        The text of the bin.

    """
    if isinstance(x_val, tuple):
        return f"[{x_val[0]:.4g}, {x_val[1]:.4g})"
    return x_val


def _bin_index(ebm: ExplainableBoostingModel, feature_index: int, graph: EBMGraph, value: FeatureValue) -> int:
    """Find the bin of ``value`` in the graph of a feature.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature.
    graph: EBMGraph
        The graph of the feature.
    value: FeatureValue
        The value of the instance.

    Returns
    -------
    int
        The index of the bin in the graph.

    Raises
    ------
    InvalidInstanceError
        If the value is missing, the value of a continuous feature is not a number, or the category was not seen in
        training.

    """
    # the missing markers of a FeatureValue: None, pd.NA, and a NaN of Python or of any NumPy float type
    if (
        value is None
        or isinstance(value, PANDAS_NA_TYPE)
        or (isinstance(value, float | np.floating) and math.isnan(value))
    ):
        msg = f"Feature {graph.feature_name!r} is missing; local explanations need every input."
        raise InvalidInstanceError(msg)
    if graph.feature_type == "continuous":
        # interpret types bins_ loosely, but the bins of a continuous feature are an array of cut points
        cuts = typing.cast(npt.NDArray[np.float64], ebm.bins_[feature_index][0])
        # a value of a continuous feature is a number or a numeric text, and float() raises for anything else
        try:
            number = float(typing.cast(typing.SupportsFloat | str, value))
        except (TypeError, ValueError) as error:
            msg = f"Feature {graph.feature_name!r} is continuous, but its value {value!r} is not a number."
            raise InvalidInstanceError(msg) from error
        return int(np.searchsorted(cuts, number, side="right"))
    category = _category(value, graph.feature_name)
    if category not in graph.x_vals:
        msg = f"Unknown category {value!r} for feature {graph.feature_name!r}."
        raise InvalidInstanceError(msg)
    return graph.x_vals.index(category)


def _category(value: FeatureValue, feature_name: str) -> str:
    """Write a value of a nominal feature as interpret writes its categories.

    interpret keeps the categories as text: ``2`` and ``2.0`` become ``"2"``, and ``True`` becomes ``"1"``. So a row of
    the training data, with its numbers and booleans, finds the bins of its categories.

    Parameters
    ----------
    value: FeatureValue
        The value of the instance.
    feature_name: str
        The name of the feature, for the error message.

    Returns
    -------
    str
        The category, as the bins of the EBM name it.

    Raises
    ------
    InvalidInstanceError
        If interpret cannot write the value as a category, as for a complex number.

    """
    try:
        # interpret has no type hints, but the function returns one text per value
        categories = typing.cast(npt.NDArray[np.str_], _densify_categorical(np.array([value], dtype=object)))
    except TypeError as error:
        msg = f"The value {value!r} of feature {feature_name!r} cannot be a category."
        raise InvalidInstanceError(msg) from error
    return str(categories[0])


def _plausible_bins(
    weights: npt.NDArray[np.float64],
    *,  # pragma: no mutate -- a positional-only weights still lets the callers pass the other options by keyword
    continuous: bool,
    min_support: float,
) -> npt.NDArray[np.bool_]:
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
    return np.logical_and(upper_edge > min_support, lower_edge < 1 - min_support)


def _local_kind(ebm: ExplainableBoostingModel) -> LocalKind:
    """Read the kind of an EBM for the local explanations.

    The model kind is the one place that reads ``link_`` and ``classes_``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.

    Returns
    -------
    LocalKind
        The class of the kind.

    Raises
    ------
    UnsupportedModelError
        If the model is not a binary classifier or a regressor, or its kind is unknown.

    """
    message = "Local explanations need a binary classifier or a regressor."
    try:
        return local_kind(model_kind(ebm))
    except UnsupportedModelError as error:
        if str(error).startswith(message):
            raise
        msg = f"{message} {error}"
        raise UnsupportedModelError(msg) from error


def _features(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, FeatureValue],
    min_support: float,
) -> list[_Feature]:
    """Read the graph, the training weights and the bin of the instance of every feature.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    instance: Mapping[str, FeatureValue]
        The value of every feature.
    min_support: float
        See ``LocalOptions.min_support``.

    Returns
    -------
    list[_Feature]
        One entry per feature.

    Raises
    ------
    UnsupportedModelError
        If the model has interactions, or a feature has no main-effect term.
    InvalidInstanceError
        If the instance lacks a feature, a value is missing, or a category was not seen in training.

    """
    if any(len(term) > 1 for term in ebm.term_features_):
        msg = "Local explanations need an EBM without interactions (fit with interactions=0)."
        raise UnsupportedModelError(msg)
    # a feature that exclude or remove_terms left out has no score, so the prediction cannot be split by feature
    with_term = main_effect_features(ebm)
    if with_term != list(range(len(ebm.feature_names_in_))):
        missing = [name for index, name in enumerate(ebm.feature_names_in_) if index not in with_term]
        msg = f"Local explanations need a main-effect term for every feature. These features have none: {missing}."
        raise UnsupportedModelError(msg)
    features = []
    for index, name in enumerate(ebm.feature_names_in_):
        if name not in instance:
            msg = f"The instance has no value for feature {name!r}."
            raise InvalidInstanceError(msg)
        graph = EBMGraph.from_ebm(ebm, index)
        weights = np.asarray(ebm.bin_weights_[main_term(ebm, index)][1:-1], dtype=float)
        bin_index = _bin_index(ebm, index, graph, instance[name])
        plausible = _plausible_bins(weights, continuous=graph.feature_type == "continuous", min_support=min_support)
        plausible[bin_index] = True
        features.append(_Feature(name, graph, weights, bin_index, instance[name], plausible))
    return features


def _robust_scores(
    feature: _Feature,
    sign: int,
    z: float,
    *,  # pragma: no mutate -- a positional-only z still lets the callers pass towards_outcome by keyword
    towards_outcome: bool,
) -> np.ndarray:
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
    scores = sign * feature.scores
    stds = z * feature.stds
    return np.where(feature.plausible, scores - stds if towards_outcome else scores + stds, np.inf)


def _counterfactual(  # ruff: ignore[too-many-arguments] -- the margin, the sort direction, the confidence and the decision after the flip are separate inputs
    features: Sequence[_Feature],
    intercept: float,
    sign: int,
    z: float,
    *,
    new_decision: ClassLabel,
    new_output: Callable[[float], float],
) -> Counterfactual | None:
    """Minimum change required: the features with the largest possible drop in favour of the outcome, until it flips.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    intercept : float
        The intercept of the model minus the threshold of the decision.
    sign : int
        ``1`` if the decision says that the score is high, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.
    new_decision : ClassLabel
        The decision after the changes.
    new_output : Callable[[float], float]
        The output of the model for the new decision, from the margin after the changes.

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
    ranked = sorted(drops, key=operator.itemgetter(0), reverse=True)
    # the margin left after the first 1, 2, ... changes; the prediction flips at the first one below zero
    flips = np.flatnonzero(margin - np.cumsum([drop for drop, _, _ in ranked]) < 0)
    if not flips.size:
        return None
    changes = ranked[: flips[0] + 1]
    new_margin = margin + sign * sum(float(f.scores[best]) - f.current for _, best, f in changes)
    return Counterfactual(
        changes=[
            Change(f.name, f.label, _bin_label(f.graph.x_vals[best]), float(f.scores[best]) - f.current)
            for _, best, f in changes
        ],
        new_decision=new_decision,
        new_output=new_output(new_margin),
    )


@attrs.frozen
class _Gain:
    """What knowing the value of one feature guarantees."""

    gain: float
    feature: _Feature


def _fixing_order(features: Sequence[_Feature], sign: int, z: float) -> list[_Gain]:
    """Features by how much knowing their value guarantees, largest first.

    Parameters
    ----------
    features : Sequence[_Feature]
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
        guaranteed = _robust_scores(feature, sign, z, towards_outcome=True)
        gains.append(_Gain(float(guaranteed[feature.bin_index] - guaranteed.min()), feature))
    return sorted(gains, key=operator.attrgetter("gain"), reverse=True)


def _sufficient_reason(features: Sequence[_Feature], intercept: float, sign: int, z: float) -> SufficientReason | None:
    """Minimum sufficient reason: fix the features that guarantee the most until no completion can flip the outcome.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    intercept : float
        The intercept of the model minus the threshold of the decision.
    sign : int
        ``1`` if the decision says that the score is high, ``-1`` otherwise.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    SufficientReason | None
        The features to fix, or None if even all of them do not guarantee the outcome.

    """
    worst = sign * intercept + sum(float(_robust_scores(f, sign, z, towards_outcome=True).min()) for f in features)
    order = _fixing_order(features, sign, z)
    # the guaranteed margin after fixing the first 0, 1, 2, ... features; the outcome is certain at the first above zero
    margins = worst + np.concatenate(([0.0], np.cumsum([step.gain for step in order])))
    enough = np.flatnonzero(margins > 0)
    if not enough.size:
        return None
    count = int(enough[0])
    return SufficientReason(
        features=[step.feature.name for step in order[:count]],
        guaranteed_margin=float(margins[count]),
        minimum=True,  # the greedy set of a binary classifier is the smallest, as the paper proves
    )


def _completions(
    features: Sequence[_Feature],
    order: Sequence[_Feature],
    intercept: float,
    sign: int,
    options: LocalOptions,
) -> list[CompletionStep]:
    """Count completions, estimated by sampling the unknown features from their training bin weights.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    order : Sequence[_Feature]
        The features in the order they are revealed.
    intercept : float
        The intercept of the model minus the threshold of the decision.
    sign : int
        ``1`` if the decision says that the score is high, ``-1`` otherwise.
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
        sampled[f.name] = f.scores[bins]
    current = {f.name: f.current for f in features}
    steps = []
    for k in range(min(options.max_steps, len(order)) + 1):
        known = {f.name for f in order[:k]}
        logit = intercept + sum(current[n] if n in known else sampled[n] for n in current)
        steps.append(CompletionStep([f.name for f in order[:k]], float(np.mean((logit > 0) == (sign == 1)))))
    return steps


def compute_local_facts(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, FeatureValue],
    options: LocalOptions | None = None,
) -> LocalFacts:
    """Compute the three local explanations of one prediction of a binary classifier or a regressor.

    The function raises an ``UnsupportedModelError`` for an EBM that is not a binary classifier or a regressor, and
    for an EBM with interactions. It raises an ``InvalidInstanceError`` for an instance that lacks a value or has an
    unknown category. A regressor explains the answer to ``LocalOptions.decision``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    instance: Mapping[str, FeatureValue]
        The value of every feature, by feature name.
    options: LocalOptions, optional
        Confidence level, sampling and support options.

    Returns
    -------
    LocalFacts
        The prediction, the contributions and the three explanations.

    Raises
    ------
    OptionRangeError
        If the confidence level is not in [0, 1).

    """
    options = options or LocalOptions()
    if not 0 <= options.confidence_level < 1:
        msg = f"confidence_level must be in [0, 1). Got {options.confidence_level}."
        raise OptionRangeError(msg)
    z = statistics.NormalDist().inv_cdf((1 + options.confidence_level) / 2)
    kind = _local_kind(ebm)
    features = _features(ebm, instance, options.min_support)
    intercept = float(np.ravel(ebm.intercept_)[0])
    score = intercept + sum(f.current for f in features)
    score_std = math.sqrt(sum(float(f.stds[f.bin_index]) ** 2 for f in features))
    frame = kind.frame(
        score,
        score_std,
        intercept=intercept,
        confidence_level=options.confidence_level,
        decision=options.decision,
    )
    sign = frame.side.sign
    # every algorithm sees the margin sign * (baseline + sum of the scores), which is positive while the decision holds
    baseline = intercept - frame.side.threshold
    prediction = Prediction(
        decision=frame.side.label,
        output=frame.output,
        lower=frame.lower,
        upper=frame.upper,
        score=score,
        score_std=score_std,
        margin=sign * (score - frame.side.threshold),
        threshold=frame.side.threshold,
        output_kind=kind.output_kind,
    )
    order = [step.feature for step in _fixing_order(features, sign, z)]
    return LocalFacts(
        prediction=prediction,
        contributions=sorted(
            (Contribution(f.name, f.label, f.current, float(f.stds[f.bin_index])) for f in features),
            key=lambda c: abs(c.score),
            reverse=True,
        ),
        counterfactual=_counterfactual(
            features,
            baseline,
            sign,
            z,
            new_decision=frame.side.opposite,
            new_output=frame.output_after,
        ),
        sufficient_reason=_sufficient_reason(features, baseline, sign, z),
        completions=_completions(features, order, baseline, sign, options),
        confidence_level=options.confidence_level,
    )
