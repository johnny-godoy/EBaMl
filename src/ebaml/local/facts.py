"""Deterministic local explanations of one EBM prediction.

The three explanations are the explainability queries of Barceló, Monet, Pérez and Subercaseaux, "Model
Interpretability through the Lens of Computational Complexity" (arXiv:2010.12265):

- *Minimum change required* (counterfactual): the fewest inputs to change so that the prediction flips.
- *Minimum sufficient reason*: the fewest inputs that fix the prediction whatever the other inputs are.
- *Count completions*: the share of completions of a partially known input that get the same prediction.

An EBM without interactions is a sum of one score per feature. The decision of a prediction is a margin: a sum of one
term per feature, positive while the decision holds (``ebaml.local.local_kinds``). The first two explanations are solved
exactly by sorting the features by how much they move the margin, and the third by sampling the unknown features from
their training distribution with a fixed seed. A decision with several margins, and the counterfactual of a chosen
class, are covering problems that sorting does not solve. A ``CoveringSolver`` of ``ebaml.local.covering_solvers``
solves them. Nothing here calls an LLM, and nothing here renders text: ``ebaml.local.fact_text`` puts the facts in words
for the prompts in ``baml_src/local_explanations.baml``.

The proof that the sorting algorithms are exact is in ``proofs/papers/local-explanations/paper.typ``. It covers a binary
classifier, a multiclass classifier and a regressor, which use the same algorithms on the margin of their decision. For
a covering problem, the paper states what a result of a solver proves and what it does not.

The confidence of the model enters through ``confidence_level``, as a number ``z`` of standard deviations. Each claim
names the scores that move by ``z`` deviations, always in the direction that makes the claim harder to prove:

- A counterfactual moves the score of the new bin of each changed feature against the flip. The features that do not
  change keep their fitted scores.
- A sufficient reason moves every score against the decision: a fixed feature at its bin, and a free feature at its
  worst plausible bin.
- The interval of the output moves the total score. Its deviation is that of a sum of independent errors.
- A margin of a multiclass classifier is the difference of two class scores. Its deviation is the sum of the two
  deviations, because an EBM stores no covariance between two classes.

A claim holds under its own model only. ``confidence_level=0`` gives ``z = 0``, so no score moves.
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
from ebaml.local.covering_problem import CoveringProblem
from ebaml.local.covering_solvers import CoveringSolver, MipCover
from ebaml.local.decisions import Decision, Margin
from ebaml.local.local_kinds import LocalKind, Outcome, OutputKind, PredictionFrame, local_kind


@attrs.frozen
class _Feature:
    """The graph of a feature plus the bin the instance falls in."""

    name: str
    graph: EBMGraph
    weights: np.ndarray
    bin_index: int
    value: FeatureValue
    plausible: np.ndarray  # bins with enough training support to be offered as alternatives
    columns: tuple[int, int] | None = None  # the two outputs of a multiclass margin that this view compares

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """Score of every bin. A view of a multiclass margin has the difference of the scores of its two classes."""
        scores = np.asarray(self.graph.scores, dtype=float)
        if self.columns is None:
            return scores
        winner, rival = self.columns
        return scores[:, winner] - scores[:, rival]

    @property
    def stds(self) -> npt.NDArray[np.float64]:
        """Standard deviation of the score of every bin. A view of a margin has the sum of the two deviations."""
        stds = np.asarray(self.graph.stds, dtype=float)
        if self.columns is None:
            return stds
        winner, rival = self.columns
        return stds[:, winner] + stds[:, rival]

    def outputs(self, bin_index: int) -> npt.NDArray[np.float64]:
        """Return the score of one bin for every output of the model.

        Parameters
        ----------
        bin_index: int
            The bin.

        Returns
        -------
        npt.NDArray[np.float64]
            One score for a binary classifier and a regressor, one for each class otherwise.

        """
        return np.atleast_1d(np.asarray(self.graph.scores, dtype=float)[bin_index])

    def output_stds(self, bin_index: int) -> npt.NDArray[np.float64]:
        """Return the standard deviation of the score of one bin for every output of the model.

        Parameters
        ----------
        bin_index: int
            The bin.

        Returns
        -------
        npt.NDArray[np.float64]
            One deviation for each output.

        """
        return np.atleast_1d(np.asarray(self.graph.stds, dtype=float)[bin_index])

    def against(self, columns: tuple[int, int] | None) -> "_Feature":
        """Return the feature as seen by one margin.

        Parameters
        ----------
        columns: tuple[int, int] | None
            The two outputs that the margin compares, or ``None`` for a model with one output.

        Returns
        -------
        _Feature
            The same feature, whose scores are those of the margin.

        """
        return attrs.evolve(self, columns=columns)

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
    search: CoveringSolver
        How to solve the covering problems: the sufficient reason of a decision with several margins, as "inside an
        interval" has, and the counterfactual of ``target_class``. ``MipCover()`` (the default) searches all sets in a
        MIP solver of OR-Tools (HiGHS), and proves the smallest set unless its time limit ends first. ``GreedyCover``
        and ``LocalSearchCover`` need no MIP solver and prove less. A decision with one margin ignores this option,
        because sorting finds its smallest set exactly.
    target_class: ClassLabel | None
        For a multiclass classifier, the class that the counterfactual must reach. ``None`` (the default) asks for the
        fewest changes after which the class is no longer the predicted class. A class asks for the fewest changes
        after which every margin of that class holds at once, over every choice of bins. ``search`` finds the changes.
        Another kind of model accepts ``None`` only.

    """

    confidence_level: float = 0.95
    max_steps: int = 5
    n_samples: int = 10_000
    seed: int = 0
    min_support: float = 0.01
    decision: Decision | None = None
    search: CoveringSolver = attrs.field(factory=MipCover)
    target_class: ClassLabel | None = None


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
        The score at which the decision stops to hold: ``0`` for the log-odds of a binary classifier, and the highest
        score of the other classes for a multiclass classifier. The score is above it when the decision says that the
        score is high.
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
    """A set of changes after which the decision does not hold, at the requested confidence.

    The set is the smallest one only if ``minimum`` is true. The default counterfactual always has that proof. The
    counterfactual of ``LocalOptions.target_class`` has it only if the search proves it: ``GreedyCover``,
    ``LocalSearchCover`` and a ``MipCover`` that ends at its time limit can return a larger set.

    Attributes
    ----------
    changes: list[Change]
        The changes.
    new_decision: ClassLabel
        The decision after the changes: the other class of a binary classifier, the class that a multiclass classifier
        then predicts, or the opposite answer of a regressor.
    new_output: float
        The output of the model for the new decision: the probability of the new class, or the predicted value.
    minimum: bool
        True if no smaller set of changes gives the new decision. Sorting proves it for the default counterfactual. For
        a chosen target class, the search proves it, or the size equals ``lower_bound``.
    lower_bound: int
        No set of fewer changes gives the new decision. It equals the number of changes when ``minimum`` is true by a
        proof of sorting or of the solver.

    """

    changes: list[Change]
    new_decision: ClassLabel
    new_output: float
    minimum: bool
    lower_bound: int


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
        True if no smaller set fixes the decision: sorting proves it for a decision with one margin, and the search
        proves it, or the size equals ``lower_bound``, for several margins.
    lower_bound: int
        No set of fewer inputs fixes the decision. It is the LP bound of the search for several margins, and the size of
        the set for one margin.

    """

    features: list[str]
    guaranteed_margin: float
    minimum: bool
    lower_bound: int


@attrs.frozen
class CompletionStep:
    """The share of completions with the same prediction when only ``fixed`` features are known."""

    fixed: list[str]
    share: float


@attrs.frozen
class LocalFacts:
    """Everything computed about one prediction, with the options that say under which conditions the facts hold.

    Attributes
    ----------
    prediction: Prediction
        The prediction to explain.
    contributions: list[Contribution]
        The effect of each input, largest first.
    counterfactual: Counterfactual | None
        The changes that flip the decision, or ``None`` if the search found none.
    sufficient_reason: SufficientReason | None
        The inputs that fix the decision, or ``None`` if the search found none.
    completions: list[CompletionStep]
        The estimated share of completions with the same prediction, for each step.
    confidence_level: float
        ``LocalOptions.confidence_level`` of the facts. At ``0`` no score moves, so no claim covers the uncertainty of
        the model.
    min_support: float
        ``LocalOptions.min_support`` of the facts. At ``0`` the counterfactual and the sufficient reason leave out no
        value.
    n_samples: int
        ``LocalOptions.n_samples`` of the facts: the size of the sample behind each share of ``completions``.

    """

    prediction: Prediction
    contributions: list[Contribution]
    counterfactual: Counterfactual | None
    sufficient_reason: SufficientReason | None
    completions: list[CompletionStep]
    confidence_level: float
    min_support: float
    n_samples: int


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
        If the model is not a classifier or a regressor, or its kind is unknown.

    """
    message = "Local explanations need a classifier or a regressor."
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
    # nosemgrep: ebaml-no-range-len -- the check compares with every position of the feature names
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


def _counterfactual(
    features: Sequence[_Feature],
    intercept: float,
    sign: int,
    z: float,
    *,
    after: Callable[[float, Sequence[tuple[_Feature, int]]], Outcome],
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
    after : Callable[[float, Sequence[tuple[_Feature, int]]], Outcome]
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
    outcome = after(new_margin, [(f, best) for _, best, f in changes])
    return Counterfactual(
        changes=[
            Change(f.name, f.label, _bin_label(f.graph.x_vals[best]), float(f.scores[best]) - f.current)
            for _, best, f in changes
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimum=True,  # sorting finds the smallest set of a decision with one margin, as the paper proves
        lower_bound=len(changes),
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
        minimum=True,  # sorting finds the smallest set of a decision with one margin, as the paper proves
        lower_bound=count,
    )


@attrs.frozen
class _Bound:
    """One margin of the decision: ``sign * (intercept + the scores)`` is positive while the margin holds.

    ``columns`` names the two outputs that a margin of a multiclass classifier compares, and ``None`` otherwise.
    """

    intercept: float
    sign: int
    columns: tuple[int, int] | None = None


def _completions(
    features: Sequence[_Feature],
    order: Sequence[_Feature],
    bounds: Sequence[_Bound],
    options: LocalOptions,
) -> list[CompletionStep]:
    """Count completions, estimated by sampling the unknown features from their training bin weights.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    order : Sequence[_Feature]
        The features in the order they are revealed.
    bounds : Sequence[_Bound]
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
    steps = []
    for k in range(min(options.max_steps, len(order)) + 1):
        known = {f.name for f in order[:k]}
        holds = np.ones(options.n_samples, dtype=bool)
        for bound in bounds:
            views = [f.against(bound.columns) for f in features]
            total = sum(v.current if v.name in known else v.scores[sampled_bins[v.name]] for v in views)
            holds &= (bound.intercept + total > 0) == (bound.sign == 1)
        steps.append(CompletionStep([f.name for f in order[:k]], float(np.mean(holds))))
    return steps


@attrs.frozen
class _GainTable:
    """The numbers of the covering problem of a decision with several margins."""

    worst: list[float]
    gains: list[list[float]]


def _gain_table(features: Sequence[_Feature], bounds: Sequence[_Bound], z: float) -> _GainTable:
    """Read, for each margin, its worst case and how much each feature raises it when fixed.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    bounds : Sequence[_Bound]
        The margins of the decision.
    z : float
        Standard deviations to move the scores by.

    Returns
    -------
    _GainTable
        The worst cases and the gains.

    """
    worst: list[float] = []
    gains: list[list[float]] = []
    for bound in bounds:
        guaranteed = [_robust_scores(f.against(bound.columns), bound.sign, z, towards_outcome=True) for f in features]
        worst.append(bound.sign * bound.intercept + sum(float(scores.min()) for scores in guaranteed))
        gains.append([
            float(scores[f.bin_index] - scores.min()) for f, scores in zip(features, guaranteed, strict=True)
        ])
    return _GainTable(worst, gains)


def _several_margins_reason(
    features: Sequence[_Feature],
    table: _GainTable,
    search: CoveringSolver,
    seed: int,
) -> SufficientReason | None:
    """Choose a sufficient reason for a decision with several margins, with the search of the user.

    Each feature is a group with one alternative: the feature keeps its value.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    table : _GainTable
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
        minimum=cover.minimum,
        lower_bound=cover.lower_bound,
    )


def _reason(
    features: Sequence[_Feature],
    bounds: Sequence[_Bound],
    z: float,
    options: LocalOptions,
) -> SufficientReason | None:
    """Choose the sufficient reason of a decision.

    One margin has a smallest set that sorting finds exactly. Several margins use the search of the user.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    bounds : Sequence[_Bound]
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
    return _several_margins_reason(features, _gain_table(features, bounds, z), options.search, options.seed)


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


def _reveal_order(features: Sequence[_Feature], bounds: Sequence[_Bound], z: float) -> list[_Feature]:
    """Order the features by how much knowing their value guarantees, largest first.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features to order.
    bounds : Sequence[_Bound]
        The margins of the decision. A feature ranks by its smallest gain over these margins.
    z : float
        The number of standard deviations that each score moves against the claim.

    Returns
    -------
    list[_Feature]
        The features in the order that the count-completions steps reveal them.

    """
    if len(bounds) == 1:
        views = [f.against(bounds[0].columns) for f in features]
        by_name = {f.name: f for f in features}
        return [by_name[step.feature.name] for step in _fixing_order(views, bounds[0].sign, z)]
    return [features[index] for index in _by_smallest_gain(_gain_table(features, bounds, z).gains)]


def _smallest_counterfactual(
    features: Sequence[_Feature],
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
    features : Sequence[_Feature]
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
            replaced: Sequence[tuple[_Feature, int]],
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


def _alternatives(views: Sequence[_Feature], z: float) -> _Alternatives:
    """List the plausible bins of a feature, other than the bin of the instance, with their gains for each margin.

    The scores of the new bin move by ``z`` deviations against the target class, as in ``_counterfactual``. A gain can
    be negative: a bin that lowers one margin can still be part of the smallest set.

    Parameters
    ----------
    views : Sequence[_Feature]
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
class _Totals:
    """The score and the intercept of each output of the model for the instance, and the confidence factor ``z``."""

    scores: list[float]
    intercepts: list[float]
    z: float


def _target_counterfactual(
    features: Sequence[_Feature],
    frame: PredictionFrame,
    margins: Sequence[Margin],
    totals: _Totals,
    *,
    options: LocalOptions,
) -> Counterfactual | None:
    """Find the fewest changes after which every margin of the target class holds.

    Each feature is a group, and each of its plausible bins other than the bin of the instance is an alternative. A set
    of changes moves each changed feature to one bin. ``options.search`` solves this covering problem over every choice
    of bins.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    frame : PredictionFrame
        The prediction.
    margins : Sequence[Margin]
        The margins of the target class against every other class.
    totals : _Totals
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
                _bin_label(feature.graph.x_vals[bin_index]),
                float(
                    (feature.outputs(bin_index) - feature.outputs(feature.bin_index))[target]
                    - (feature.outputs(bin_index) - feature.outputs(feature.bin_index))[frame.column],
                ),
            )
            for feature, bin_index in changed
        ],
        new_decision=outcome.decision,
        new_output=outcome.output,
        minimum=cover.minimum,
        lower_bound=cover.lower_bound,
    )


def _choose_counterfactual(
    features: Sequence[_Feature],
    frame: PredictionFrame,
    totals: _Totals,
    *,
    kind: LocalKind,
    options: LocalOptions,
) -> Counterfactual | None:
    """Find the counterfactual that the options ask for: the target class if there is one, else the smallest.

    Parameters
    ----------
    features : Sequence[_Feature]
        The features.
    frame : PredictionFrame
        The prediction.
    totals : _Totals
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


def compute_local_facts(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, FeatureValue],
    options: LocalOptions | None = None,
) -> LocalFacts:
    """Compute the three local explanations of one prediction of a classifier or a regressor.

    The classifier is binary or multiclass. The function raises an ``UnsupportedModelError`` for an EBM that is not a
    classifier or a regressor, and for an EBM with interactions. It raises an ``InvalidInstanceError`` for an instance
    that lacks a value or has an unknown category. A regressor explains the answer to ``LocalOptions.decision``.

    A counterfactual and a sufficient reason are the smallest sets only if their ``minimum`` is true. Sorting proves
    it for a decision with one margin. A decision with several margins and the counterfactual of a target class
    depend on ``LocalOptions.search``.

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
    intercepts = np.ravel(ebm.intercept_).tolist()
    scores = (np.array(intercepts) + sum(f.outputs(f.bin_index) for f in features)).tolist()
    score_stds = np.sqrt(sum(f.output_stds(f.bin_index) ** 2 for f in features)).tolist()
    frame = kind.frame(
        scores,
        score_stds,
        intercepts=intercepts,
        confidence_level=options.confidence_level,
        decision=options.decision,
    )
    margins = frame.side.margins
    binding = frame.side.binding(scores)
    # every algorithm sees the margin sign * (intercept - threshold + the scores), which is positive while it holds
    bounds = [_Bound(margin.offset(intercepts), margin.sign, margin.columns) for margin in margins]
    prediction = Prediction(
        decision=frame.side.label,
        output=frame.output,
        lower=frame.lower,
        upper=frame.upper,
        score=scores[frame.column],
        score_std=score_stds[frame.column],
        margin=binding.value(scores),
        threshold=binding.limit(scores),
        output_kind=kind.output_kind,
    )
    order = _reveal_order(features, bounds, z)
    counterfactual = _choose_counterfactual(
        features,
        frame,
        _Totals(scores, intercepts, z),
        kind=kind,
        options=options,
    )
    return LocalFacts(
        prediction=prediction,
        contributions=sorted(
            (
                Contribution(
                    f.name,
                    f.label,
                    float(f.outputs(f.bin_index)[frame.column]),
                    float(f.output_stds(f.bin_index)[frame.column]),
                )
                for f in features
            ),
            key=lambda c: abs(c.score),
            reverse=True,
        ),
        counterfactual=counterfactual,
        sufficient_reason=_reason(features, bounds, z, options),
        completions=_completions(features, order, bounds, options),
        confidence_level=options.confidence_level,
        min_support=options.min_support,
        n_samples=options.n_samples,
    )
