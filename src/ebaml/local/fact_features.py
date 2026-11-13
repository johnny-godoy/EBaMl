"""Read the features of one instance from an EBM: the bin of its value, the plausible bins and the robust scores."""

import math
import typing
from collections.abc import Mapping

import attrs
import numpy as np
import numpy.typing as npt
from interpret.utils._clean_x import _densify_categorical  # ruff: ignore[import-private-name] -- interpret has no public function that writes a value as the text of its category

from ebaml.core.custom_types import PANDAS_NA_TYPE, ExplainableBoostingModel, FeatureValue
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.terms import main_effect_features, main_term


@attrs.frozen
class Feature:
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

    def against(self, columns: tuple[int, int] | None) -> "Feature":
        """Return the feature as seen by one margin.

        Parameters
        ----------
        columns: tuple[int, int] | None
            The two outputs that the margin compares, or ``None`` for a model with one output.

        Returns
        -------
        Feature
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


def bin_label(x_val: tuple[float, float] | str) -> str:
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


def read_features(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, FeatureValue],
    min_support: float,
) -> list[Feature]:
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
    list[Feature]
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
        features.append(Feature(name, graph, weights, bin_index, instance[name], plausible))
    return features


def robust_scores(
    feature: Feature,
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
    feature : Feature
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
