"""Graphs for EBM models."""

import warnings
from collections.abc import Sequence
from typing import Literal, Protocol, Self, runtime_checkable

import attrs
import numpy as np
import numpy.typing as npt
import scipy
from interpret.glassbox._ebm._utils import convert_to_intervals  # ruff: ignore[import-private-name] -- interpret has no public convert_to_intervals

from ebaml.core.custom_types import (
    ExplainableBoostingModel,
    FeatureType,
    Normalization,
    coerce_enum,
    normalize_feature_type,
)
from ebaml.core.exceptions import (
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    InvalidOptionError,
)
from ebaml.graphs.graph_text import parse_graph_text, xy_to_json

# nominal features are unordered categories
_FORMAT_ALIASES = {FeatureType.NOMINAL: FeatureType.CATEGORICAL}

# the simplification of a graph that is too large grows by 0.1% per attempt, up to 10%
_SIMPLIFICATION_STEP = 0.001
_MAX_SIMPLIFICATION_STEPS = 100

# the preambles that describe a graph to the LLM, by feature format
_DESCRIPTIONS = {
    FeatureType.CONTINUOUS: (
        "This graph represents a continuous-valued feature. The keys are intervals"
        " that represent ranges where the function predicts the same value.\n\n"
    ),
    FeatureType.CATEGORICAL: (
        "This graph represents categorical feature. Each key represents a possible value that the feature can take.\n\n"
    ),
    FeatureType.BOOLEAN: (
        "This graph represents a boolean feature. The keys are 'True' and 'False',"
        " the two possible values of the feature.\n\n"
    ),
}


@runtime_checkable
class TokenBudget(Protocol):
    """The result of a token budget check."""

    @property
    def estimated_tokens(self) -> int:
        """The estimated number of tokens of the text."""
        ...

    @property
    def status(self) -> Literal["ok", "warn", "over"]:
        """``"ok"`` within ``max_tokens``, ``"warn"`` above it, and ``"over"`` beyond ``overbudget_percent``."""
        ...


@runtime_checkable
class BudgetCheck(Protocol):
    """A function that estimates the tokens of a text and compares them with a budget.

    ``baml_sdk.check_token_budget`` is such a function.
    """

    def __call__(self, *, text: str, max_tokens: int, overbudget_percent: int) -> TokenBudget:
        """Compare the estimated tokens of ``text`` with ``max_tokens``, and with the tolerance above it."""
        ...


# the formats whose graphs are simplified until they fit into the token budget
_SIMPLIFIABLE = {FeatureType.CONTINUOUS}

# we assume that the left key (the lower number, typically -1 or 0), is 'False'
_BOOLEAN_KEYS = ["False", "True"]


def _is_boolean(x_vals: Sequence) -> bool:
    """Whether the x-axis values are exactly the keys ``False`` and ``True`` (case-insensitive).

    Parameters
    ----------
    x_vals: Sequence
        The values of the x-axis.

    Returns
    -------
    bool
        True if the x-axis values are the keys ``False`` and ``True``.

    """
    return tuple(str(x).upper() for x in x_vals) == ("FALSE", "TRUE")


def _describe_format(feature_format: FeatureType) -> str:
    """Describe to the LLM how to read a graph of the given format.

    Parameters
    ----------
    feature_format: FeatureType
        The format of the feature.

    Returns
    -------
    str
        The preamble of the textual representation of the graph.

    Raises
    ------
    InvalidOptionError.from_enum
        If the feature format has no graph description.

    """
    try:
        return _DESCRIPTIONS[feature_format]
    except KeyError:
        raise InvalidOptionError.from_enum(feature_format, FeatureType) from None


def _auto_y_precision(scores: Sequence[float] | npt.NDArray[np.floating]) -> int:
    """Compute the y-axis precision that keeps at least 3 significant digits of the variation of the scores.

    Parameters
    ----------
    scores: Sequence[float] | npt.NDArray[np.floating]
        The scores of the graph.

    Returns
    -------
    int
        The number of decimals.

    """
    total_variation = np.max(scores) - np.min(scores)
    precision = 1
    # a flat graph has no scale to adapt to (and would never leave the loop below)
    if total_variation:
        y_fraction = total_variation / 100.0
        while y_fraction < 1:
            y_fraction *= 10
            precision += 1
    return precision


@attrs.define
class EBMGraph:
    """A datastructure for the graphs of Explainable Boosting Machines."""

    feature_name: str = attrs.field()
    feature_type: str = attrs.field()
    # intervals for continuous features, categories for nominal ones
    x_vals: Sequence[tuple[float, float]] | Sequence[str] = attrs.field()
    # from_ebm stores NumPy arrays here
    scores: Sequence[float] | npt.NDArray[np.floating] = attrs.field()
    stds: Sequence[float] | npt.NDArray[np.floating] = attrs.field()

    @classmethod
    def from_ebm(
        cls,
        ebm: ExplainableBoostingModel,
        feature_index: int,
        normalization: Normalization | str = Normalization.NONE,
        *,
        use_feature_bounds: bool = True,
    ) -> Self:
        """Extract the graph of a feature from an Explainable Boosting Machine.

        This is a low-level function. It does not return the final format in which the graph is presented to the LLM.

        The purpose of this function is to extract the graph from the interals of the EBM and return it in a format
        that is easy to work with.

        Parameters
        ----------
        ebm: ExplainableBoostingModel
            The EBM model.
        feature_index: int
            The index of the feature in the EBM.
        normalization: Normalization | str, default=Normalization.NONE
            How to normalize the graph (shift on the y-axis).
        use_feature_bounds: bool, default=True
            If True, the first and last x-axis bins are the min and max values of the feature stored in the EBM.
            If false, the first and last value are -inf and inf.

        Returns
        -------
        EBMGraph
            The graph.

        Raises
        ------
        ValueError
            If the feature type is not continuous or nominal.
        InvalidOptionError
            If the normalization option is not valid.
        InvalidOptionError.from_enum
            If the normalization option is not valid.

        """
        # read the variables from the ebm
        feature_name = ebm.feature_names_in_[feature_index]
        feature_type = ebm.feature_types_in_[feature_index]
        scores = ebm.term_scores_[feature_index][1:-1].copy()  # Drop missing and unknown bins
        stds = ebm.standard_deviations_[feature_index][1:-1]

        # normalize the graph
        match normalization:
            case Normalization.Z_SCORE:
                normalization_constant = np.mean(scores)
            case Normalization.MIN_MAX:
                normalization_constant = np.min(scores)
            case Normalization.NONE:
                normalization_constant = 0
            case _:
                raise InvalidOptionError.from_enum(str(normalization), Normalization)
        scores -= normalization_constant

        # read the x-axis bins from the ebm
        match feature_type:
            case "continuous":
                x_vals = convert_to_intervals(ebm.bins_[feature_index][0])
                # feature bounds apply to continuous features only
                if use_feature_bounds:
                    x_vals[0] = (ebm.feature_bounds_[feature_index][0], x_vals[0][1])
                    x_vals[-1] = (x_vals[-1][0], ebm.feature_bounds_[feature_index][1])
            case "nominal":
                # the bins map each category to its (1-based, after the missing bin) position
                positions = {position - 1: category for category, position in ebm.bins_[feature_index][0].items()}
                x_vals = [positions[idx] for idx in range(len(positions))]
            case _:
                raise InvalidOptionError(feature_type, ["continuous", "nominal"])
        if len(x_vals) != len(scores):
            msg = (
                f"The number of bins ({len(x_vals)}) and scores ({len(scores)}) does not match"
                f" for feature {feature_name} of type {feature_type}."
            )
            raise ValueError(
                msg,
            )
        return EBMGraph(feature_name, feature_type, x_vals, scores, stds)

    def simplify(self, min_variation_per_cent: float = 0.0) -> Self:
        """Simplifies the graph. Removes redundant (flat) bins from the graph.

        Parameters
        ----------
        min_variation_per_cent: float, default=0.0
            Parameter that controlls the degree of simplification.
            If min_variation_per_cent>0, the function simplifies the graph by removing bins that correspond to a less
            that min_variation_per_cent change in the score, considering the overal min/max difference of score for the
            feature as 100%.

        Returns
        -------
        EBMGraph:
            The simplified graph.

        Raises
        ------
        InvalidOptionError
            If the feature type is not continuous.
        ValueError
            If min_variation_per_cent is not between 0 and 1.

        """
        match normalize_feature_type(self.feature_type):
            case FeatureType.CONTINUOUS:
                pass
            case _:
                raise InvalidOptionError(self.feature_type, ["continuous"])
        if not 0 <= min_variation_per_cent <= 1:
            msg = f"min_variation_per_cent must be between 0 and 1. Got {min_variation_per_cent}."
            raise ValueError(msg)
        x_vals, scores, stds = self.x_vals, self.scores, self.stds
        total_variation = np.max(scores) - np.min(scores)
        new_x_vals, new_scores, new_stds = [x_vals[0]], [scores[0]], [stds[0]]
        later_bins = range(1, len(x_vals))  # pragma: no mutate -- the first bin compared with itself changes nothing
        for idx in later_bins:
            b0, b1 = x_vals[idx]
            if np.abs(float(new_scores[-1]) - float(scores[idx])) <= total_variation * min_variation_per_cent:
                # extend the previous bin to b1
                new_x_vals[-1] = (new_x_vals[-1][0], b1)
                # guarantee that the the confidence bands of the simplified graph cover the original graph as well as
                # its confidence bands
                new_stds[-1] = max(new_stds[-1], stds[idx])
            else:
                new_x_vals.append((b0, b1))
                new_scores.append(scores[idx])
                new_stds.append(stds[idx])
        return EBMGraph(
            self.feature_name,
            self.feature_type,
            new_x_vals,
            new_scores,
            new_stds,
        )

    def _resolve_format(self, feature_format: FeatureType | str | None) -> FeatureType:
        """Determine the feature format to present, auto-detecting it when it is not given.

        Parameters
        ----------
        feature_format: FeatureType | str, optional
            The requested format.

        Returns
        -------
        FeatureType
            The format to present.

        """
        # a simple auto-detect for boolean features
        if _is_boolean(self.x_vals):
            feature_format = FeatureType.BOOLEAN
        if feature_format is None:
            feature_format = normalize_feature_type(self.feature_type)
        feature_format = coerce_enum(FeatureType, feature_format)
        return _FORMAT_ALIASES.get(feature_format, feature_format)

    def _check_boolean_bins(self, feature_format: FeatureType) -> None:
        """Check that a graph that is presented as boolean has exactly two x-axis values.

        Parameters
        ----------
        feature_format: FeatureType
            The format to present it in.

        Raises
        ------
        ValueError
            If the format is boolean and the graph does not have exactly two x-axis values.

        """
        match feature_format, self.x_vals:
            case FeatureType.BOOLEAN, [_, _]:
                pass
            case FeatureType.BOOLEAN, _:
                msg = f"Requested a boolean format, but the feature has more than x-axis values: {self.x_vals}"
                raise ValueError(msg)

    def to_body_text(
        self,
        *,  # pragma: no mutate -- a positional-only self still lets the callers pass the other options by keyword
        feature_format: FeatureType,
        x_axis_precision: int | None,
        y_axis_precision: int,
        confidence_bounds: bool,
        confidence_level: float,
    ) -> str:
        """Render the lines of text (name, type, means and optional bounds) that follow the preamble.

        Parameters
        ----------
        feature_format: FeatureType
            The format of the feature.
        x_axis_precision: int, optional
            The precision of the values on the x-axis. If None, the precision is not adjusted.
        y_axis_precision: int
            The precision of the values on the y-axis.
        confidence_bounds: bool
            Whether to include confidence bounds.
        confidence_level: float
            The desired confidence level of the bounds.

        Returns
        -------
        str
            The textual representation of the graph, without the preamble.

        """
        scores = self.scores
        x_vals = self.x_vals
        # adjust the precision on the x-axis. this can be dangerous if we don't know the distribution of the data
        # (it can impact the accuracy). however, this is a necessary formatting feature because certain (old?)
        # versions of EBMs introduce trailing digits for the xbins (e.g. age as a floating point number with 10
        # significant digits)
        if x_axis_precision is not None:
            x_vals = [
                (np.round(x[0], x_axis_precision).astype(str), np.round(x[1], x_axis_precision).astype(str))
                for x in x_vals
            ]
        # formatting for boolean features
        match feature_format:
            case FeatureType.BOOLEAN:
                x_vals = _BOOLEAN_KEYS

        prompt = f"Feature Name: {self.feature_name}\n"
        prompt += f"Feature Type: {feature_format}\n"
        prompt += f"Means: {xy_to_json(x_vals, np.round(scores, y_axis_precision).astype(str))}\n"
        if confidence_bounds:
            # confidence bounds via normal approximation
            factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
            for label, sign in (("Lower", -1), ("Upper", 1)):
                bounds = [score + sign * factor * std for score, std in zip(scores, self.stds, strict=True)]
                prompt += (
                    f"{label} Bounds ({confidence_level * 100:.0f}%-Confidence Interval):"
                    f" {xy_to_json(x_vals, np.round(bounds, y_axis_precision).astype(str))}\n"
                )
        return prompt

    def to_text(  # ruff: ignore[too-many-arguments] -- keyword-only options that descriptors.py forwards by name
        self,
        *,
        include_description: bool = True,
        feature_format: FeatureType | str | None = None,
        x_axis_precision: int | None = None,
        y_axis_precision: int | Literal["auto"] | None = None,
        confidence_bounds: bool = True,
        confidence_level: float = 0.95,
        max_tokens: int = 3000,
        overbudget_percent: int = 50,
        check_budget: BudgetCheck | None = None,
    ) -> str:
        """Convert the graph to text. This is the text that we then pass to the LLM.

        The function takes care of a variety of different formatting issues that can arise in the process of
        converting a graph to text.

        Parameters
        ----------
        include_description: bool, default=True
            Whether to include a short descriptive preamble that describes the graph to the LLM.
        feature_format: FeatureType | str, optional
            The format of the feature (continuous, categorical, boolean).
            If not provided, auto-detection is attempted based on the feature type and the x-axis values.
            A boolean format needs exactly two x-axis values, and raises a ValueError otherwise.
        x_axis_precision: int, optional
            The precision of the values on the x-axis.
            If not provided, the precision is not adjusted.
        y_axis_precision: int | "auto", optional
            The precision of the values on the y-axis.
            If not provided or "auto", the precision is auto-detected based on the range of the y-axis
        confidence_bounds: bool, default=True
            Whether to inlcude confidence bounds.
        confidence_level: float, default=0.95
            The desired confidence level of the bounds.
        max_tokens: int, default=3000
            The token budget of the textual description of the self. The token count is a rough estimate
            (there is no tokenizer before the LLM call), so exceeding the budget only emits a warning.
        overbudget_percent: int, default=50
            How far above max_tokens (in percent) the estimate may go before the graph is considered obviously
            too large. Beyond that bound the function simplifies the graph, or raises if it cannot.
            The LLM provider's own error remains the real limit.
        check_budget: BudgetCheck, optional
            Estimates the tokens of a text and compares them with ``max_tokens`` and ``overbudget_percent``, for
            example ``baml_sdk.check_token_budget``. If not provided, the text is not checked: the function writes it
            once, at the first simplification level, and ``max_tokens`` and ``overbudget_percent`` have no effect.

        Returns
        -------
        str
            The textual representation of the self.

        Raises
        ------
        EBMGraphSimplificationError
            If the graph cannot be simplified enough to fit into the max_tokens limit.
        EBMGraphTokenLimitError
            If the graph exceeds the max_tokens limit even after simplification.

        """
        feature_format = self._resolve_format(feature_format)
        description_text = _describe_format(feature_format)
        self._check_boolean_bins(feature_format)
        simplifiable = feature_format in _SIMPLIFIABLE
        # the precision is fixed by the full graph, so that every simplification level is rounded alike
        if y_axis_precision in {None, "auto"}:
            y_axis_precision = _auto_y_precision(self.scores)

        # simplify the graph until it fits into the max_tokens limit
        simplification_step = 0
        while True:
            simplified_graph = self
            if simplifiable:
                simplified_graph = self.simplify(min_variation_per_cent=simplification_step * _SIMPLIFICATION_STEP)
            prompt = simplified_graph.to_body_text(
                feature_format=feature_format,
                x_axis_precision=x_axis_precision,
                y_axis_precision=y_axis_precision,
                confidence_bounds=confidence_bounds,
                confidence_level=confidence_level,
            )
            if include_description:
                prompt = description_text + prompt

            if check_budget is None:
                return prompt

            # estimate the number of tokens; the provider error is the real limit
            budget = check_budget(
                text=prompt,
                max_tokens=max_tokens,
                overbudget_percent=overbudget_percent,
            )
            match budget.status:
                case "ok":
                    return prompt
                case "warn":
                    warnings.warn(
                        f"The graph for feature {self.feature_name} is estimated at {budget.estimated_tokens} tokens,"
                        f" above the budget of {max_tokens}. The LLM may reject it.",
                        stacklevel=2,  # pragma: no mutate -- skip_file_prefixes sets the stack level itself
                        # skip this file and the frame that beartype adds, so the warning points at the caller
                        skip_file_prefixes=(__file__, "<@beartype"),
                    )
                    return prompt
            # obviously over budget
            if not simplifiable:
                msg = (
                    f"The graph for feature {self.feature_name} of type"
                    f" {self.feature_type} is estimated at {budget.estimated_tokens} tokens and"
                    f" exceeds the token budget of {max_tokens}."
                )
                raise EBMGraphTokenLimitError(msg)
            if simplification_step > _MAX_SIMPLIFICATION_STEPS:
                msg = (
                    f"The graph for feature {self.feature_name} of type"
                    f" {self.feature_type} is estimated at {budget.estimated_tokens} tokens even at"
                    " a simplification level of 10%. This graph is too complex to"
                    f" be passed to the LLM within the token budget of {max_tokens} tokens."
                )
                raise EBMGraphSimplificationError(msg)
            simplification_step += 1

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Parse the textual representation of a graph back into a graph. The inverse of ``to_text``.

        See ``ebaml.graphs.graph_text.parse_graph_text`` for the details of the format.

        Parameters
        ----------
        text: str
            The text of the graph, with the confidence bounds included.

        Returns
        -------
        EBMGraph
            The graph.

        """
        return cls(*parse_graph_text(text))
