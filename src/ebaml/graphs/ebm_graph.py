"""Graphs for EBM models."""

import warnings
from collections.abc import Sequence
from typing import Literal, Protocol, Self, cast, runtime_checkable

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
    EBMGraphShapeError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    InvalidOptionError,
    OptionRangeError,
    UnsupportedModelError,
)
from ebaml.graphs.graph_text import parse_graph_text, xy_to_json
from ebaml.graphs.model_kinds import model_kind

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


# the preamble that a graph of a multiclass classifier adds after the preamble of its format
_MULTICLASS_DESCRIPTION = (
    "The model is a multiclass classifier, so the graph has one series of means and bounds for each class. A line"
    " 'Class: <name>' starts the series of each class. In every bin, the scores have the same mean over all the"
    " classes of the model, so only the differences between the classes change the predicted probabilities.\n\n"
)


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


def _is_boolean(x_vals: Sequence[tuple[float, float]] | Sequence[str]) -> bool:
    """Whether the x-axis values are exactly the keys ``False`` and ``True`` (case-insensitive).

    Parameters
    ----------
    x_vals: Sequence[tuple[float, float]] | Sequence[str]
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


def _auto_y_precision(scores: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.floating]) -> int:
    """Compute the y-axis precision that keeps at least 3 significant digits of the variation of the scores.

    Parameters
    ----------
    scores: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.floating]
        The scores of the graph, of every class.

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


def _class_columns(names: Sequence[str], selected: Sequence[str] | None) -> list[int]:
    """Find the columns of the selected classes.

    Parameters
    ----------
    names: Sequence[str]
        The names of all the classes, in the order of the columns.
    selected: Sequence[str], optional
        The classes to keep, in the order to show them. If not provided, every class is kept.

    Returns
    -------
    list[int]
        The column of each selected class, without repeats.

    Raises
    ------
    InvalidOptionError
        If a selected class is not a class of the model, the selection is empty, or it is one string.

    """
    if selected is None:
        return list(range(len(names)))
    # a string is a sequence of strings too, so without this check "setosa" would select the classes "s", "e", ...
    if isinstance(selected, str):
        msg = f"classes={selected!r} (pass a list of class names)"
        raise InvalidOptionError(msg, names)
    if not selected:
        raise InvalidOptionError(str(list(selected)), names)
    for name in selected:
        if name not in names:
            raise InvalidOptionError(name, names)
    return [names.index(name) for name in dict.fromkeys(selected)]


@attrs.frozen
class _Selected:
    """The scores and the standard deviations of the selected outputs, with the names of the classes."""

    scores: npt.NDArray[np.float64]
    stds: npt.NDArray[np.float64]
    classes: list[str]


def _select_outputs(
    outputs: Sequence[str],
    feature_name: str,
    scores: npt.NDArray[np.float64],
    stds: npt.NDArray[np.float64],
    classes: Sequence[str] | None,
) -> _Selected:
    """Centre the scores of a multiclass graph over the classes, and keep the selected classes.

    Parameters
    ----------
    outputs: Sequence[str]
        The names of the outputs of the model.
    feature_name: str
        The name of the feature, for the error messages.
    scores: npt.NDArray[np.float64]
        The scores of each bin: one per bin, or one row per bin with one column per output.
    stds: npt.NDArray[np.float64]
        The standard deviations, with the shape of the scores.
    classes: Sequence[str], optional
        The classes to keep. See ``EBMGraph.from_ebm``.

    Returns
    -------
    _Selected
        The scores and the standard deviations of the classes to show. A model with one output has no classes.

    Raises
    ------
    EBMGraphShapeError
        If the scores of a bin do not match the outputs of the model.
    UnsupportedModelError
        If ``classes`` is given for a model with one output.

    """
    # a model with several outputs (a multiclass classifier) has one column of scores per output
    columns_per_bin = (len(outputs),) if len(outputs) > 1 else ()
    if scores.shape[1:] != columns_per_bin:
        msg = f"The scores of {feature_name} have the shape {scores.shape}, but the model has {len(outputs)} outputs."
        raise EBMGraphShapeError(msg)
    if not columns_per_bin:
        if classes is not None:
            msg = f"The option classes needs a multiclass classifier, but the graph of {feature_name} has one output."
            raise UnsupportedModelError(msg)
        return _Selected(scores, stds, [])
    centred = scores - scores.mean(axis=1, keepdims=True)
    columns = _class_columns(outputs, classes)
    return _Selected(centred[:, columns], stds[:, columns], [outputs[column] for column in columns])


@attrs.frozen
class _Series:
    """The scores of one output of a graph, with the name of its class (``None`` for a graph with one output)."""

    label: str | None
    scores: npt.NDArray[np.float64]
    stds: npt.NDArray[np.float64]


@attrs.define
class EBMGraph:
    """A datastructure for the graphs of Explainable Boosting Machines.

    A graph of a binary classifier or a regressor has one score per bin, and ``classes`` is empty. A graph of a
    multiclass classifier has one row per bin with one score per class, and ``classes`` names the columns.
    """

    feature_name: str = attrs.field()
    feature_type: str = attrs.field()
    # intervals for continuous features, categories for nominal ones
    x_vals: Sequence[tuple[float, float]] | Sequence[str] = attrs.field()
    # from_ebm stores NumPy arrays here; shape (n_bins,), or (n_bins, n_classes) for a multiclass classifier
    scores: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.floating] = attrs.field()
    stds: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.floating] = attrs.field()
    # the class of each column of scores and stds; empty for a graph with one output
    classes: Sequence[str] = attrs.field(default=())

    @classmethod
    def from_ebm(
        cls,
        ebm: ExplainableBoostingModel,
        feature_index: int,
        normalization: Normalization | str = Normalization.NONE,
        *,
        use_feature_bounds: bool = True,
        classes: Sequence[str] | None = None,
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
        classes: Sequence[str], optional
            The classes to keep in the graph of a multiclass classifier, in the order to show them. If not provided,
            the graph has every class. The text of a model with many classes is long, so a subset keeps it short.
            A model with one output raises ``UnsupportedModelError`` for this option, and so does a model of an
            unknown kind for any call. Scores that do not match the outputs of the model raise ``EBMGraphShapeError``.

        Returns
        -------
        EBMGraph
            The graph.

        Raises
        ------
        InvalidOptionError
            If the feature type is not continuous or nominal.
        InvalidOptionError.from_enum
            If the normalization option is not valid.
        EBMGraphShapeError
            If the number of bins does not match the number of scores.

        Notes
        -----
        A multiclass classifier has one score per bin and per class. Softmax does not change when the same constant is
        added to the scores of all the classes of a bin, so the graph centres the scores of each bin over all the
        classes of the model, before it selects ``classes``. The standard deviations stay those of the EBM.
        The normalization then shifts all the scores by one constant, so the differences between the classes of a bin
        do not change.
        ``proofs/papers/multiclass-scores/paper.typ`` proves that these graphs keep the predictions of the EBM.

        """
        # read the variables from the ebm
        feature_name = ebm.feature_names_in_[feature_index]
        feature_type = ebm.feature_types_in_[feature_index]
        scores = np.array(ebm.term_scores_[feature_index][1:-1], dtype=float)  # Drop missing and unknown bins
        stds = np.asarray(ebm.standard_deviations_[feature_index][1:-1], dtype=float)
        selected = _select_outputs(model_kind(ebm).outputs, feature_name, scores, stds, classes)
        scores, stds = selected.scores, selected.stds

        # normalize the graph by one constant for all the classes, which keeps the differences between the classes
        match normalization:
            case Normalization.Z_SCORE:
                normalization_constant = np.mean(scores)
            case Normalization.MIN_MAX:
                normalization_constant = np.min(scores)
            case Normalization.NONE:
                normalization_constant = 0
            case _:
                raise InvalidOptionError.from_enum(normalization, Normalization)
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
                # interpret types bins_ as arrays, but the bins of a nominal feature are a dict of category to position
                bins = cast("dict[str, int]", ebm.bins_[feature_index][0])
                positions = {position - 1: category for category, position in bins.items()}
                x_vals = [positions[idx] for idx in range(len(positions))]
            case _:
                raise InvalidOptionError(feature_type, ["continuous", "nominal"])
        if len(x_vals) != len(scores):
            msg = (
                f"The number of bins ({len(x_vals)}) and scores ({len(scores)}) does not match"
                f" for feature {feature_name} of type {feature_type}."
            )
            raise EBMGraphShapeError(msg)
        return cls(feature_name, feature_type, x_vals, scores, stds, selected.classes)

    def simplify(self, min_variation_per_cent: float = 0.0) -> Self:
        """Simplifies the graph. Removes redundant (flat) bins from the graph.

        Parameters
        ----------
        min_variation_per_cent: float, default=0.0
            Parameter that controlls the degree of simplification.
            If min_variation_per_cent>0, the function simplifies the graph by removing bins that correspond to a less
            that min_variation_per_cent change in the score, considering the overal min/max difference of score for the
            feature as 100%. A bin of a multiclass graph merges only if the score of every class changes that little.

        Returns
        -------
        EBMGraph:
            The simplified graph.

        Raises
        ------
        InvalidOptionError
            If the feature type is not continuous.
        OptionRangeError
            If min_variation_per_cent is not between 0 and 1.

        """
        match normalize_feature_type(self.feature_type):
            case FeatureType.CONTINUOUS:
                pass
            case _:
                raise InvalidOptionError(self.feature_type, ["continuous"])
        if not 0 <= min_variation_per_cent <= 1:
            msg = f"min_variation_per_cent must be between 0 and 1. Got {min_variation_per_cent}."
            raise OptionRangeError(msg)
        # a continuous graph has intervals of numbers on its x-axis, so float() changes no value
        x_vals = [(float(low), float(high)) for low, high in self.x_vals]
        scores, stds = np.asarray(self.scores, dtype=float), np.asarray(self.stds, dtype=float)
        threshold = float(np.max(scores) - np.min(scores)) * min_variation_per_cent
        # Python floats, one row per bin, because NumPy calls on single bins are slow in this loop
        rows = scores.reshape(len(x_vals), -1).tolist()
        # the bins of each merged bin; a bin joins the last group if no score moved more than the threshold
        groups = [[0]]
        later_bins = range(1, len(x_vals))  # pragma: no mutate -- the first bin compared with itself changes nothing
        for idx in later_bins:
            first = rows[groups[-1][0]]
            if max(abs(a - b) for a, b in zip(first, rows[idx], strict=True)) <= threshold:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        # the groups are runs of neighbouring bins, so reduceat over the first bin of each group reduces each run
        firsts = [group[0] for group in groups]
        return type(self)(
            self.feature_name,
            self.feature_type,
            [(x_vals[group[0]][0], x_vals[group[-1]][1]) for group in groups],
            scores[firsts].tolist(),
            # the largest std of a merged bin, so the confidence bands of the simplified graph cover the original graph
            # as well as its confidence bands
            np.maximum.reduceat(stds, firsts, axis=0).tolist(),
            self.classes,
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
        EBMGraphShapeError
            If the format is boolean and the graph does not have exactly two x-axis values.

        """
        match feature_format, self.x_vals:
            case FeatureType.BOOLEAN, [_, _]:
                pass
            case FeatureType.BOOLEAN, _:
                msg = f"Requested a boolean format, but the feature has more than x-axis values: {self.x_vals}"
                raise EBMGraphShapeError(msg)

    def _series(self) -> list[_Series]:
        """Split the scores into one series for each output of the graph.

        Returns
        -------
        list[_Series]
            One series for a graph with one output, else one series per class.

        Raises
        ------
        EBMGraphShapeError
            If the scores and the standard deviations do not have the same shape, or the scores do not have one column
            per class.

        """
        scores, stds = np.asarray(self.scores, dtype=float), np.asarray(self.stds, dtype=float)
        if scores.shape != stds.shape:
            msg = f"The graph of {self.feature_name} has scores of shape {scores.shape} but stds of shape {stds.shape}."
            raise EBMGraphShapeError(msg)
        columns_per_bin = (len(self.classes),) if self.classes else ()
        if scores.shape[1:] != columns_per_bin:
            msg = (
                f"The graph of {self.feature_name} has scores of shape {scores.shape} for {len(self.classes)} classes."
            )
            raise EBMGraphShapeError(msg)
        if not self.classes:
            return [_Series(None, scores, stds)]
        return [_Series(label, scores[:, column], stds[:, column]) for column, label in enumerate(self.classes)]

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
            The textual representation of the graph, without the preamble. A multiclass graph has a ``Class:`` line
            before the means and the bounds of each class.

        """
        x_vals = self.x_vals
        # adjust the precision on the x-axis. this can be dangerous if we don't know the distribution of the data
        # (it can impact the accuracy). however, this is a necessary formatting feature because certain (old?)
        # versions of EBMs introduce trailing digits for the xbins (e.g. age as a floating point number with 10
        # significant digits)
        # only an interval has ends to round: a category name stays as it is
        if x_axis_precision is not None:
            x_vals = [
                (
                    np.round(float(x[0]), x_axis_precision).astype(str),
                    np.round(float(x[1]), x_axis_precision).astype(str),
                )
                if isinstance(x, tuple)
                else x
                for x in x_vals
            ]
        # formatting for boolean features
        match feature_format:
            case FeatureType.BOOLEAN:
                x_vals = _BOOLEAN_KEYS
            case _:
                pass

        prompt = f"Feature Name: {self.feature_name}\n"
        prompt += f"Feature Type: {feature_format}\n"
        for series in self._series():
            if series.label is not None:
                prompt += f"Class: {series.label}\n"
            prompt += f"Means: {xy_to_json(x_vals, np.round(series.scores, y_axis_precision).astype(str))}\n"
            if confidence_bounds:
                # confidence bounds via normal approximation
                factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
                for label, sign in (("Lower", -1), ("Upper", 1)):
                    bounds = series.scores + sign * factor * series.stds
                    prompt += (
                        f"{label} Bounds ({confidence_level * 100:.10g}%-Confidence Interval):"
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
            Whether to include a short descriptive preamble that describes the graph to the LLM. The preamble of a
            multiclass graph also says how to read the series of the classes.
        feature_format: FeatureType | str, optional
            The format of the feature (continuous, categorical, boolean).
            If not provided, auto-detection is attempted based on the feature type and the x-axis values.
            A boolean format needs exactly two x-axis values, and raises an EBMGraphShapeError otherwise.
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
        # check the shape before the simplification, which would hide or garble a shape that does not fit the classes
        self._series()
        feature_format = self._resolve_format(feature_format)
        description_text = _describe_format(feature_format) + (_MULTICLASS_DESCRIPTION if self.classes else "")
        self._check_boolean_bins(feature_format)
        simplifiable = feature_format in _SIMPLIFIABLE
        # the precision is fixed by the full graph, so that every simplification level is rounded alike
        if y_axis_precision in {None, "auto"}:
            y_axis_precision = _auto_y_precision(self.scores)

        # simplify the graph until it fits into the max_tokens limit
        simplification_step = 0
        while True:
            simplified_graph = (
                self.simplify(min_variation_per_cent=simplification_step * _SIMPLIFICATION_STEP)
                if simplifiable
                else self
            )
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
                case _:
                    # the status is "over": obviously over budget
                    if not simplifiable:
                        msg = (
                            f"The graph for feature {self.feature_name} of type"
                            f" {self.feature_type} is estimated at {budget.estimated_tokens} tokens and"
                            f" exceeds the token budget of {max_tokens}."
                        )
                        raise EBMGraphTokenLimitError(msg)
                    if simplification_step >= _MAX_SIMPLIFICATION_STEPS:
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
        return cls(**attrs.asdict(parse_graph_text(text), recurse=False))
