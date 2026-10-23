"""The textual representation of EBM graphs: what we pass to the LLM, and how to read it back."""

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
import orjson
import scipy

import baml_sdk
from ebaml.custom_types import FeatureType, coerce_enum, normalize_feature_type
from ebaml.exceptions import EBMGraphSimplificationError, EBMGraphTokenLimitError, InvalidOptionError

if TYPE_CHECKING:
    # graphs.py imports this module, so the import would be circular at runtime
    from ebaml.graphs import EBMGraph

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

# the lines of the textual representation of a graph, as written by ``graph_to_text``
_NAME_LINE = re.compile(r"^[ \t]*Feature Name: (.*)$", re.MULTILINE)
_TYPE_LINE = re.compile(r"^[ \t]*Feature Type: (.*)$", re.MULTILINE)
_MEANS_LINE = re.compile(r"^[ \t]*Means: (.*)$", re.MULTILINE)
_LOWER_LINE = re.compile(r"^[ \t]*Lower Bounds \((\d+(?:\.\d+)?)%-Confidence Interval\): (.*)$", re.MULTILINE)
_UPPER_LINE = re.compile(r"^[ \t]*Upper Bounds \(\d+(?:\.\d+)?%-Confidence Interval\): (.*)$", re.MULTILINE)

# the formats whose graphs are simplified until they fit into the token budget
_SIMPLIFIABLE = {FeatureType.CONTINUOUS}

# we assume that the left key (the lower number, typically -1 or 0), is 'False'
_BOOLEAN_KEYS = ["False", "True"]


def xy_to_json(x_vals: list[str] | list[tuple[float, float]], y_vals: list[float]) -> str:
    """Convert a sequence of x_vals and y_vals to a json string.

    Parameters
    ----------
    x_vals : list[str] | list[tuple[float, float]]
        The x values.
    y_vals : list[float]
        The y values.

    Returns
    -------
    str
        A json string representation of the x and y values.

    """
    keys = [f"({x[0]}, {x[1]})" if isinstance(x, tuple) else str(x) for x in x_vals]
    return "{" + ", ".join(f'"{key}": {y}' for key, y in zip(keys, y_vals, strict=True)) + "}"


def parse_str_tuple_to_float_tuple(str_tuple: str) -> tuple[float, float]:
    """Parse a string tuple to a float tuple.

    Parameters
    ----------
    str_tuple : str
        The string tuple to parse.

    Returns
    -------
    tuple[float, float]
        The parsed float tuple.

    """
    return tuple(float(x) for x in str_tuple[1:-1].split(","))


def _is_boolean(x_vals: list) -> bool:
    """Whether the x-axis values are exactly the keys ``False`` and ``True`` (case-insensitive).

    Parameters
    ----------
    x_vals: list
        The values of the x-axis.

    Returns
    -------
    bool
        True if the x-axis values are the keys ``False`` and ``True``.

    """
    return tuple(str(x).upper() for x in x_vals) == ("FALSE", "TRUE")


def _search_line(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    """Find a line of the textual representation of a graph.

    Parameters
    ----------
    pattern: re.Pattern[str]
        The pattern of the line.
    text: str
        The text that contains the graph.
    what: str
        The name of the line, for the error message.

    Returns
    -------
    re.Match[str]
        The match of the first line that fits the pattern.

    Raises
    ------
    ValueError
        If the text has no such line.

    """
    match = pattern.search(text)
    if match is None:
        msg = f"The graph text has no '{what}' line."
        raise ValueError(msg)
    return match


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


def _auto_y_precision(scores: list[float]) -> int:
    """Compute the y-axis precision that keeps at least 3 significant digits of the variation of the scores.

    Parameters
    ----------
    scores: list[float]
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


def _resolve_format(graph: "EBMGraph", feature_format: FeatureType | None) -> FeatureType:
    """Determine the feature format to present, auto-detecting it when it is not given.

    Parameters
    ----------
    graph: EBMGraph
        The graph to present.
    feature_format: FeatureType, optional
        The requested format.

    Returns
    -------
    FeatureType
        The format to present.

    """
    # a simple auto-detect for boolean features
    if _is_boolean(graph.x_vals):
        feature_format = FeatureType.BOOLEAN
    if feature_format is None:
        feature_format = normalize_feature_type(graph.feature_type)
    feature_format = coerce_enum(FeatureType, feature_format)
    return _FORMAT_ALIASES.get(feature_format, feature_format)


def _check_boolean_bins(graph: "EBMGraph", feature_format: FeatureType) -> None:
    """Check that a graph that is presented as boolean has exactly two x-axis values.

    Parameters
    ----------
    graph: EBMGraph
        The graph to present.
    feature_format: FeatureType
        The format to present it in.

    Raises
    ------
    ValueError
        If the format is boolean and the graph does not have exactly two x-axis values.

    """
    match feature_format, graph.x_vals:
        case FeatureType.BOOLEAN, [_, _]:
            pass
        case FeatureType.BOOLEAN, _:
            msg = f"Requested a boolean format, but the feature has more than x-axis values: {graph.x_vals}"
            raise ValueError(msg)


def _render_body(  # ruff: ignore[too-many-arguments] -- keyword-only options that descriptors.py forwards by name
    *,
    graph: "EBMGraph",
    feature_format: FeatureType,
    x_axis_precision: int | None,
    y_axis_precision: int,
    confidence_bounds: bool,
    confidence_level: float,
) -> str:
    """Render the lines of text (name, type, means and optional bounds) that follow the preamble.

    Parameters
    ----------
    graph: EBMGraph
        The graph to render.
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
    scores = graph.scores
    x_vals = graph.x_vals
    # adjust the precision on the x-axis. this can be dangerous if we don't know the distribution of the data
    # (it can impact the accuracy). however, this is a necessary formatting feature because certain (old?)
    # versions of EBMs introduce trailing digits for the xbins (e.g. age as a floating point number with 10
    # significant digits)
    if x_axis_precision is not None:
        x_vals = [
            (np.round(x[0], x_axis_precision).astype(str), np.round(x[1], x_axis_precision).astype(str)) for x in x_vals
        ]
    # formatting for boolean features
    match feature_format:
        case FeatureType.BOOLEAN:
            x_vals = _BOOLEAN_KEYS

    prompt = f"Feature Name: {graph.feature_name}\n"
    prompt += f"Feature Type: {feature_format}\n"
    prompt += f"Means: {xy_to_json(x_vals, np.round(scores, y_axis_precision).astype(str))}\n"
    if confidence_bounds:
        # confidence bounds via normal approximation
        factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
        for label, sign in (("Lower", -1), ("Upper", 1)):
            bounds = [score + sign * factor * std for score, std in zip(scores, graph.stds, strict=True)]
            prompt += (
                f"{label} Bounds ({confidence_level * 100:.0f}%-Confidence Interval):"
                f" {xy_to_json(x_vals, np.round(bounds, y_axis_precision).astype(str))}\n"
            )
    return prompt


def graph_to_text(  # ruff: ignore[too-many-arguments] -- keyword-only options that descriptors.py forwards by name
    graph: "EBMGraph",
    *,
    include_description: bool = True,
    feature_format: FeatureType | None = None,
    x_axis_precision: int | None = None,
    y_axis_precision: int | None = None,
    confidence_bounds: bool = True,
    confidence_level: float = 0.95,
    max_tokens: int = 3000,
    overbudget_percent: int = 50,
) -> str:
    """Convert the graph to text. This is the text that we then pass to the LLM.

    The function takes care of a variety of different formatting issues that can arise in the process of
    converting a graph to text. It is also available as ``EBMGraph.to_text``.

    Parameters
    ----------
    graph: EBMGraph
        The graph to convert.
    include_description: bool, default=True
        Whether to include a short descriptive preamble that describes the graph to the LLM.
    feature_format: FeatureType, optional
        The format of the feature (continuous, categorical, boolean).
        If not provided, auto-detection is attempted based on the feature type and the x-axis values.
        A boolean format needs exactly two x-axis values, and raises a ValueError otherwise.
    x_axis_precision: int, optional
        The precision of the values on the x-axis.
        If not provided, the precision is not adjusted.
    y_axis_precision: int, optional
        The precision of the values on the y-axis.
        If not provided, the precision is auto-detected based on the range of the y-axis
    confidence_bounds: bool, default=True
        Whether to inlcude confidence bounds.
    confidence_level: float, default=0.95
        The desired confidence level of the bounds.
    max_tokens: int, default=3000
        The token budget of the textual description of the graph. The token count is a rough estimate
        (there is no tokenizer before the LLM call), so exceeding the budget only emits a warning.
    overbudget_percent: int, default=50
        How far above max_tokens (in percent) the estimate may go before the graph is considered obviously
        too large. Beyond that bound the function simplifies the graph, or raises if it cannot.
        The LLM provider's own error remains the real limit.

    Returns
    -------
    str
        The textual representation of the graph.

    Raises
    ------
    EBMGraphSimplificationError
        If the graph cannot be simplified enough to fit into the max_tokens limit.
    EBMGraphTokenLimitError
        If the graph exceeds the max_tokens limit even after simplification.

    """
    feature_format = _resolve_format(graph, feature_format)
    description_text = _describe_format(feature_format)
    _check_boolean_bins(graph, feature_format)
    simplifiable = feature_format in _SIMPLIFIABLE
    # the precision is fixed by the full graph, so that every simplification level is rounded alike
    if y_axis_precision in {None, "auto"}:
        y_axis_precision = _auto_y_precision(graph.scores)

    # simplify the graph until it fits into the max_tokens limit
    simplification_step = 0
    while True:
        simplified_graph = graph
        if simplifiable:
            simplified_graph = graph.simplify(min_variation_per_cent=simplification_step * _SIMPLIFICATION_STEP)
        prompt = _render_body(
            graph=simplified_graph,
            feature_format=feature_format,
            x_axis_precision=x_axis_precision,
            y_axis_precision=y_axis_precision,
            confidence_bounds=confidence_bounds,
            confidence_level=confidence_level,
        )
        if include_description:
            prompt = description_text + prompt

        # estimate the number of tokens (in BAML); the provider error is the real limit
        budget = baml_sdk.check_token_budget(
            text=prompt,
            max_tokens=max_tokens,
            overbudget_percent=overbudget_percent,
        )
        match budget.status:
            case "ok":
                return prompt
            case "warn":
                warnings.warn(
                    f"The graph for feature {graph.feature_name} is estimated at {budget.estimated_tokens} tokens,"
                    f" above the budget of {max_tokens}. The LLM may reject it.",
                    stacklevel=2,
                )
                return prompt
        # obviously over budget
        if not simplifiable:
            msg = (
                f"The graph for feature {graph.feature_name} of type"
                f" {graph.feature_type} is estimated at {budget.estimated_tokens} tokens and"
                f" exceeds the token budget of {max_tokens}."
            )
            raise EBMGraphTokenLimitError(msg)
        if simplification_step > _MAX_SIMPLIFICATION_STEPS:
            msg = (
                f"The graph for feature {graph.feature_name} of type"
                f" {graph.feature_type} is estimated at {budget.estimated_tokens} tokens even at"
                " a simplification level of 10%. This graph is too complex to"
                f" be passed to the LLM within the token budget of {max_tokens} tokens."
            )
            raise EBMGraphSimplificationError(msg)
        simplification_step += 1


def parse_graph_text(
    text: str,
) -> tuple[str, str, list[tuple[float, float]], list[float], list[float]]:
    """Read the fields of a graph from its textual representation. The inverse of ``graph_to_text``.

    The text may contain other lines (such as the preamble or an LLM prompt around the graph). Only the lines
    that ``graph_to_text`` writes are read: the name, the type, the means and both confidence bounds. The standard
    deviations are estimated from the bounds with the normal approximation, at the confidence level that the
    bounds are labeled with, so they are only as precise as the rounding of the text.

    Parameters
    ----------
    text: str
        The text of the graph, with the confidence bounds included.

    Returns
    -------
    tuple[str, str, list[tuple[float, float]], list[float], list[float]]
        The feature name, the feature type, the x-axis intervals, the scores and the standard deviations.

    Raises
    ------
    ValueError
        If a line of the graph is missing, is not valid JSON, or the means and the bounds do not share the same
        intervals.
    InvalidOptionError
        If the feature type is not continuous.

    """
    feature_name = _search_line(_NAME_LINE, text, "Feature Name").group(1).strip()
    feature_type = _search_line(_TYPE_LINE, text, "Feature Type").group(1).strip()
    match normalize_feature_type(feature_type):
        case FeatureType.CONTINUOUS:
            pass
        case _:
            raise InvalidOptionError(feature_type, ["continuous"])
    means = orjson.loads(_search_line(_MEANS_LINE, text, "Means").group(1))
    lower_match = _search_line(_LOWER_LINE, text, "Lower Bounds")
    lower = orjson.loads(lower_match.group(2))
    upper = orjson.loads(_search_line(_UPPER_LINE, text, "Upper Bounds").group(1))
    if not means.keys() == lower.keys() == upper.keys():
        msg = "The means and the confidence bounds of the graph text do not have the same intervals."
        raise ValueError(msg)
    confidence_level = float(lower_match.group(1)) / 100
    factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
    return (
        feature_name,
        feature_type,
        [parse_str_tuple_to_float_tuple(interval) for interval in means],
        list(means.values()),
        [(upper[interval] - lower[interval]) / 2 / factor for interval in means],
    )
