"""Helpers for the textual representation of EBM graphs: the JSON lines, and how to read a graph text back."""

import re
from collections.abc import Iterable, Sequence

import orjson
import scipy

from ebaml.core.custom_types import FeatureType, normalize_feature_type
from ebaml.core.exceptions import InvalidOptionError

# the lines of the textual representation of a graph, as written by ``EBMGraph.to_text``
_NAME_LINE = re.compile(r"^[ \t]*Feature Name: (.*)$", re.MULTILINE)
_TYPE_LINE = re.compile(r"^[ \t]*Feature Type: (.*)$", re.MULTILINE)
_MEANS_LINE = re.compile(r"^[ \t]*Means: (.*)$", re.MULTILINE)
_LOWER_LINE = re.compile(r"^[ \t]*Lower Bounds \((\d+(?:\.\d+)?)%-Confidence Interval\): (.*)$", re.MULTILINE)
_UPPER_LINE = re.compile(r"^[ \t]*Upper Bounds \(\d+(?:\.\d+)?%-Confidence Interval\): (.*)$", re.MULTILINE)


def xy_to_json(x_vals: Sequence[str | tuple[float | str, float | str]], y_vals: Iterable[float | str]) -> str:
    """Convert a sequence of x_vals and y_vals to a json string.

    Parameters
    ----------
    x_vals : Sequence[str | tuple[float | str, float | str]]
        The x values. The graph text passes interval ends as rounded strings.
    y_vals : Iterable[float | str]
        The y values. The graph text passes them as rounded strings.

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


def parse_graph_text(
    text: str,
) -> tuple[str, str, list[tuple[float, float]], list[float], list[float]]:
    """Read the fields of a graph from its textual representation. The inverse of ``EBMGraph.to_text``.

    The text may contain other lines (such as the preamble or an LLM prompt around the graph). Only the lines
    that ``EBMGraph.to_text`` writes are read: the name, the type, the means and both confidence bounds. The standard
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
