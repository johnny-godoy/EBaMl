"""Helpers for the textual representation of EBM graphs: the JSON lines, and how to read a graph text back."""

import re
from collections.abc import Iterable, Sequence

import attrs
import orjson
import scipy

from ebaml.core.custom_types import FeatureType, normalize_feature_type
from ebaml.core.exceptions import GraphTextError, InvalidOptionError

# the lines of the textual representation of a graph, as written by ``EBMGraph.to_text``
_NAME_LINE = re.compile(r"^[ \t]*Feature Name: (.*)$", re.MULTILINE)
_TYPE_LINE = re.compile(r"^[ \t]*Feature Type: (.*)$", re.MULTILINE)
_MEANS_LINE = re.compile(r"^[ \t]*Means: (.*)$", re.MULTILINE)
_LOWER_LINE = re.compile(r"^[ \t]*Lower Bounds \((\d+(?:\.\d+)?)%-Confidence Interval\): (.*)$", re.MULTILINE)
_UPPER_LINE = re.compile(r"^[ \t]*Upper Bounds \(\d+(?:\.\d+)?%-Confidence Interval\): (.*)$", re.MULTILINE)
_CLASS_LINE = re.compile(r"^[ \t]*Class: (.*)$", re.MULTILINE)


def xy_to_json(x_vals: Sequence[str | tuple[float | str, float | str]], y_vals: Iterable[float | str]) -> str:
    r"""Convert a sequence of x_vals and y_vals to a json string.

    Parameters
    ----------
    x_vals : Sequence[str | tuple[float | str, float | str]]
        The x values. The graph text passes interval ends as rounded strings.
    y_vals : Iterable[float | str]
        The y values. The graph text passes them as rounded strings.

    Returns
    -------
    str
        A json string representation of the x and y values. The keys are escaped as JSON strings. The y values are
        written as they are, to keep their exact digits.

    Examples
    --------
    >>> xy_to_json([("0.0", "1.5"), "red"], ["0.25", "-1.0"])
    '{"(0.0, 1.5)": 0.25, "red": -1.0}'
    >>> print(xy_to_json(['Master "B"'], ["0.5"]))
    {"Master \"B\"": 0.5}

    """
    keys = [f"({x[0]}, {x[1]})" if isinstance(x, tuple) else x for x in x_vals]
    return "{" + ", ".join(f"{orjson.dumps(key).decode()}: {y}" for key, y in zip(keys, y_vals, strict=True)) + "}"


# nosemgrep: ebaml-return-attrs-not-container -- an interval is a homogeneous pair, like EBMGraph.x_vals
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

    Raises
    ------
    GraphTextError
        If the text is not two numbers in brackets.

    Examples
    --------
    >>> parse_str_tuple_to_float_tuple("(0.5, 2)")
    (0.5, 2.0)
    >>> parse_str_tuple_to_float_tuple("(a, b)")
    Traceback (most recent call last):
    ...
    ebaml.core.exceptions.GraphTextError: '(a, b)' is not an interval of two numbers.

    """
    try:
        low, high = (float(part) for part in str_tuple[1:-1].split(","))
    except ValueError as error:
        msg = f"{str_tuple!r} is not an interval of two numbers."
        raise GraphTextError(msg) from error
    return low, high


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
    GraphTextError
        If the text has no such line.

    """
    match = pattern.search(text)
    if match is None:
        msg = f"The graph text has no '{what}' line."
        raise GraphTextError(msg)
    return match


# nosemgrep: ebaml-return-attrs-not-container -- the keys are the intervals of the line, so they are data
def _load_line(line: str, what: str) -> dict[str, float]:
    """Read the JSON object of one line of the textual representation of a graph.

    Parameters
    ----------
    line: str
        The JSON text of the line.
    what: str
        The name of the line, for the error message.

    Returns
    -------
    dict[str, float]
        The value of each interval.

    Raises
    ------
    GraphTextError
        If the line is not valid JSON.

    """
    try:
        values: dict[str, float] = orjson.loads(line)
    except orjson.JSONDecodeError as error:
        msg = f"The '{what}' line of the graph text is not valid JSON."
        raise GraphTextError(msg) from error
    return values


@attrs.frozen
class _Series:
    """The intervals, the means and the standard deviations of one series of a graph text."""

    intervals: list[str]
    scores: list[float]
    stds: list[float]


def _parse_series(text: str) -> _Series:
    """Read the means and the confidence bounds of one series of a graph text.

    Parameters
    ----------
    text: str
        The text of the series: the means line and both bounds lines.

    Returns
    -------
    _Series
        The intervals, the means and the standard deviations estimated from the bounds.

    Raises
    ------
    GraphTextError
        If a line is missing, is not valid JSON, or the means and the bounds do not share the same intervals.

    """
    means = _load_line(_search_line(_MEANS_LINE, text, "Means").group(1), "Means")
    lower_match = _search_line(_LOWER_LINE, text, "Lower Bounds")
    lower = _load_line(lower_match.group(2), "Lower Bounds")
    upper = _load_line(_search_line(_UPPER_LINE, text, "Upper Bounds").group(1), "Upper Bounds")
    if not means.keys() == lower.keys() == upper.keys():
        msg = "The means and the confidence bounds of the graph text do not have the same intervals."
        raise GraphTextError(msg)
    confidence_level = float(lower_match.group(1)) / 100
    factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
    return _Series(
        intervals=list(means),
        scores=list(means.values()),
        stds=[(upper[interval] - lower[interval]) / 2 / factor for interval in means],
    )


@attrs.frozen
class ParsedGraph:
    """The fields of a graph that ``parse_graph_text`` reads from a text. They are the fields of ``EBMGraph``."""

    feature_name: str
    feature_type: str
    x_vals: list[tuple[float, float]]
    # one score per bin, or one row per bin with one score per class
    scores: list[float] | list[list[float]]
    stds: list[float] | list[list[float]]
    classes: list[str]


def parse_graph_text(text: str) -> ParsedGraph:
    """Read the fields of a graph from its textual representation. The inverse of ``EBMGraph.to_text``.

    The text may contain other lines (such as the preamble or an LLM prompt around the graph). Only the lines
    that ``EBMGraph.to_text`` writes are read: the name, the type, the means and both confidence bounds. The standard
    deviations are estimated from the bounds with the normal approximation, at the confidence level that the
    bounds are labeled with, so they are only as precise as the rounding of the text.

    A graph of a multiclass classifier has one series per class: a ``Class:`` line, then the means and the bounds of
    that class. The scores and the standard deviations then have one row per bin and one column per class.

    Parameters
    ----------
    text: str
        The text of the graph, with the confidence bounds included.

    Returns
    -------
    ParsedGraph
        The feature name, the feature type, the x-axis intervals, the scores, the standard deviations and the classes.

    Raises
    ------
    GraphTextError
        If a line of the graph is missing, is not valid JSON, the means and the bounds do not share the same
        intervals, or a multiclass text has a series before its first ``Class:`` line.
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
    class_lines = list(_CLASS_LINE.finditer(text))
    if not class_lines:
        series = _parse_series(text)
        return ParsedGraph(
            feature_name=feature_name,
            feature_type=feature_type,
            x_vals=[parse_str_tuple_to_float_tuple(interval) for interval in series.intervals],
            scores=series.scores,
            stds=series.stds,
            classes=[],
        )
    if _MEANS_LINE.search(text, 0, class_lines[0].start()):
        msg = "The graph text has a series before its first 'Class' line."
        raise GraphTextError(msg)
    # the series of a class runs from its Class line to the next one
    ends = [match.start() for match in class_lines[1:]] + [len(text)]
    by_class = [_parse_series(text[match.end() : end]) for match, end in zip(class_lines, ends, strict=True)]
    if any(series.intervals != by_class[0].intervals for series in by_class):
        msg = "The classes of the graph text do not have the same intervals."
        raise GraphTextError(msg)
    return ParsedGraph(
        feature_name=feature_name,
        feature_type=feature_type,
        x_vals=[parse_str_tuple_to_float_tuple(interval) for interval in by_class[0].intervals],
        scores=[list(row) for row in zip(*(series.scores for series in by_class), strict=True)],
        stds=[list(row) for row in zip(*(series.stds for series in by_class), strict=True)],
        # to_text writes each label as it is, so a space at either end belongs to the label
        classes=[match.group(1) for match in class_lines],
    )
