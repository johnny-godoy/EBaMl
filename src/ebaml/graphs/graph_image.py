"""Draw the graph of a feature as an image: SVG for the documentation site, and PNG for a vision model.

The image and the text of a graph read the same ``EBMGraph``: the same series (``EBMGraph.series``), the same feature
format (``EBMGraph.resolve_format``) and the same confidence bounds (``confidence_factor``). So the two views agree bin
for bin. The image shows the graph that it gets, with no simplification.

The functions return the image in memory and write no file. The drawing needs ``matplotlib``, which the extra
``vision`` installs. An import of this module without ``matplotlib`` raises ``MissingExtraError``.
"""

import io
import threading
from collections.abc import Callable, Mapping, Sequence

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import FeatureType, XValues
from ebaml.core.exceptions import EBMGraphShapeError, MissingExtraError
from ebaml.graphs.ebm_graph import BOOLEAN_KEYS, EBMGraph
from ebaml.graphs.graph_text import confidence_factor

try:
    from matplotlib import rc_context, style
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as error:
    msg = 'A graph image needs matplotlib. Install the extra: pip install "ebaml[vision]".'
    raise MissingExtraError(msg) from error

# the size of the image: 640 x 360 pixels
_SIZE_INCHES = (6.4, 3.6)
_DOTS_PER_INCH = 100
# one color for each series, in the order of the classes; a graph with more classes than colors uses them again
_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
_BAND_OPACITY = 0.2
_LINE_WIDTH = 2
# the share of the space of one category that its group of bars fills
_BAR_GROUP_WIDTH = 0.8
_CAP_SIZE = 3
# fixed settings, so that the same graph gives the same bytes with any matplotlibrc. DejaVu Sans comes with
# matplotlib. The SVG keeps its text as text, and a name with "$" is not read as a formula.
_SETTINGS: Mapping[str, str | float | bool] = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "svg.fonttype": "none",
    "svg.hashsalt": "ebaml",
    "text.parse_math": False,
}
# no date and no version of matplotlib in the file, so that the bytes stay the same from run to run
_METADATA: Mapping[str, Mapping[str, None]] = {
    "svg": {"Date": None, "Creator": None},
    "png": {"Software": None},
}
# matplotlib keeps its settings in one global object, so one lock keeps two drawings of a thread pool apart
_LOCK = threading.Lock()


@attrs.frozen
class _Band:
    """The scores of one series of a graph, with the bounds of their confidence interval.

    Attributes
    ----------
    label: str | None
        The class of the series, or ``None`` for a graph with one output.
    means: npt.NDArray[np.float64]
        The score of each bin.
    lower, upper: npt.NDArray[np.float64]
        The bounds of the confidence interval of each bin. They equal ``means`` for an image with no bounds.

    """

    label: str | None
    means: npt.NDArray[np.float64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


def _bands(graph: EBMGraph, confidence_level: float | None) -> list[_Band]:
    """Compute the bounds of each series of a graph, as the graph text computes them.

    Parameters
    ----------
    graph: EBMGraph
        The graph.
    confidence_level: float | None
        The confidence level of the bounds, or ``None`` for no bounds.

    Returns
    -------
    list[_Band]
        One band for each series of the graph.

    """
    factor = 0.0 if confidence_level is None else confidence_factor(confidence_level)
    return [
        _Band(series.label, series.scores, series.scores - factor * series.stds, series.scores + factor * series.stds)
        for series in graph.series()
    ]


def _edges(x_vals: XValues) -> npt.NDArray[np.float64]:
    """Find the edges of the bins of a continuous feature.

    An infinite end (``use_feature_bounds=False``) cannot be drawn. The image draws it one typical bin width beyond the
    next edge. The typical width is the median of the finite widths.

    Parameters
    ----------
    x_vals: XValues
        The intervals of the bins, in order.

    Returns
    -------
    npt.NDArray[np.float64]
        The low end of each bin, then the high end of the last bin.

    Raises
    ------
    EBMGraphShapeError
        If a bin does not start where the bin before it ends.

    """
    intervals = np.asarray([(float(low), float(high)) for low, high in x_vals], dtype=float)
    if not np.array_equal(intervals[1:, 0], intervals[:-1, 1]):
        msg = "The graph image needs bins that touch: each bin must start where the bin before it ends."
        raise EBMGraphShapeError(msg)
    edges = np.append(intervals[:, 0], intervals[-1, 1])
    widths = np.diff(edges)
    finite_widths = widths[np.isfinite(widths)]
    typical_width = float(np.median(finite_widths)) if finite_widths.size else 1.0
    if np.isinf(edges).all():
        # one bin from -inf to inf has no finite edge to start from
        return np.array([0.0, typical_width])
    if np.isinf(edges[0]):
        edges[0] = edges[1] - typical_width
    if np.isinf(edges[-1]):
        edges[-1] = edges[-2] + typical_width
    return edges


def _paint_steps(axes: Axes, graph: EBMGraph, bands: Sequence[_Band]) -> str:
    """Draw each series as a step line through the bins, with its confidence interval as a band.

    Parameters
    ----------
    axes: Axes
        The plot area.
    graph: EBMGraph
        The graph of a continuous feature.
    bands: Sequence[_Band]
        The series and their bounds.

    Returns
    -------
    str
        The name of the mark that shows the confidence interval.

    """
    edges = _edges(graph.x_vals)
    for band_index, band in enumerate(bands):
        color = _COLORS[band_index % len(_COLORS)]
        axes.stairs(band.upper, edges, baseline=band.lower, fill=True, color=color, alpha=_BAND_OPACITY, linewidth=0)
        axes.stairs(band.means, edges, baseline=None, color=color, linewidth=_LINE_WIDTH, label=band.label)
    return "band"


def _paint_bars(axes: Axes, categories: Sequence[str], bands: Sequence[_Band]) -> str:
    """Draw one group of bars for each category, with one bar for each series and its confidence interval.

    Parameters
    ----------
    axes: Axes
        The plot area.
    categories: Sequence[str]
        The name of each category, in the order of the bins.
    bands: Sequence[_Band]
        The series and their bounds.

    Returns
    -------
    str
        The name of the mark that shows the confidence interval.

    """
    positions = np.arange(len(categories), dtype=float)
    bar_width = _BAR_GROUP_WIDTH / len(bands)
    for band_index, band in enumerate(bands):
        offset = (band_index - (len(bands) - 1) / 2) * bar_width
        axes.bar(
            positions + offset,
            band.means,
            bar_width,
            yerr=np.vstack([band.means - band.lower, band.upper - band.means]),
            color=_COLORS[band_index % len(_COLORS)],
            capsize=_CAP_SIZE,
            label=band.label,
        )
    axes.set_xticks(positions, labels=list(categories))
    return "error bars"


def _paint_categories(axes: Axes, graph: EBMGraph, bands: Sequence[_Band]) -> str:
    """Draw the bars of a categorical feature, with the names of its categories.

    Parameters
    ----------
    axes: Axes
        The plot area.
    graph: EBMGraph
        The graph of a categorical feature.
    bands: Sequence[_Band]
        The series and their bounds.

    Returns
    -------
    str
        The name of the mark that shows the confidence interval.

    """
    return _paint_bars(axes, [str(category) for category in graph.x_vals], bands)


def _paint_booleans(axes: Axes, _graph: EBMGraph, bands: Sequence[_Band]) -> str:
    """Draw the bars of a boolean feature, with the keys ``False`` and ``True`` that the graph text writes.

    Parameters
    ----------
    axes: Axes
        The plot area.
    _graph: EBMGraph
        The graph of a boolean feature. ``EBMGraph.resolve_format`` checked that it has two bins.
    bands: Sequence[_Band]
        The series and their bounds.

    Returns
    -------
    str
        The name of the mark that shows the confidence interval.

    """
    return _paint_bars(axes, BOOLEAN_KEYS, bands)


# the drawing of each feature format; a new format is a new function and a new row
_PAINTERS: Mapping[FeatureType, Callable[[Axes, EBMGraph, Sequence[_Band]], str]] = {
    FeatureType.CONTINUOUS: _paint_steps,
    FeatureType.CATEGORICAL: _paint_categories,
    FeatureType.BOOLEAN: _paint_booleans,
}


@attrs.frozen(kw_only=True)
class GraphImageOptions:
    """How to draw the image of a graph.

    Attributes
    ----------
    score_axis: str, default="score"
        The title of the score axis, for example ``ModelKind.score_axis``.
    title: str, optional
        The title above the plot. If not provided, the title names the feature.
    feature_format: FeatureType | str, optional
        The format of the feature, as ``EBMGraph.to_text`` reads it. If not provided, it is detected from the graph.
    confidence_level: float | None, default=0.95
        The confidence level of the band or the error bars. ``None`` draws no bounds.

    """

    score_axis: str = "score"
    title: str | None = None
    feature_format: FeatureType | str | None = None
    confidence_level: float | None = 0.95


def graph_figure(graph: EBMGraph, options: GraphImageOptions | None = None) -> Figure:
    """Draw the graph of a feature as a matplotlib figure.

    ``graph_svg`` and ``graph_png`` save this figure with fixed settings. Call this function to change the figure
    before you save it. It reads the matplotlib settings that are active when you call it.

    Parameters
    ----------
    graph: EBMGraph
        The graph.
    options: GraphImageOptions, optional
        The labels, the feature format and the confidence level. If not provided, the defaults of
        ``GraphImageOptions`` apply.

    Returns
    -------
    Figure
        The figure: a step line with a band for a continuous feature, and bars with error bars for a categorical or
        boolean feature. A multiclass graph has one series for each class and a legend.

    """
    options = GraphImageOptions() if options is None else options
    presented = graph.resolve_format(options.feature_format)
    bands = _bands(graph, options.confidence_level)
    figure = Figure(figsize=_SIZE_INCHES, dpi=_DOTS_PER_INCH, layout="constrained", facecolor="white")
    axes = figure.add_subplot()
    axes.axhline(0.0, color="#888888", linestyle="--", linewidth=1)
    mark = _PAINTERS[presented](axes, graph, bands)
    axes.set_title(f"Graph of {graph.feature_name}" if options.title is None else options.title)
    axes.set_xlabel(graph.feature_name)
    axes.set_ylabel(options.score_axis)
    if graph.classes:
        axes.legend(title="class")
    if options.confidence_level is not None:
        caption = f"{mark}: {options.confidence_level * 100:.10g}% confidence interval"
        figure.text(0.99, 0.01, caption, ha="right", fontsize=8)
    return figure


def _render(graph: EBMGraph, options: GraphImageOptions | None, file_format: str) -> bytes:
    """Draw a graph with the fixed settings, and save it in a file format, in memory.

    Parameters
    ----------
    graph: EBMGraph
        The graph.
    options: GraphImageOptions, optional
        The labels, the feature format and the confidence level.
    file_format: str
        ``"svg"`` or ``"png"``.

    Returns
    -------
    bytes
        The content of the file.

    """
    buffer = io.BytesIO()
    with _LOCK, style.context("default"), rc_context(dict(_SETTINGS)):
        figure = graph_figure(graph, options)
        figure.savefig(buffer, format=file_format, metadata=dict(_METADATA[file_format]))
    return buffer.getvalue()


def graph_svg(graph: EBMGraph, options: GraphImageOptions | None = None) -> str:
    """Draw the graph of a feature as an SVG document of 640 by 360 pixels, on a white background.

    The labels stay text in the document. The same graph gives the same document from run to run.

    Parameters
    ----------
    graph: EBMGraph
        The graph to draw.
    options: GraphImageOptions, optional
        The options of ``graph_figure``.

    Returns
    -------
    str
        The SVG document.

    Examples
    --------
    >>> graph = EBMGraph("age", "continuous", [(18.0, 30.0), (30.0, 60.0)], [-0.5, 0.25], [0.1, 0.05])
    >>> "Graph of age" in graph_svg(graph)
    True

    """
    return _render(graph, options, "svg").decode()


def graph_png(graph: EBMGraph, options: GraphImageOptions | None = None) -> bytes:
    """Draw the graph of a feature as a PNG image of 640 by 360 pixels, the format that a vision model reads.

    The same graph gives the same bytes from run to run.

    Parameters
    ----------
    graph: EBMGraph
        The graph that the model reads.
    options: GraphImageOptions, optional
        The labels and the bounds, as ``graph_figure`` reads them.

    Returns
    -------
    bytes
        The PNG image.

    Examples
    --------
    >>> graph = EBMGraph("color", "nominal", ["red", "blue"], [0.3, -0.3], [0.1, 0.1])
    >>> graph_png(graph, GraphImageOptions(score_axis="change of the predicted value"))[1:4]
    b'PNG'

    """
    return _render(graph, options, "png")
