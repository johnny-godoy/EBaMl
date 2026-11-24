"""Tests for the images of the graphs: the drawn values agree with the graph text, and the bytes stay the same."""

import html
import importlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch

import ebaml
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EBMGraphShapeError, MissingExtraError
from ebaml.graphs import graph_image
from ebaml.graphs.ebm_graph import EBMGraph, confidence_factor
from ebaml.graphs.graph_image import GraphImageOptions, graph_figure, graph_png, graph_svg
from ebaml.graphs.model_kinds import model_kind

INF = float("inf")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CONTINUOUS = "a"
CATEGORICAL = "c"
KINDS = ("binary", "multiclass", "regressor")


@pytest.fixture(scope="module")
def models(
    continuous_ebm: tuple[ExplainableBoostingClassifier, object],
    multiclass_ebm: ExplainableBoostingClassifier,
    regression_ebm: ExplainableBoostingRegressor,
) -> dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor]:
    """Give one fitted model of each kind, by the name of the kind.

    Returns
    -------
    dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor]
        The binary classifier, the multiclass classifier and the regressor. Each has the features ``a`` and ``c``.

    """
    return {"binary": continuous_ebm[0], "multiclass": multiclass_ebm, "regressor": regression_ebm}


def graph_of(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
    feature: str,
) -> EBMGraph:
    model = models[kind]
    return EBMGraph.from_ebm(model, list(model.feature_names_in_).index(feature))


def options_of(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> GraphImageOptions:
    return GraphImageOptions(score_axis=model_kind(models[kind]).score_axis)


def texts_of(svg: str) -> list[str]:
    return [html.unescape(text) for text in re.findall(r">([^<]+)</text>", svg)]


def steps_of(figure: Figure) -> list[StepPatch]:
    (axes,) = figure.axes
    return [patch for patch in axes.patches if isinstance(patch, StepPatch)]


def lines_of(figure: Figure) -> list[StepPatch]:
    # each series draws its band first, then the step line of its means
    return steps_of(figure)[1::2]


def bands_of(figure: Figure) -> list[StepPatch]:
    return steps_of(figure)[::2]


def values_of(patch: StepPatch) -> npt.NDArray[np.float64]:
    values, _, _ = patch.get_data()
    return np.asarray(values, dtype=float)


def edges_of(patch: StepPatch) -> list[float]:
    _, edges, _ = patch.get_data()
    edge_values: list[float] = np.asarray(edges, dtype=float).tolist()
    return edge_values


def baseline_of(patch: StepPatch) -> npt.NDArray[np.float64]:
    _, _, baseline = patch.get_data()
    return np.asarray(baseline, dtype=float)


def cont_graph() -> EBMGraph:
    return EBMGraph(
        "age",
        "continuous",
        [(18.0, 30.0), (30.0, 60.0), (60.0, 90.0)],
        [-0.5, 0.25, 0.1],
        [0.1, 0.05, 0.2],
    )


def cat_graph() -> EBMGraph:
    return EBMGraph("color", "nominal", ["red", "blue"], [0.3, -0.3], [0.1, 0.2])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("feature", [CONTINUOUS, CATEGORICAL])
def test_each_model_kind_and_feature_type_renders_as_png_and_svg(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
    feature: str,
) -> None:
    graph = graph_of(models, kind, feature)
    assert graph_png(graph, options_of(models, kind)).startswith(PNG_SIGNATURE)
    assert graph_svg(graph, options_of(models, kind)).rstrip().endswith("</svg>")


@pytest.mark.parametrize("kind", KINDS)
def test_the_steps_and_the_band_of_a_continuous_graph_equal_its_scores_bounds_and_bins(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> None:
    graph = graph_of(models, kind, CONTINUOUS)
    factor = confidence_factor(0.95)
    edges = [low for low, _ in graph.x_vals] + [graph.x_vals[-1][1]]
    figure = graph_figure(graph, options_of(models, kind))
    lines, bands = lines_of(figure), bands_of(figure)
    for series, band, line in zip(graph.series(), bands, lines, strict=True):
        assert np.array_equal(values_of(line), series.scores)
        assert edges_of(line) == edges
        assert np.array_equal(values_of(band), series.scores + factor * series.stds)
        assert np.array_equal(baseline_of(band), series.scores - factor * series.stds)


@pytest.mark.parametrize("kind", KINDS)
def test_the_bars_and_the_error_bars_of_a_categorical_graph_equal_its_scores_and_bounds(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> None:
    graph = graph_of(models, kind, CATEGORICAL)
    factor = confidence_factor(0.95)
    (axes,) = graph_figure(graph, options_of(models, kind)).axes
    containers = [container for container in axes.containers if isinstance(container, BarContainer)]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == [str(category) for category in graph.x_vals]
    for series, container in zip(graph.series(), containers, strict=True):
        assert np.array_equal(np.asarray(container.datavalues, dtype=float), series.scores)
        assert container.errorbar is not None
        segments = np.asarray(container.errorbar.lines[2][0].get_segments(), dtype=float)
        assert np.allclose(segments[:, 0, 1], series.scores - factor * series.stds)
        assert np.allclose(segments[:, 1, 1], series.scores + factor * series.stds)


@pytest.mark.parametrize("kind", KINDS)
def test_each_class_of_the_graph_is_one_series_of_the_image(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> None:
    graph = graph_of(models, kind, CONTINUOUS)
    labels = [line.get_label() for line in lines_of(graph_figure(graph, options_of(models, kind)))]
    assert labels == (list(graph.classes) if graph.classes else [labels[0]])
    assert len(labels) == len(graph.series())


@pytest.mark.parametrize("kind", KINDS)
def test_the_drawn_means_equal_the_means_that_the_graph_text_writes(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> None:
    graph = graph_of(models, kind, CONTINUOUS)
    read_back = EBMGraph.from_text(graph.to_text(y_axis_precision=6))
    (line, *_) = lines_of(graph_figure(graph, options_of(models, kind)))
    drawn = np.column_stack([values_of(line) for line in lines_of(graph_figure(graph, options_of(models, kind)))])
    # the text merges neighbouring bins with equal scores, so look up the text bin that holds each drawn bin
    text_lows = np.asarray([low for low, _ in read_back.x_vals], dtype=float)
    holding_bin = np.searchsorted(text_lows, edges_of(line)[:-1], side="right") - 1
    text_scores = np.asarray(read_back.scores, dtype=float).reshape(len(text_lows), -1)[holding_bin]
    assert np.allclose(drawn, text_scores, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_the_labels_of_the_svg_are_the_names_of_the_graph(
    models: dict[str, ExplainableBoostingClassifier | ExplainableBoostingRegressor],
    kind: str,
) -> None:
    graph = graph_of(models, kind, CATEGORICAL)
    texts = texts_of(graph_svg(graph, options_of(models, kind)))
    assert f"Graph of {CATEGORICAL}" in texts
    assert CATEGORICAL in texts
    assert model_kind(models[kind]).score_axis in texts
    assert {str(category) for category in graph.x_vals} <= set(texts)
    assert set(graph.classes) <= set(texts)
    assert "error bars: 95% confidence interval" in texts


def test_a_title_replaces_the_default_title() -> None:
    texts = texts_of(graph_svg(cont_graph(), GraphImageOptions(title="Age and risk")))
    assert "Age and risk" in texts
    assert "Graph of age" not in texts


def test_a_name_with_dollar_signs_and_markup_stays_text() -> None:
    graph = EBMGraph("$income$ <&>", "continuous", [(0.0, 1.0), (1.0, 2.0)], [0.1, 0.2], [0.0, 0.0])
    assert "$income$ <&>" in texts_of(graph_svg(graph))


def test_the_same_graph_gives_the_same_bytes_with_any_matplotlib_settings() -> None:
    png, svg = graph_png(cont_graph()), graph_svg(cont_graph())
    with mpl.rc_context({"lines.linewidth": 7.0, "font.size": 20.0, "svg.fonttype": "path"}):
        assert graph_png(cont_graph()) == png
        assert graph_svg(cont_graph()) == svg


def draw_continuous_graph(_job: int) -> bytes:
    return graph_png(cont_graph())


def test_drawings_in_parallel_threads_give_the_same_bytes() -> None:
    expected = graph_png(cont_graph())
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(draw_continuous_graph, range(8)))
    assert images == [expected] * 8


def test_the_png_has_640_by_360_pixels() -> None:
    png = graph_png(cat_graph())
    width, height = int.from_bytes(png[16:20]), int.from_bytes(png[20:24])
    assert (width, height) == (640, 360)


def test_a_graph_with_no_bounds_draws_no_band_and_no_caption() -> None:
    options = GraphImageOptions(confidence_level=None)
    (band,) = bands_of(graph_figure(cont_graph(), options))
    assert np.array_equal(values_of(band), baseline_of(band))
    assert not [text for text in texts_of(graph_svg(cont_graph(), options)) if "confidence" in text]


def test_the_caption_names_the_confidence_level() -> None:
    texts = texts_of(graph_svg(cont_graph(), GraphImageOptions(confidence_level=0.9)))
    assert "band: 90% confidence interval" in texts


def test_an_infinite_end_is_drawn_one_median_bin_width_beyond_the_next_edge() -> None:
    x_vals = [(-INF, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, INF)]
    graph = EBMGraph("x", "continuous", x_vals, [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    (line,) = lines_of(graph_figure(graph, GraphImageOptions()))
    assert edges_of(line) == [-0.5, 1.0, 2.0, 4.0, 5.5]


def test_one_bin_with_two_infinite_ends_is_drawn_one_unit_wide() -> None:
    graph = EBMGraph("x", "continuous", [(-INF, INF)], [1.0], [0.0])
    (line,) = lines_of(graph_figure(graph, GraphImageOptions()))
    assert edges_of(line) == [0.0, 1.0]


def test_bins_that_do_not_touch_are_rejected() -> None:
    graph = EBMGraph("x", "continuous", [(0.0, 1.0), (2.0, 3.0)], [0.0, 1.0], [0.0, 0.0])
    with pytest.raises(EBMGraphShapeError, match="bins that touch"):
        graph_png(graph)


def test_a_boolean_graph_has_the_keys_that_the_graph_text_writes() -> None:
    graph = EBMGraph("member", "nominal", ["no", "yes"], [0.1, -0.1], [0.0, 0.0])
    (axes,) = graph_figure(graph, GraphImageOptions(feature_format="boolean")).axes
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["False", "True"]


def test_a_boolean_format_of_three_bins_is_rejected_as_the_text_rejects_it() -> None:
    graph = EBMGraph("member", "nominal", ["a", "b", "c"], [0.1, 0.0, -0.1], [0.0, 0.0, 0.0])
    with pytest.raises(EBMGraphShapeError, match="exactly two"):
        graph_png(graph, GraphImageOptions(feature_format="boolean"))


def test_a_graph_with_more_classes_than_colors_uses_the_colors_again() -> None:
    classes = [f"class {index}" for index in range(11)]
    graph = EBMGraph("x", "continuous", [(0.0, 1.0)], [[float(index) for index in range(11)]], [[0.0] * 11], classes)
    lines = lines_of(graph_figure(graph, GraphImageOptions()))
    assert lines[10].get_edgecolor() == lines[0].get_edgecolor()
    assert lines[1].get_edgecolor() != lines[0].get_edgecolor()


def test_the_image_functions_load_from_ebaml_on_first_use() -> None:
    assert ebaml.graph_png is graph_png
    assert ebaml.graph_svg is graph_svg
    assert ebaml.graph_figure is graph_figure
    assert ebaml.GraphImageOptions is GraphImageOptions


def test_an_import_without_matplotlib_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, graph_image.__name__)
    with pytest.raises(MissingExtraError, match=re.escape('pip install "ebaml[vision]"')):
        importlib.import_module(graph_image.__name__)
