"""Tests for the view of a graph in each mode and for the switch that turns a call to text mode."""

import base64
import sys
import warnings

import pytest
from structlog.testing import capture_logs

from ebaml.core.custom_types import GraphMode, VisionFallback
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import MissingExtraError, VisionFallbackWarning, VisionUnsupportedError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.graph_image import GraphImageOptions, graph_png
from ebaml.llm.graph_views import (
    ImageSettings,
    RenderedGraph,
    VisionSwitch,
    graph_view,
    rejected_image_error,
    shrinks_with_text,
)
from ebaml.llm.options import GraphOptions
from tests.unit.ebaml.conftest import PandasFrame

MODEL = "openrouter/text-only"
EXTRACTED: EBMGraph = EBMGraph(
    "age",
    "continuous",
    [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    [0.1, 0.1, 0.5],
    [0.01, 0.02, 0.03],
)
SHOWN: EBMGraph = EBMGraph("age", "continuous", [(0.0, 2.0), (2.0, 3.0)], [0.1, 0.5], [0.02, 0.03])
RENDERED: RenderedGraph = RenderedGraph(extracted=EXTRACTED, shown=SHOWN, text="GRAPH TEXT")
SCORE_AXIS = "change of the log-odds of 1"


@pytest.fixture
def settings(continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame]) -> ImageSettings:
    return ImageSettings(continuous_ebm[0], None, 0.95)


def png_of(view_png: str | None) -> bytes:
    assert view_png is not None
    return base64.b64decode(view_png)


def test_text_mode_sends_the_text_only(settings: ImageSettings) -> None:
    view = graph_view(GraphMode.TEXT, RENDERED, settings)
    assert (view.text, view.png_base64) == ("GRAPH TEXT", None)


def test_vision_mode_sends_the_image_of_the_extracted_graph_only(settings: ImageSettings) -> None:
    view = graph_view(GraphMode.VISION, RENDERED, settings)
    assert view.text is None
    assert png_of(view.png_base64) == graph_png(EXTRACTED, GraphImageOptions(score_axis=SCORE_AXIS))


def test_mixed_mode_sends_the_text_and_the_image_of_the_graph_that_the_text_renders(settings: ImageSettings) -> None:
    view = graph_view(GraphMode.MIXED, RENDERED, settings)
    assert view.text == "GRAPH TEXT"
    assert png_of(view.png_base64) == graph_png(SHOWN, GraphImageOptions(score_axis=SCORE_AXIS))


def test_the_settings_take_the_format_and_the_bounds_of_the_text(
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    ebm = continuous_ebm[0]
    with_bounds = ImageSettings.from_options(ebm, GraphOptions(feature_format="continuous", confidence_level=0.9))
    without = ImageSettings.from_options(ebm, GraphOptions(confidence_bounds=False))
    assert (with_bounds.feature_format, with_bounds.confidence_level) == ("continuous", 0.9)
    assert without.confidence_level is None


def test_the_image_draws_the_bounds_of_the_settings(
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    view = graph_view(GraphMode.VISION, RENDERED, ImageSettings(continuous_ebm[0], None, None))
    assert png_of(view.png_base64) == graph_png(
        EXTRACTED,
        GraphImageOptions(score_axis=SCORE_AXIS, confidence_level=None),
    )


@pytest.mark.parametrize("mode", [GraphMode.TEXT, GraphMode.MIXED])
def test_a_shorter_text_shortens_the_requests_that_send_text(mode: GraphMode) -> None:
    assert shrinks_with_text(mode)


def test_a_shorter_text_does_not_shorten_a_request_in_vision_mode() -> None:
    assert not shrinks_with_text(GraphMode.VISION)


def test_the_error_of_a_rejected_image_names_the_model_quotes_the_provider_and_lists_the_fixes() -> None:
    error = rejected_image_error("openrouter", MODEL, "The provider said: no images\n")
    assert str(error) == (
        f"The openrouter provider rejected the image of the graph: the model {MODEL} reads no images.\n"
        "The provider said: no images\n"
        "To fix it:\n"
        f"  1. Pick a model that reads images in place of {MODEL}. The provider lists the input of each model.\n"
        '  2. Pass graph=GraphOptions(mode="text") to send each graph as text.\n'
        '  3. Pass graph=GraphOptions(vision_fallback="text") to send the graphs as text when the model reads no'
        " images, with a VisionFallbackWarning."
    )


def test_the_raise_fallback_raises_the_error_and_every_later_request_raises_it_too() -> None:
    switch = VisionSwitch(GraphMode.VISION, VisionFallback.RAISE, MODEL)
    error = VisionUnsupportedError("no images")
    with pytest.raises(VisionUnsupportedError) as raised:
        switch.fall_back(error)
    assert raised.value is error
    with pytest.raises(VisionUnsupportedError) as again:
        switch.next_mode()
    assert again.value is error


def fall_back_twice(switch: VisionSwitch) -> None:
    switch.fall_back(VisionUnsupportedError("first"))
    switch.fall_back(VisionUnsupportedError("second"))


def start_without_images_and_fall_back() -> None:
    switch = VisionSwitch.start(GraphMode.VISION, VisionFallback.TEXT, MODEL, vision=False)
    switch.fall_back(VisionUnsupportedError("later rejection"))


def test_the_text_fallback_turns_the_switch_to_text_with_one_warning_and_one_log_event() -> None:
    switch = VisionSwitch(GraphMode.MIXED, VisionFallback.TEXT, MODEL)
    with capture_logs() as logs, pytest.warns(VisionFallbackWarning, match='not in "mixed" mode') as warned:
        fall_back_twice(switch)
    assert switch.next_mode() is GraphMode.TEXT
    assert len(warned) == 1
    assert "The cause: first" in str(warned[0].message)
    assert [(log["event"], log["model"], log["requested_mode"]) for log in logs] == [
        ("vision_fallback", MODEL, "mixed"),
    ]


def test_the_warning_points_at_the_code_that_called_ebaml() -> None:
    switch = VisionSwitch(GraphMode.VISION, VisionFallback.TEXT, MODEL)
    with pytest.warns(VisionFallbackWarning) as warned:
        switch.fall_back(VisionUnsupportedError("no images"))
    assert warned[0].filename == __file__


def test_the_warning_can_become_an_error() -> None:
    switch = VisionSwitch(GraphMode.VISION, VisionFallback.TEXT, MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisionFallbackWarning)
        with pytest.raises(VisionFallbackWarning):
            switch.fall_back(VisionUnsupportedError("no images"))


@pytest.mark.parametrize("mode", [GraphMode.VISION, GraphMode.MIXED])
def test_a_model_that_reads_no_images_by_its_options_raises_at_the_start(mode: GraphMode) -> None:
    with pytest.raises(
        VisionUnsupportedError,
        match=rf'ModelOptions\(vision=False\) says that the model {MODEL}.*"{mode}"',
    ):
        VisionSwitch.start(mode, VisionFallback.RAISE, MODEL, vision=False)


def test_a_model_that_reads_no_images_by_its_options_starts_in_text_mode_with_the_text_fallback() -> None:
    with pytest.warns(VisionFallbackWarning):
        switch = VisionSwitch.start(GraphMode.VISION, VisionFallback.TEXT, MODEL, vision=False)
    assert switch.next_mode() is GraphMode.TEXT


@pytest.mark.parametrize("declared", [{"vision": True}, {"vision": None}, {"vision": False}])
def test_text_mode_starts_in_text_mode_whatever_the_model_reads(declared: dict[str, bool | None]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        switch = VisionSwitch.start(GraphMode.TEXT, VisionFallback.RAISE, MODEL, **declared)
    assert switch.next_mode() is GraphMode.TEXT


@pytest.mark.parametrize("declared", [{"vision": True}, {"vision": None}])
def test_a_model_that_may_read_images_keeps_the_mode(declared: dict[str, bool | None]) -> None:
    assert VisionSwitch.start(GraphMode.MIXED, VisionFallback.RAISE, MODEL, **declared).next_mode() is GraphMode.MIXED


def test_the_start_of_a_call_with_images_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "ebaml.graphs.graph_image")
    with pytest.raises(MissingExtraError):
        VisionSwitch.start(GraphMode.VISION, VisionFallback.RAISE, MODEL, vision=None)


def test_the_start_of_a_call_in_text_mode_needs_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "ebaml.graphs.graph_image")
    assert VisionSwitch.start(GraphMode.TEXT, VisionFallback.RAISE, MODEL, vision=None).next_mode() is GraphMode.TEXT


def test_a_switch_that_falls_back_to_text_after_a_model_without_images_by_its_options_warns_once() -> None:
    with pytest.warns(VisionFallbackWarning) as warned:
        start_without_images_and_fall_back()
    assert len(warned) == 1
