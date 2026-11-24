"""Choose what a request sends for a graph in each mode, and switch a call to text mode when the model reads no images.

``GraphMode.TEXT`` sends the graph text, ``GraphMode.VISION`` the image of the extracted graph, and ``GraphMode.MIXED``
the text with the image of the graph that the text renders. ``VisionSwitch`` holds the mode of one call. The first
model that reads no images decides for the whole call: the switch raises ``VisionUnsupportedError``, or turns the call
to text mode and emits one ``VisionFallbackWarning``, as ``VisionFallback`` says.
"""

import base64
import importlib
import threading
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Self

import attrs
import structlog

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureType, GraphMode, VisionFallback, coerce_enum
from ebaml.core.exceptions import VisionFallbackWarning, VisionUnsupportedError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.model_kinds import model_kind
from ebaml.llm.options import DEFAULT_MODEL, GraphOptions, ModelOptions

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

# the directory of the package; the warning skips its frames
_PACKAGE_DIRECTORY = str(Path(__file__).resolve().parents[1])
# the module that draws the images; it needs the extra `vision`, so a call in text mode never imports it
_IMAGE_MODULE = "ebaml.graphs.graph_image"


@attrs.frozen
class GraphView:
    """What one request sends for a graph: the text, the PNG image in base64, or both.

    Attributes
    ----------
    text: str | None
        The graph text, or ``None`` in vision mode.
    png_base64: str | None
        The PNG image of the graph in base64, or ``None`` in text mode.

    """

    text: str | None
    png_base64: str | None


@attrs.frozen
class RenderedGraph:
    """A graph as the EBM gives it, and the simplified graph with its text.

    Attributes
    ----------
    extracted: EBMGraph
        The graph from ``EBMGraph.from_ebm``, with no simplification. Vision mode draws it.
    shown: EBMGraph
        The graph that ``text`` renders, after the simplification for the token budget. Mixed mode draws it.
    text: str
        The graph text.

    """

    extracted: EBMGraph
    shown: EBMGraph
    text: str


@attrs.frozen
class ImageSettings:
    """The labels and the bounds of the images of one call.

    Attributes
    ----------
    ebm: ExplainableBoostingModel
        The EBM. Its kind names the score axis.
    feature_format: FeatureType | None
        The feature format of the graph text, so that the image presents the feature in the same way.
    confidence_level: float | None
        The level of the bounds, or ``None`` when the graph text has no bounds.

    """

    ebm: ExplainableBoostingModel
    feature_format: FeatureType | None
    confidence_level: float | None

    @classmethod
    def from_options(cls, ebm: ExplainableBoostingModel, options: GraphOptions) -> Self:
        """Take the settings from the options of the graph text, so that the image presents what the text presents.

        Parameters
        ----------
        ebm: ExplainableBoostingModel
            The EBM.
        options: GraphOptions
            The options of the graph text.

        Returns
        -------
        ImageSettings
            The settings: the bounds only if the text has bounds.

        """
        feature_format = None if options.feature_format is None else coerce_enum(FeatureType, options.feature_format)
        return cls(ebm, feature_format, options.confidence_level if options.confidence_bounds else None)


def image_module() -> ModuleType:
    """Import the module that draws the graph images.

    The import raises ``MissingExtraError`` if the extra ``vision`` is not installed. A call in vision or mixed mode
    imports the module before its first request, so the error comes before any cost.

    Returns
    -------
    ModuleType
        ``ebaml.graphs.graph_image``.

    """
    return importlib.import_module(_IMAGE_MODULE)


def _png_base64(graph: EBMGraph, settings: ImageSettings) -> str:
    """Draw a graph as a PNG image and encode it in base64.

    Parameters
    ----------
    graph: EBMGraph
        The graph.
    settings: ImageSettings
        The labels and the bounds.

    Returns
    -------
    str
        The PNG image in base64.

    """
    drawing = image_module()
    options = drawing.GraphImageOptions(
        score_axis=model_kind(settings.ebm).score_axis,
        feature_format=settings.feature_format,
        confidence_level=settings.confidence_level,
    )
    png: bytes = drawing.graph_png(graph, options)
    return base64.b64encode(png).decode("ascii")


def _text_view(rendered: RenderedGraph, _settings: ImageSettings) -> GraphView:
    return GraphView(text=rendered.text, png_base64=None)


def _vision_view(rendered: RenderedGraph, settings: ImageSettings) -> GraphView:
    return GraphView(text=None, png_base64=_png_base64(rendered.extracted, settings))


def _mixed_view(rendered: RenderedGraph, settings: ImageSettings) -> GraphView:
    return GraphView(text=rendered.text, png_base64=_png_base64(rendered.shown, settings))


# the view of each mode; a new mode is a new function and a new row
_VIEWS: Mapping[GraphMode, Callable[[RenderedGraph, ImageSettings], GraphView]] = {
    GraphMode.TEXT: _text_view,
    GraphMode.VISION: _vision_view,
    GraphMode.MIXED: _mixed_view,
}


def graph_view(mode: GraphMode, rendered: RenderedGraph, settings: ImageSettings) -> GraphView:
    """Choose what a request sends for a graph in a mode.

    Parameters
    ----------
    mode: GraphMode
        The mode of the request.
    rendered: RenderedGraph
        The extracted graph, and the simplified graph with its text.
    settings: ImageSettings
        The labels and the bounds of an image.

    Returns
    -------
    GraphView
        The text, the image, or both. The image of mixed mode shows the graph that the text renders, so the two agree
        bin for bin.

    """
    return _VIEWS[mode](rendered, settings)


# the modes whose request sends the graph text, so that a shorter text shortens the request
_SENDS_TEXT = frozenset({GraphMode.TEXT, GraphMode.MIXED})


def shrinks_with_text(mode: GraphMode) -> bool:
    """Tell whether a shorter graph text makes a request of a mode shorter.

    Parameters
    ----------
    mode: GraphMode
        The mode of the request.

    Returns
    -------
    bool
        True for the modes that send the graph text. A request in vision mode sends the image only.

    """
    return mode in _SENDS_TEXT


def _fix_steps(model: str) -> str:
    """List the steps that fix a call in vision or mixed mode to a model that reads no images.

    Parameters
    ----------
    model: str
        The model, as ``provider/model``.

    Returns
    -------
    str
        The steps, one numbered line each.

    """
    return (
        "To fix it:\n"
        f"  1. Pick a model that reads images in place of {model}. The provider lists the input of each model.\n"
        "  2. Pass graph=GraphOptions(mode=GraphMode.TEXT) to send each graph as text.\n"
        "  3. Pass graph=GraphOptions(vision_fallback=VisionFallback.TEXT) to send the graphs as text when the model"
        " reads no images, with a VisionFallbackWarning."
    )


def rejected_image_error(provider: str, model: str, said: str) -> VisionUnsupportedError:
    """Explain a request whose image the provider rejected.

    Parameters
    ----------
    provider: str
        The provider that rejected the request.
    model: str
        The model of the request, as ``provider/model``.
    said: str
        A line that quotes the provider, or an empty string.

    Returns
    -------
    VisionUnsupportedError
        The error, with the steps that fix it.

    """
    return VisionUnsupportedError(
        f"The {provider} provider rejected the image of the graph: the model {model} reads no images.\n"
        f"{said}{_fix_steps(model)}",
    )


@attrs.define
class VisionSwitch:
    """The graph mode of one call, which the first model that reads no images can turn to text mode.

    The threads that describe the graphs of one call share the switch. Each request reads the mode when it starts. So
    after the first rejection, the requests that start later send text, and the call emits one warning.

    Attributes
    ----------
    mode: GraphMode
        The mode of the next request.
    fallback: VisionFallback
        What the switch does when the model reads no images.
    model: str
        The model of the call, as ``provider/model``, for the messages.

    """

    mode: GraphMode
    fallback: VisionFallback
    model: str
    _lock: threading.Lock = attrs.field(factory=threading.Lock, init=False, repr=False)
    _raised: VisionUnsupportedError | None = attrs.field(default=None, init=False, repr=False)

    def next_mode(self) -> GraphMode:
        """Give the mode of the next request of the call.

        Returns
        -------
        GraphMode
            The mode. After a rejection with ``VisionFallback.RAISE``, the method raises that rejection again, so that
            the graphs that wait in the thread pool send no request.

        """
        if self._raised is not None:
            raise self._raised
        return self.mode

    @classmethod
    def for_call(cls, options: GraphOptions, model: ModelOptions | str | None) -> Self:
        """Start the switch of a call from its options.

        Parameters
        ----------
        options: GraphOptions
            The mode and the fallback.
        model: ModelOptions | str | None
            The model of the call. Its ``vision`` field can say that the model reads no images. ``None`` selects the
            default clients.

        Returns
        -------
        VisionSwitch
            The switch that the requests of the call share.

        """
        if model is None:
            return cls.start(options.mode, options.vision_fallback, DEFAULT_MODEL, vision=None)
        parsed = ModelOptions.parse(model)
        label = f"{parsed.provider}/{parsed.model}"
        return cls.start(options.mode, options.vision_fallback, label, vision=parsed.vision)

    @classmethod
    def start(cls, mode: GraphMode, fallback: VisionFallback, model: str, *, vision: bool | None) -> Self:
        """Start the switch of a call, and apply what the options say before any request.

        Parameters
        ----------
        mode: GraphMode
            The mode that the caller asks for.
        fallback: VisionFallback
            What the switch does when the model reads no images.
        model: str
            The model of the call, as ``provider/model``.
        vision: bool | None
            ``ModelOptions.vision``. ``False`` applies the fallback at once.

        Returns
        -------
        VisionSwitch
            The switch. If it stays in vision or mixed mode, the function imports the drawing module, so that a missing
            extra ``vision`` raises ``MissingExtraError`` before any request.

        """
        switch = cls(mode, fallback, model)
        switch.check_declared(vision=vision)
        if switch.mode is not GraphMode.TEXT:
            image_module()
        return switch

    def check_declared(self, *, vision: bool | None) -> None:
        """Apply the fallback before any request if the options say that the model reads no images.

        Parameters
        ----------
        vision: bool | None
            ``ModelOptions.vision``. Only ``False`` has an effect, and only in vision or mixed mode.

        """
        if vision is False and self.mode is not GraphMode.TEXT:
            self.fall_back(
                VisionUnsupportedError(
                    f"ModelOptions(vision=False) says that the model {self.model} reads no images, and the graph mode"
                    f' is "{self.mode}".\n{_fix_steps(self.model)}',
                ),
            )

    def fall_back(self, error: VisionUnsupportedError) -> None:
        """Turn the call to text mode, or raise the error, as the fallback says.

        With ``VisionFallback.RAISE``, the method raises ``error``, and ``next_mode`` raises it too. With
        ``VisionFallback.TEXT``, the first call of the method turns the switch to text mode and emits the warning, and
        a later call does nothing.

        Parameters
        ----------
        error: VisionUnsupportedError
            Why the model cannot take the image.

        """
        if self.fallback is VisionFallback.RAISE:
            self._raised = error
            raise error
        with self._lock:
            if self.mode is GraphMode.TEXT:
                return
            requested, self.mode = self.mode, GraphMode.TEXT
        logger.warning("vision_fallback", model=self.model, requested_mode=str(requested), reason=str(error))
        warnings.warn(
            VisionFallbackWarning(
                f'The model {self.model} reads no images, so the call sends the graphs as text and not in "{requested}"'
                f" mode. The cause: {error}",
            ),
            stacklevel=2,  # pragma: no mutate -- skip_file_prefixes sets the stack level itself
            # skip the files of ebaml and the frames that beartype adds, so the warning points at the caller
            skip_file_prefixes=(_PACKAGE_DIRECTORY, "<@beartype"),
        )
