"""The conversation about one graph: its text, the three turns of its description, and the retry of a cut-off reply.

``describe_graph`` runs one such conversation, and ``describe_ebm`` runs one for each feature. Like
``ebaml.llm.pipelines``, the module imports no ``baml_sdk``: the conversation yields the steps of
``ebaml.llm.sdk_steps``, and a driver runs them.
"""

import functools
from collections.abc import Generator, Sequence

import attrs
from baml_bridge.errors import BamlError

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EBMGraphSimplificationError, VisionUnsupportedError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.token_budget import TokenEstimator, check_token_budget, estimate_tokens
from ebaml.graphs.token_weights import ModelKey
from ebaml.llm.graph_views import GraphView, ImageSettings, RenderedGraph, VisionSwitch, graph_view, shrinks_with_text
from ebaml.llm.options import GraphOptions
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.retry import ShrinkingGraphCall, call_with_shrinking_graph
from ebaml.llm.sdk_steps import CountedText, Described, ModelIdentity, SDKCall, SDKValue, Step


def _estimator(options: GraphOptions, model: ModelIdentity | None) -> TokenEstimator | None:
    """Get the token estimator of the learner in the options, for the model that the call uses.

    Parameters
    ----------
    options: GraphOptions
        The options of the graph. Its ``token_learner`` can be ``None``.
    model: ModelIdentity | None
        The model of the requests. ``None`` selects the default clients, which the learner does not know by name.

    Returns
    -------
    TokenEstimator | None
        The estimator of the model, or ``None`` when the options have no learner.

    """
    if options.token_learner is None:
        return None
    return options.token_learner.estimator_for(None if model is None else ModelKey(model.provider, model.model))


def render_graph(
    ebm: ExplainableBoostingModel,
    feature_index: int,
    options: GraphOptions,
    model: ModelIdentity | None,
) -> RenderedGraph:
    """Extract an EBM graph and convert it to text, with the token budget check.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM model.
    feature_index: int
        The index of the feature to convert to text.
    options: GraphOptions
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.
    model: ModelIdentity | None
        The model of the requests. The token estimator of the learner in ``options`` depends on it.

    Returns
    -------
    RenderedGraph
        The extracted graph, the simplified graph that the text renders, and the text.

    """
    graph = EBMGraph.from_ebm(
        ebm,
        feature_index,
        options.normalization,
        use_feature_bounds=options.use_feature_bounds,
        classes=options.classes,
    )
    shown: list[EBMGraph] = []
    text = graph.to_text(
        include_description=options.include_description,
        feature_format=options.feature_format,
        x_axis_precision=options.x_axis_precision,
        y_axis_precision=options.y_axis_precision,
        confidence_bounds=options.confidence_bounds,
        confidence_level=options.confidence_level,
        max_tokens=options.max_tokens,
        overbudget_percent=options.overbudget_percent,
        check_budget=_estimator(options, model) or check_token_budget,
        on_render=shown.append,
    )
    (shown_graph,) = shown
    return RenderedGraph(extracted=graph, shown=shown_graph, text=text)


def is_truncated(error: BamlError) -> bool:
    """Tell whether an LLM call failed because the reply hit the ``max_tokens`` limit.

    Parameters
    ----------
    error: BamlError
        The error raised by a ``baml_sdk`` call.

    Returns
    -------
    bool
        True for a ``FinishReasonError`` whose finish reason is ``MaxTokens``.

    """
    value = error.value
    return error.class_name == "ai.errors.FinishReasonError" and getattr(value, "finish_reason", None) == "MaxTokens"


@attrs.frozen(kw_only=True)
class GraphJob:
    """The settings that every graph of one call shares, and the graph mode that its requests share."""

    ebm: ExplainableBoostingModel
    num_sentences: int
    graph_options: GraphOptions
    prompt_options: SDKValue
    model: SDKValue
    switch: VisionSwitch
    image: ImageSettings
    omitted_terms: Sequence[str] = ()


def _ask_for_view(job: GraphJob, graph: RenderedGraph, sent: list[GraphView]) -> Generator[Step, SDKValue, CountedText]:
    """Send the three turns of the description of one graph in the mode of the call.

    If the model rejects the image, the switch of the call raises ``VisionUnsupportedError``, or turns the call to text
    mode, and the function asks again with the text.

    Parameters
    ----------
    job: GraphJob
        The settings and the switch of the call.
    graph: RenderedGraph
        The graph and its text.
    sent: list[GraphView]
        The views of the requests that succeeded. The function appends the view that it sent.

    Returns
    -------
    CountedText
        The description and its token usage.

    Yields
    ------
    Step
        The call of ``describe_graph_cot``, and after a rejected image the call with the text.

    Raises
    ------
    VisionUnsupportedError
        If the model rejects the image and the fallback is ``VisionFallback.RAISE``, or if a request with no image
        fails with this error.

    """
    view = graph_view(job.switch.next_mode(), graph, job.image)
    try:
        with provider_errors(job.switch.model):
            counted: CountedText = yield SDKCall(
                "describe_graph_cot",
                {
                    "graph": view.text,
                    "num_sentences": job.num_sentences,
                    "opts": job.prompt_options,
                    "model": job.model,
                    "omitted_terms": list(job.omitted_terms),
                    "png_base64": view.png_base64,
                },
            )
    except VisionUnsupportedError as error:
        if view.png_base64 is None:
            # a request with no image cannot fail for its image, so the cause is elsewhere and text mode cannot help
            raise
        job.switch.fall_back(error)
        return (yield from _ask_for_view(job, graph, sent))
    sent.append(view)
    return counted


def _rebuild(job: GraphJob, feature_index: int, budget: int) -> RenderedGraph:
    """Render a graph again for a smaller token budget, after a reply that ``max_tokens`` cut off.

    Parameters
    ----------
    job: GraphJob
        The settings and the switch of the call.
    feature_index: int
        The index of the feature.
    budget: int
        The token budget of the text.

    Returns
    -------
    RenderedGraph
        The graph with a shorter text.

    Raises
    ------
    EBMGraphSimplificationError
        If the mode of the call sends no text. A smaller text would not change the request, so the retry stops.

    """
    if not shrinks_with_text(job.switch.mode):
        msg = f'A request in "{job.switch.mode}" mode sends no graph text, so a smaller text cannot shorten it.'
        raise EBMGraphSimplificationError(msg)
    options = attrs.evolve(job.graph_options, max_tokens=budget, overbudget_percent=0)
    return render_graph(job.ebm, feature_index, options, job.model)


def describe_graph_counted(
    job: GraphJob,
    feature_index: int,
    graph: RenderedGraph,
) -> Generator[Step, SDKValue, Described]:
    """Describe one graph with the LLM, simplifying the graph further whenever the reply is cut off by ``max_tokens``.

    A failure that the user can fix raises an ``LLMRequestError`` (see ``ebaml.llm.provider_errors``). Any other failure
    raises the ``BamlError`` of the LLM call. A reply cut off by ``max_tokens`` is retried first, as
    ``ebaml.llm.retry.call_with_shrinking_graph`` says. A model that rejects the image raises
    ``VisionUnsupportedError``, or the call turns to text mode, as ``GraphOptions.vision_fallback`` says.

    Parameters
    ----------
    job: GraphJob
        The EBM, the number of sentences, the options of the graph text (a retry changes ``max_tokens`` and
        ``overbudget_percent``), the prompt options, the model of the requests (``None`` selects the default clients),
        the switch of the graph mode, the settings of the images, and the interaction terms that the description must
        name as left out.
    feature_index: int
        The index of the feature to describe.
    graph: RenderedGraph
        The graph and its text, as made by ``render_graph``.

    Returns
    -------
    Described
        The description with its token usage, and the number of attempts that were cut off. The usage counts the
        attempt that succeeded only.

    Yields
    ------
    Step
        The calls of ``describe_graph_cot``.

    """
    truncated: list[BamlError] = []
    graph_options = job.graph_options
    estimator = _estimator(graph_options, job.model)
    estimate = estimate_tokens if estimator is None else estimator.estimate
    sent: list[GraphView] = []
    counted = yield from call_with_shrinking_graph(
        ShrinkingGraphCall(
            call=functools.partial(_ask_for_view, job, sent=sent),
            graph=graph,
            error_type=BamlError,
            is_truncated=is_truncated,
            estimate_tokens=lambda rendered: estimate(rendered.text),
            rebuild=lambda budget: _rebuild(job, feature_index, budget),
            on_truncated=truncated.append,
            label=job.ebm.feature_names_in_[feature_index],
        ),
    )
    # the last view is the one of the successful attempt; a request with an image teaches the text weights nothing
    last = sent[-1]
    if graph_options.token_learner is not None and last.png_base64 is None and last.text is not None:
        graph_options.token_learner.observe(counted.usage.samples, last.text)
    return Described(counted, len(truncated))
