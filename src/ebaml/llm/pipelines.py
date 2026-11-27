"""The pipelines of the LLM calls: one implementation for the sync and the async forms.

Each pipeline is a generator that yields the steps of ``ebaml.llm.sdk_steps``. A driver runs each step, then sends
the reply into the generator, or throws the error of the step into it. ``ebaml.llm.descriptors`` holds the sync
driver, and ``ebaml.llm.async_descriptors`` holds the async driver. The graph rendering, the shrinking retry, the
vision fallback, the provider errors and the facts live here once, so a fix reaches both forms.

The module imports no ``baml_sdk``. The classes of ``baml_sdk`` that a pipeline builds reach it in ``SDKTypes``, and
the replies reach it through ``send``. So a unit test drives a pipeline with plain replies.
"""

import functools
from collections.abc import Generator, Sequence

import attrs
import structlog
from baml_bridge.errors import BamlError

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import (
    EBMGraphSimplificationError,
    InvalidOptionError,
    LLMReplyError,
    LLMRequestError,
    VisionUnsupportedError,
)
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.terms import main_effect_features
from ebaml.graphs.token_budget import TokenEstimator, check_token_budget, estimate_tokens
from ebaml.graphs.token_weights import ModelKey
from ebaml.llm.difficulties import DifficultyProposal, features_text, read_proposal
from ebaml.llm.graph_views import GraphView, ImageSettings, RenderedGraph, VisionSwitch, graph_view, shrinks_with_text
from ebaml.llm.model_parts import feature_importances_text, features_without_graph, skipped_interactions
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.progress_bars import progress
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.retry import ShrinkingGraphCall, call_with_shrinking_graph
from ebaml.llm.sdk_steps import (
    CountedText,
    Described,
    FanOut,
    FeatureOutcome,
    ModelIdentity,
    SDKCall,
    SDKTypes,
    SDKValue,
    Step,
    TokenCounts,
)
from ebaml.local.audience import resolve_audience
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.fact_types import LocalFacts
from ebaml.local.facts import compute_local_facts

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

GRAPH_SENTENCES = 7
_QUOTE_CHARACTERS = 300


def prompt_options_pipeline(prompt: PromptOptions) -> Generator[Step, SDKValue, SDKValue]:
    """Build the BAML prompt options from the prompt options, falling back to the BAML defaults.

    Parameters
    ----------
    prompt: PromptOptions
        The descriptions to use. A description that is ``None`` keeps the BAML default.

    Returns
    -------
    SDKValue
        The ``baml_sdk.GraphPromptOptions`` of the BAML LLM client.

    Yields
    ------
    Step
        The call that reads the BAML defaults.

    """
    options = yield SDKCall("default_graph_options", {})
    descriptions = {
        "expert_description": prompt.expert_description,
        "graph_description": prompt.graph_description,
        "dataset_description": prompt.dataset_description,
        "task_description": prompt.task_description,
    }
    overrides = {name: value for name, value in descriptions.items() if value is not None}
    return options.model_copy(update=overrides)


def model_spec_pipeline(
    model: ModelOptions | str | None,
    types: SDKTypes,
) -> Generator[Step, SDKValue, SDKValue]:
    """Build the BAML model spec from the model options, and check that BAML knows the provider.

    The check builds the client and discards it. It sends no request, so it runs before the first paid call.

    Parameters
    ----------
    model: ModelOptions | str | None
        The model, or the ``"provider/model"`` shorthand. ``None`` selects the default clients of the BAML code.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    SDKValue
        The ``baml_sdk.ModelSpec`` to pass as ``model`` to the BAML functions, or ``None`` for the default clients.

    Yields
    ------
    Step
        The call that checks the model, and after a rejection the call that lists the providers.

    Raises
    ------
    InvalidOptionError
        If BAML does not know the provider.
    BamlError
        If the check fails for another reason.

    """
    if model is None:
        return None
    options = ModelOptions.parse(model)
    spec = types.model_spec(
        provider=options.provider,
        model=options.model,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        temperature=options.temperature,
        max_output_tokens=options.max_output_tokens,
        headers=dict(options.headers) or None,
    )
    try:
        yield SDKCall("check_model", {"spec": spec})
    except BamlError as error:
        if error.class_name != "baml.errors.InvalidArgument":
            raise
        providers = yield SDKCall("supported_providers", {})
        raise InvalidOptionError(options.provider, providers) from error
    return spec


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


def log_usage(what: str, usage: TokenCounts, *, truncated_attempts: int) -> None:
    """Log the token usage reported by the LLM provider.

    The usage leaves out the attempts that were cut off by ``max_tokens``, because the provider error carries no token
    count. ``truncated_attempts`` tells how many attempts the usage leaves out, so the usage is a lower bound.

    Parameters
    ----------
    what: str
        A label identifying the call in the log message.
    usage: TokenCounts
        The token usage of the replies.
    truncated_attempts: int
        The number of attempts that were cut off, and that the usage does not count.

    """
    logger.info(
        "usage",
        what=what,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        truncated_attempts=truncated_attempts,
    )


@attrs.frozen
class _TokenTotals:
    """The tokens of several replies, added up."""

    input_tokens: int
    output_tokens: int


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


def describe_graph_pipeline(request: DescribeGraphRequest, types: SDKTypes) -> Generator[Step, SDKValue, str]:
    """Describe a graph: the pipeline of ``describe_graph`` and its async form.

    Parameters
    ----------
    request: DescribeGraphRequest
        The EBM, the index of the feature, the number of sentences, the options of the graph, the prompts and the model.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    str
        The description of the graph.

    Yields
    ------
    Step
        The calls of the description.

    """
    ebm = request.ebm
    feature_index = request.feature_index
    omitted_terms = skipped_interactions(ebm, {feature_index})
    model_spec = yield from model_spec_pipeline(request.model, types)
    switch = VisionSwitch.for_call(request.graph, request.model)
    rendered = render_graph(ebm, feature_index, request.graph, model_spec)
    prompt_options = yield from prompt_options_pipeline(request.prompt)
    job = GraphJob(
        ebm=ebm,
        num_sentences=request.num_sentences,
        graph_options=request.graph,
        prompt_options=prompt_options,
        model=model_spec,
        switch=switch,
        image=ImageSettings.from_options(ebm, request.graph),
        omitted_terms=omitted_terms,
    )
    described = yield from describe_graph_counted(job, feature_index, rendered)
    log_usage(
        f"describe_graph({ebm.feature_names_in_[feature_index]})",
        described.counted.usage,
        truncated_attempts=described.truncated_attempts,
    )
    return described.counted.text


@attrs.frozen
class FeatureFailure:
    """A feature whose graph description failed, and the error of its request.

    Attributes
    ----------
    feature: str
        The name of the feature.
    error: LLMRequestError
        The error. It concerns this feature only: an error that concerns every request stops ``describe_ebm``.

    """

    feature: str
    error: LLMRequestError


@functools.singledispatch
def feature_error(error: Exception) -> LLMRequestError:
    """Turn the error of one graph pipeline into the error of its feature, or raise it when it stops the whole call.

    An ``LLMRequestError`` concerns one feature when its class says that it does not affect every request. A
    ``BamlError`` that ``provider_errors`` does not explain is a reply that stays cut off or does not parse, so it
    concerns one feature, and the function returns an ``LLMReplyError``. The function raises any other error again,
    and an ``LLMRequestError`` that affects every request, because such an error stops the whole call.

    Parameters
    ----------
    error: Exception
        The error of the pipeline. The registered functions of the ``LLMRequestError`` and the ``BamlError`` classes
        return the error of the feature.

    """
    raise error


@feature_error.register
def _feature_error_of_request(error: LLMRequestError) -> LLMRequestError:
    if error.affects_every_request:
        raise error
    return error


@feature_error.register
def _feature_error_of_reply(error: BamlError) -> LLMRequestError:
    reply_error = LLMReplyError(
        "The reply of the LLM for one graph was cut off by max_tokens after the retries, or did not parse.\n"
        f"The BAML runtime said: {str(error)[:_QUOTE_CHARACTERS]}\n"
        "To fix it:\n"
        '  1. Raise max_output_tokens, for example model=ModelOptions("provider", "model", max_output_tokens=4000).\n'
        "  2. Lower max_tokens in GraphOptions, so that the graph text and the reply are shorter.\n"
        "  3. Run the call again, or choose another model, if the reply did not parse.",
    )
    reply_error.__cause__ = error
    return reply_error


@attrs.frozen
class _Settled:
    """The descriptions of the features that succeeded, by name, and the features that failed, in order."""

    described: dict[str, Described]
    failed_features: list[FeatureFailure]

    def graph_descriptions(self) -> str:
        """Join the descriptions for the summary prompt.

        Returns
        -------
        str
            One paragraph for each feature that succeeded: its name and its description.

        """
        return "\n\n".join(f"{name}: {item.counted.text}" for name, item in self.described.items())


def _settle(feature_names: Sequence[str], outcomes: Sequence[FeatureOutcome]) -> _Settled:
    """Split the outcomes of the graphs into descriptions and failures, and log the failures.

    If every feature failed, the summary has nothing to cover, so the function raises the error of the first feature.

    Parameters
    ----------
    feature_names: Sequence[str]
        The name of each feature, in the order of the outcomes.
    outcomes: Sequence[FeatureOutcome]
        The outcome of the graph of each feature.

    Returns
    -------
    _Settled
        The descriptions and the failures.

    """
    pairs = list(zip(feature_names, outcomes, strict=True))
    described = {name: outcome.described for name, outcome in pairs if outcome.described is not None}
    failed = [FeatureFailure(name, outcome.error) for name, outcome in pairs if outcome.error is not None]
    if failed and not described:
        raise failed[0].error
    if failed:
        logger.warning("features_failed", features=[failure.feature for failure in failed])
    return _Settled(described, failed)


@attrs.frozen
class EBMDescription:
    """The description of an EBM, and the parts of the model that it leaves out.

    A reader of ``text`` learns from the text itself what the description leaves out. A program reads the same
    information from the three lists, with no need to parse a log or the text.

    Attributes
    ----------
    text: str
        The description. Pass it as ``global_explanation`` to ``explain_instance``.
    skipped_terms: list[str]
        The names of the interaction terms that the description leaves out. The list is empty for a model without
        interactions.
    skipped_features: list[str]
        The names of the features that have no main-effect term, because ``exclude`` or ``remove_terms`` left them
        out. The description has no graph for them.
    failed_features: list[FeatureFailure]
        The features whose graph description failed, each with its error, in the order of the features. The summary
        names them and does not cover them. The list is empty when every graph description succeeded.

    """

    text: str
    skipped_terms: list[str]
    skipped_features: list[str]
    failed_features: list[FeatureFailure] = attrs.field(factory=list, kw_only=True)


def describe_ebm_pipeline(request: DescribeEBMRequest, types: SDKTypes) -> Generator[Step, SDKValue, EBMDescription]:
    """Describe an EBM: the pipeline of ``describe_ebm`` and its async form.

    Parameters
    ----------
    request: DescribeEBMRequest
        The EBM, the number of sentences, the number of features described at the same time, the progress bar, the
        options of the graph, the prompts and the model.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    EBMDescription
        The description of the EBM, and the names of the interaction terms and the features that it leaves out.

    Yields
    ------
    Step
        The checks of the model, the ``FanOut`` of the graphs, and the call of the summary.

    """
    # Convert all objects to text and check the model before prompting the LLM the first time, so that errors in
    # processing one of the graphs or in the model surface before any LLM call is paid for.
    # a feature that exclude or remove_terms left out has no graph to describe
    ebm = request.ebm
    features = main_effect_features(ebm)
    feature_names = [ebm.feature_names_in_[feature_index] for feature_index in features]
    omitted_terms = skipped_interactions(ebm, features)
    omitted_features = features_without_graph(ebm, features)
    feature_importances = feature_importances_text(ebm)
    model_spec = yield from model_spec_pipeline(request.model, types)
    graphs = [render_graph(ebm, feature_index, request.graph, model_spec) for feature_index in features]
    opts = yield from prompt_options_pipeline(request.prompt)
    job = GraphJob(
        ebm=ebm,
        num_sentences=GRAPH_SENTENCES,
        graph_options=request.graph,
        prompt_options=opts,
        model=model_spec,
        switch=VisionSwitch.for_call(request.graph, request.model),
        image=ImageSettings.from_options(ebm, request.graph),
    )

    with progress(len(features) + 1, "describe_ebm", enabled=request.progress_bar) as finish_step:
        outcomes: list[FeatureOutcome] = yield FanOut(
            [
                describe_graph_counted(job, feature_index, graph)
                for feature_index, graph in zip(features, graphs, strict=True)
            ],
            request.max_concurrency,
            finish_step,
            feature_error,
        )
        settled = _settle(feature_names, outcomes)
        with provider_errors(job.switch.model):
            summary = yield SDKCall(
                "summarize_ebm",
                {
                    "feature_importances": feature_importances,
                    "graph_descriptions": settled.graph_descriptions(),
                    "num_sentences": request.num_sentences,
                    "opts": opts,
                    "model": model_spec,
                    "omitted_terms": omitted_terms,
                    "omitted_features": omitted_features,
                    "failed_features": [failure.feature for failure in settled.failed_features],
                },
            )
        finish_step()
    described = list(settled.described.values())
    usage = _TokenTotals(
        input_tokens=summary.usage.input_tokens + sum(item.counted.usage.input_tokens for item in described),
        output_tokens=summary.usage.output_tokens + sum(item.counted.usage.output_tokens for item in described),
    )
    log_usage("describe_ebm", usage, truncated_attempts=sum(item.truncated_attempts for item in described))
    return EBMDescription(summary.text, omitted_terms, omitted_features, failed_features=settled.failed_features)


@attrs.frozen
class LocalExplanation:
    """The explanation of one prediction: three kinds of explanation in words, and the facts they come from."""

    counterfactual: str
    sufficient_reason: str
    completions: str
    facts: LocalFacts


def explain_instance_pipeline(
    request: ExplainInstanceRequest,
    types: SDKTypes,
) -> Generator[Step, SDKValue, LocalExplanation]:
    """Explain one prediction: the pipeline of ``explain_instance`` and its async form.

    Parameters
    ----------
    request: ExplainInstanceRequest
        The EBM, the instance, the global explanation, the audience, the number of sentences, the language, the local
        options, the prompts and the model.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    LocalExplanation
        The three explanations and the facts they come from.

    Yields
    ------
    Step
        The checks of the model and the call of the explanation.

    """
    audience = resolve_audience(request.audience)
    facts = compute_local_facts(request.ebm, request.instance, request.local)
    texts = facts_to_texts(facts, audience)
    model_spec = yield from model_spec_pipeline(request.model, types)
    opts = yield from prompt_options_pipeline(request.prompt)
    with provider_errors():
        result = yield SDKCall(
            "explain_local",
            {
                "global_explanation": request.global_explanation,
                "facts": types.local_fact_texts(**attrs.asdict(texts, recurse=False)),
                "audience_guidance": audience.guidance(),
                "num_sentences": request.num_sentences,
                "opts": opts,
                "language": request.language,
                "model": model_spec,
            },
        )
    logger.info(
        "usage",
        what="explain_instance",
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )
    return LocalExplanation(result.counterfactual, result.sufficient_reason, result.completions, facts)


def propose_difficulties_pipeline(
    request: ProposeDifficultiesRequest,
    types: SDKTypes,
) -> Generator[Step, SDKValue, DifficultyProposal]:
    """Ask how hard the change of each input is: the pipeline of ``propose_difficulties`` and its async form.

    Parameters
    ----------
    request: ProposeDifficultiesRequest
        The EBM, the instance, the base cost strategy, the prompts and the model.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    DifficultyProposal
        The difficulty and the reason of each input that the reply rates, and the ``ScaledCost`` that they give.

    Yields
    ------
    Step
        The checks of the model and the call of the proposal.

    """
    ebm = request.ebm
    model_spec = yield from model_spec_pipeline(request.model, types)
    features = features_text(ebm, request.instance)
    opts = yield from prompt_options_pipeline(request.prompt)
    with provider_errors():
        result = yield SDKCall("propose_difficulty", {"features": features, "opts": opts, "model": model_spec})
    logger.info(
        "usage",
        what="propose_difficulties",
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )
    feature_names = [ebm.feature_names_in_[feature_index] for feature_index in main_effect_features(ebm)]
    return read_proposal(result.difficulties, feature_names, request.base)
