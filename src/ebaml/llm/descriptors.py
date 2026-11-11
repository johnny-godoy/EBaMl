"""Describe EBM graphs and models using LLMs.

The prompts and the chat conversations live in ``baml_src/talk_to_ebm.baml``; this module extracts the graphs from the
EBM, converts them to text, and calls the generated ``baml_sdk`` functions. Each function takes one request
from ``ebaml.llm.requests``, and the ``model`` field of the request chooses the LLM of the call.
"""

import functools
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import attrs
import structlog
from baml_bridge.errors import BamlError

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import InvalidOptionError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.terms import main_effect_features, main_term
from ebaml.graphs.token_budget import TokenEstimator, check_token_budget, estimate_tokens
from ebaml.graphs.token_weights import ModelKey
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.progress_bars import progress
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ExplainInstanceRequest
from ebaml.llm.retry import ShrinkingGraphCall, call_with_shrinking_graph
from ebaml.llm.sdk_setup import ensure_sdk
from ebaml.local.audience import resolve_audience
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.facts import LocalFacts, compute_local_facts

# `baml_sdk` is generated and gitignored, so a checkout may lack it. Generate it before the import, or explain the fix.
ensure_sdk()

import baml_sdk  # ruff: ignore[module-import-not-at-top-of-file] -- ensure_sdk must run first

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

_GRAPH_SENTENCES = 7
# the default of the option argument, as a module-level value that nothing changes: the option class is frozen
_DEFAULT_PROMPT_OPTIONS = PromptOptions()


def _feature_importances_to_text(ebm: ExplainableBoostingModel) -> str:
    """Convert the importances of the main-effect terms of an EBM to text, one line per feature.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM model.

    Returns
    -------
        str: Textual representation of the feature importances.

    """
    # the importances are kept by term, and term i is not always feature i
    importances = ebm.term_importances()
    return "\n".join(
        f"{ebm.feature_names_in_[feature_idx]}: {importances[main_term(ebm, feature_idx)]:.2f}"
        for feature_idx in main_effect_features(ebm)
    )


def _skipped_interactions(ebm: ExplainableBoostingModel, described: Collection[int]) -> list[str]:
    """Name the interaction terms of the described features, which the descriptions skip, and log a warning.

    The names go to the prompt, so that the text of the description says what it leaves out. The warning tells the
    program the same.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM.
    described : Collection[int]
        The indices of the features that the call describes.

    Returns
    -------
    list[str]
        The names of the skipped terms, in the order of the model.

    """
    skipped = [
        name
        for name, features in zip(ebm.term_names_, ebm.term_features_, strict=True)
        if (
            len(features) > 1  # pragma: no mutate -- a term of zero features does not exist
            and any(feature in described for feature in features)
        )
    ]
    if skipped:
        logger.warning("interactions_skipped", terms=skipped, reason="the descriptions cover the main effects only")
    return skipped


def _features_without_graph(ebm: ExplainableBoostingModel, described: Collection[int]) -> list[str]:
    """Name the features that have no main-effect term, which the descriptions skip, and log a warning.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM.
    described : Collection[int]
        The indices of the features that have a graph.

    Returns
    -------
    list[str]
        The names of the features that ``exclude`` or ``remove_terms`` left without a graph.

    """
    skipped = [name for index, name in enumerate(ebm.feature_names_in_) if index not in described]
    if skipped:
        logger.warning("features_skipped", features=skipped, reason="the features have no main-effect term")
    return skipped


def _prompt_options(prompt: PromptOptions) -> baml_sdk.GraphPromptOptions:
    """Build the BAML prompt options from the prompt options, falling back to the BAML defaults.

    Parameters
    ----------
    prompt: PromptOptions
        The descriptions to use. A description that is ``None`` keeps the BAML default.

    Returns
    -------
    baml_sdk.GraphPromptOptions
        The prompt options for the BAML LLM client.

    """
    options = baml_sdk.default_graph_options()
    descriptions = {
        "expert_description": prompt.expert_description,
        "graph_description": prompt.graph_description,
        "dataset_description": prompt.dataset_description,
        "task_description": prompt.task_description,
    }
    overrides = {name: value for name, value in descriptions.items() if value is not None}
    return baml_sdk.GraphPromptOptions(**{**options.model_dump(), **overrides})


def _model_spec(model: ModelOptions | str | None) -> baml_sdk.ModelSpec | None:
    """Build the BAML model spec from the model options, and check that BAML knows the provider.

    The check builds the client and discards it. It sends no request, so it runs before the first paid call.

    Parameters
    ----------
    model: ModelOptions | str | None
        The model, or the ``"provider/model"`` shorthand. ``None`` selects the default clients of the BAML code.

    Returns
    -------
    baml_sdk.ModelSpec | None
        The spec to pass as ``model`` to the BAML functions, or ``None`` for the default clients.

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
    spec = baml_sdk.ModelSpec(
        provider=options.provider,
        model=options.model,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        temperature=options.temperature,
        max_output_tokens=options.max_output_tokens,
        headers=dict(options.headers) or None,
    )
    try:
        baml_sdk.check_model(spec=spec)
    except BamlError as error:
        if error.class_name != "baml.errors.InvalidArgument":
            raise
        raise InvalidOptionError(options.provider, baml_sdk.supported_providers()) from error
    return spec


def _estimator(options: GraphOptions, model: baml_sdk.ModelSpec | None) -> TokenEstimator | None:
    """Get the token estimator of the learner in the options, for the model that the call uses.

    Parameters
    ----------
    options: GraphOptions
        The options of the graph. Its ``token_learner`` can be ``None``.
    model: baml_sdk.ModelSpec | None
        The model of the requests. ``None`` selects the default clients, which the learner does not know by name.

    Returns
    -------
    TokenEstimator | None
        The estimator of the model, or ``None`` when the options have no learner.

    """
    if options.token_learner is None:
        return None
    return options.token_learner.estimator_for(None if model is None else ModelKey(model.provider, model.model))


def _graph_text(
    ebm: ExplainableBoostingModel,
    feature_index: int,
    options: GraphOptions,
    model: baml_sdk.ModelSpec | None,
) -> str:
    """Convert an EBM graph to text, with the token budget check.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM model.
    feature_index: int
        The index of the feature to convert to text.
    options: GraphOptions
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.
    model: baml_sdk.ModelSpec | None
        The model of the requests. The token estimator of the learner in ``options`` depends on it.

    Returns
    -------
    str
        The textual representation of the EBM graph.

    """
    graph = EBMGraph.from_ebm(
        ebm,
        feature_index,
        options.normalization,
        use_feature_bounds=options.use_feature_bounds,
        classes=options.classes,
    )
    return graph.to_text(
        include_description=options.include_description,
        feature_format=options.feature_format,
        x_axis_precision=options.x_axis_precision,
        y_axis_precision=options.y_axis_precision,
        confidence_bounds=options.confidence_bounds,
        confidence_level=options.confidence_level,
        max_tokens=options.max_tokens,
        overbudget_percent=options.overbudget_percent,
        check_budget=_estimator(options, model) or check_token_budget,
    )


def _is_truncated(error: BamlError) -> bool:
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


def _log_usage(
    what: str,
    counted: baml_sdk.Counted,
    *,
    truncated_attempts: int,
) -> str:
    """Log the token usage reported by the LLM provider and return the text.

    The usage leaves out the attempts that were cut off by ``max_tokens``, because the provider error carries no token
    count. ``truncated_attempts`` tells how many attempts the usage leaves out, so the usage is a lower bound.

    Parameters
    ----------
    what: str
        A label identifying the call in the log message.
    counted: baml_sdk.Counted
        The LLM reply together with its token usage.
    truncated_attempts: int
        The number of attempts that were cut off, and that the usage does not count.

    Returns
    -------
    str
        The text of the reply.

    """
    usage = counted.usage
    logger.info(
        "usage",
        what=what,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        truncated_attempts=truncated_attempts,
    )
    return counted.text


@attrs.frozen
class _Described:
    """The description of one graph, with its token usage and the number of attempts that were cut off."""

    counted: baml_sdk.Counted
    truncated_attempts: int


@attrs.frozen
class _GraphJob:
    """The settings that every graph of one call shares."""

    ebm: ExplainableBoostingModel
    num_sentences: int
    graph_options: GraphOptions
    prompt_options: baml_sdk.GraphPromptOptions
    model: baml_sdk.ModelSpec | None
    omitted_terms: Sequence[str] = ()


def _describe_graph_counted(job: _GraphJob, feature_index: int, graph: str) -> _Described:
    """Describe one graph with the LLM, simplifying the graph further whenever the reply is cut off by ``max_tokens``.

    A failure that the user can fix raises an ``LLMRequestError`` (see ``ebaml.llm.provider_errors``). Any other failure
    raises the ``BamlError`` of the LLM call. A reply cut off by ``max_tokens`` is retried first, as
    ``ebaml.llm.retry.call_with_shrinking_graph`` says.

    Parameters
    ----------
    job: _GraphJob
        The EBM, the number of sentences, the options of the graph text (a retry changes ``max_tokens`` and
        ``overbudget_percent``), the prompt options as made by ``_prompt_options``, the model of the requests as
        made by ``_model_spec`` (``None`` selects the default clients), and the interaction terms that the description
        must name as left out.
    feature_index: int
        The index of the feature to describe.
    graph: str
        The text of the graph, as made by ``_graph_text``.

    Returns
    -------
    _Described
        The description with its token usage, and the number of attempts that were cut off. The usage counts the
        attempt that succeeded only.

    """
    truncated: list[BamlError] = []
    graph_options = job.graph_options
    estimator = _estimator(graph_options, job.model)
    estimate = estimate_tokens if estimator is None else estimator.estimate
    sent: list[str] = []

    def ask(text: str) -> baml_sdk.Counted:
        sent.append(text)
        return baml_sdk.describe_graph_cot(
            graph=text,
            num_sentences=job.num_sentences,
            opts=job.prompt_options,
            model=job.model,
            omitted_terms=list(job.omitted_terms),
        )

    with provider_errors():
        counted = call_with_shrinking_graph(
            ShrinkingGraphCall(
                call=ask,
                graph=graph,
                error_type=BamlError,
                is_truncated=_is_truncated,
                estimate_tokens=estimate,
                rebuild=lambda budget: _graph_text(
                    job.ebm,
                    feature_index,
                    attrs.evolve(graph_options, max_tokens=budget, overbudget_percent=0),
                    job.model,
                ),
                on_truncated=truncated.append,
                label=job.ebm.feature_names_in_[feature_index],
            ),
        )
    if graph_options.token_learner is not None:
        # the last text that the call sent is the one that the successful attempt used
        graph_options.token_learner.observe(counted.usage.samples, sent[-1])
    return _Described(counted, len(truncated))


def describe_graph(request: DescribeGraphRequest) -> str:
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    A reply cut off by ``max_tokens`` is retried with a simplified graph, up to 4 times. The logged usage leaves out the
    cut-off attempts, because the provider error carries no token count. It is a lower bound, and the ``usage`` log
    event has the number of those attempts in ``truncated_attempts``. The description covers the main effect of the
    feature. The prompt names the interaction terms of the feature, so that the description says that it leaves them
    out, and an ``interactions_skipped`` warning names them too.

    Parameters
    ----------
    request: DescribeGraphRequest
        The EBM, the index of the feature, the number of sentences, the options of the graph, the prompts and the model.

    Returns
    -------
    str
        The description of the graph.

    """
    ebm = request.ebm
    feature_index = request.feature_index
    omitted_terms = _skipped_interactions(ebm, {feature_index})
    model_spec = _model_spec(request.model)
    graph_text = _graph_text(ebm, feature_index, request.graph, model_spec)
    job = _GraphJob(
        ebm,
        request.num_sentences,
        request.graph,
        _prompt_options(request.prompt),
        model_spec,
        omitted_terms,
    )
    described = _describe_graph_counted(job, feature_index, graph_text)
    return _log_usage(
        f"describe_graph({ebm.feature_names_in_[feature_index]})",
        described.counted,
        truncated_attempts=described.truncated_attempts,
    )


def sample_graph_tokens(
    graph: str,
    model: ModelOptions | str | None = None,
    *,
    prompt: PromptOptions = _DEFAULT_PROMPT_OPTIONS,
) -> list[baml_sdk.TokenSample]:
    """Send one request with the text of a graph and return its token samples.

    The request is the first turn of ``describe_graph``. Each sample has the body of the request and the input tokens
    that the provider counted. Set ``max_output_tokens`` in the model options to a small number, such as 16, to keep the
    cost low: a reply that the limit cuts off is not an error here. ``scripts/fit_token_weights.py`` uses the function
    to fit the token weights of a model.

    Parameters
    ----------
    graph: str
        The text of the graph, as ``EBMGraph.to_text`` makes it.
    model: ModelOptions | str | None, default=None
        The LLM of the call, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``.
    prompt: PromptOptions, default=PromptOptions()
        The descriptions that the prompt uses.

    Returns
    -------
    list[baml_sdk.TokenSample]
        One sample for each request that the provider counted. The list is empty when the provider reports no count.

    """
    model_spec = _model_spec(model)
    with provider_errors():
        return baml_sdk.sample_graph_tokens(graph=graph, opts=_prompt_options(prompt), model=model_spec)


def chat_with_mcp(
    command: str,
    arguments: Sequence[str],
    model: ModelOptions | str | None = None,
    *,
    max_steps: int = 12,
) -> int:
    """Chat in the terminal with an agent that reads an EBM through an MCP server.

    BAML starts the server once and keeps it for the whole chat. Each message that you type gets an answer from the
    LLM, which calls the tools of the server as it needs them. The chat reads standard input and ends on an empty line.

    Parameters
    ----------
    command: str
        The program that starts the MCP server, for example the path of ``ebaml-mcp``.
    arguments: Sequence[str]
        The arguments of the program, for example ``["--model", "model.skops", "--data", "train.csv"]``.
    model: ModelOptions | str | None, default=None
        The LLM of the chat, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``.
    max_steps: int, default=12
        The most rounds of tool calls for one answer.

    Returns
    -------
    int
        The number of questions that the chat answered.

    """
    model_spec = _model_spec(model)
    with provider_errors():
        return baml_sdk.chat_with_mcp(command=command, args=list(arguments), model=model_spec, max_steps=max_steps)


def _describe_graphs(
    describe: Callable[[int, str], _Described],
    features: Collection[int],
    graph_texts: Collection[str],
    max_concurrency: int,
    finish_step: Callable[[], None],
) -> list[_Described]:
    """Describe the graphs of the features at the same time, and count each one that finishes.

    The function counts the graphs in the thread of the caller, in the order in which they finish. It returns the
    descriptions in the order of the features. If a graph fails, the function cancels the graphs that have not started
    and raises the error of the first graph that failed.

    Parameters
    ----------
    describe: Callable[[int, str], _Described]
        Describes one graph from the index of its feature and its text.
    features: Collection[int]
        The indices of the features to describe.
    graph_texts: Collection[str]
        The text of the graph of each feature, in the order of ``features``.
    max_concurrency: int
        The maximum number of graphs that the function describes at the same time.
    finish_step: Callable[[], None]
        Called once for each graph that finishes.

    Returns
    -------
    list[_Described]
        The description of each graph, in the order of ``features``.

    """
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures: list[Future[_Described]] = [
            pool.submit(describe, feature_index, graph_text)
            for feature_index, graph_text in zip(features, graph_texts, strict=True)
        ]
        try:
            for finished in as_completed(futures):
                finished.result()
                finish_step()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return [future.result() for future in futures]


@attrs.frozen
class EBMDescription:
    """The description of an EBM, and the parts of the model that it leaves out.

    A reader of ``text`` learns from the text itself what the description leaves out. A program reads the same
    information from the two lists, with no need to parse a log or the text.

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

    """

    text: str
    skipped_terms: list[str]
    skipped_features: list[str]


def describe_ebm(request: DescribeEBMRequest) -> EBMDescription:
    """Ask the LLM to describe an EBM.

    Each feature graph is described like in ``describe_graph`` (concurrently, at most ``max_concurrency`` at a time,
    each retried with a simplified graph if the reply is cut off by ``max_tokens``), and then the LLM summarizes the
    descriptions. The descriptions skip the interaction terms, and a feature that ``exclude`` or ``remove_terms`` left
    out, because it has no graph. The summary prompt names both, so that the text says what it leaves out. An
    ``interactions_skipped`` and a ``features_skipped`` warning name them too.
    The logged usage leaves out the cut-off attempts, because the provider error carries no token count. It is a lower
    bound, and the ``usage`` log event has the number of those attempts in ``truncated_attempts``.

    Parameters
    ----------
    request: DescribeEBMRequest
        The EBM, the number of sentences, the number of features described at the same time, whether to show a progress
        bar on the error stream (one step for each feature and one for the summary; without a terminal, the bar writes
        one line for each step), the options of the graph, the prompts and the model.

    Returns
    -------
    EBMDescription
        The description of the EBM, and the names of the interaction terms and the features that it leaves out.

    """
    # Convert all objects to text and check the model before prompting the LLM the first time, so that errors in
    # processing one of the graphs or in the model surface before any LLM call is paid for.
    # a feature that exclude or remove_terms left out has no graph to describe
    ebm = request.ebm
    features = main_effect_features(ebm)
    feature_names = [ebm.feature_names_in_[feature_index] for feature_index in features]
    omitted_terms = _skipped_interactions(ebm, features)
    omitted_features = _features_without_graph(ebm, features)
    feature_importances = _feature_importances_to_text(ebm)
    model_spec = _model_spec(request.model)
    graph_texts = [_graph_text(ebm, feature_index, request.graph, model_spec) for feature_index in features]
    opts = _prompt_options(request.prompt)
    job = _GraphJob(ebm, _GRAPH_SENTENCES, request.graph, opts, model_spec)

    with progress(len(features) + 1, "describe_ebm", enabled=request.progress_bar) as finish_step:
        described = _describe_graphs(
            functools.partial(_describe_graph_counted, job),
            features,
            graph_texts,
            request.max_concurrency,
            finish_step,
        )
        graph_descriptions = "\n\n".join(
            f"{name}: {item.counted.text}"
            for name, item in zip(
                feature_names,
                described,
                strict=True,  # pragma: no mutate -- one item per feature, so strict never raises
            )
        )
        with provider_errors():
            summary = baml_sdk.summarize_ebm(
                feature_importances=feature_importances,
                graph_descriptions=graph_descriptions,
                num_sentences=request.num_sentences,
                opts=opts,
                model=model_spec,
                omitted_terms=omitted_terms,
                omitted_features=omitted_features,
            )
        finish_step()
    usage = baml_sdk.TokenUsage(
        input_tokens=summary.usage.input_tokens + sum(item.counted.usage.input_tokens for item in described),
        output_tokens=summary.usage.output_tokens + sum(item.counted.usage.output_tokens for item in described),
        samples=[],
    )
    text = _log_usage(
        "describe_ebm",
        baml_sdk.Counted(text=summary.text, usage=usage),
        truncated_attempts=sum(item.truncated_attempts for item in described),
    )
    return EBMDescription(text, omitted_terms, omitted_features)


@attrs.frozen
class LocalExplanation:
    """The explanation of one prediction: three kinds of explanation in words, and the facts they come from."""

    counterfactual: str
    sufficient_reason: str
    completions: str
    facts: LocalFacts


def explain_instance(request: ExplainInstanceRequest) -> LocalExplanation:
    """Explain one prediction of an EBM with three explanations, computed exactly and put in words by the LLM.

    The facts come from ``ebaml.local.compute_local_facts``: the minimum change that flips the prediction
    (counterfactual), a minimum set of inputs that fixes it (sufficient reason), and the share of completions of a
    partially known input that get the same prediction (count completions). The global explanation of the model, e.g.
    the output of ``describe_ebm``, is prepended to each prompt.

    Parameters
    ----------
    request: ExplainInstanceRequest
        The EBM, the instance, the global explanation, the audience, the number of sentences, the language, the local
        options, the prompts and the model.

    Returns
    -------
    LocalExplanation
        The three explanations and the facts they come from.

    """
    audience = resolve_audience(request.audience)
    facts = compute_local_facts(request.ebm, request.instance, request.local)
    texts = facts_to_texts(facts, audience)
    model_spec = _model_spec(request.model)
    with provider_errors():
        result = baml_sdk.explain_local(
            global_explanation=request.global_explanation,
            facts=baml_sdk.LocalFactTexts(**attrs.asdict(texts, recurse=False)),
            audience_guidance=audience.guidance(),
            num_sentences=request.num_sentences,
            opts=_prompt_options(request.prompt),
            language=request.language,
            model=model_spec,
        )
    logger.info(
        "usage",
        what="explain_instance",
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )
    return LocalExplanation(result.counterfactual, result.sufficient_reason, result.completions, facts)
