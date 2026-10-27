"""Describe EBM graphs and models using LLMs.

The prompts and the chat conversations live in ``baml_src/talk_to_ebm.baml``; this module extracts the graphs from the
EBM, converts them to text, and calls the generated ``baml_sdk`` functions. Each function takes ``model`` to choose the
LLM of the call.
"""

import functools
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import attrs
import structlog
from baml_bridge.errors import BamlError

from ebaml.core.custom_types import Audience, ExplainableBoostingModel, coerce_enum
from ebaml.core.exceptions import InvalidOptionError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.retry import call_with_shrinking_graph
from ebaml.llm.sdk_setup import ensure_sdk
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.facts import LocalFacts, LocalOptions, compute_local_facts

# `baml_sdk` is generated and gitignored, so a checkout may lack it. Generate it before the import, or explain the fix.
ensure_sdk()

import baml_sdk  # ruff: ignore[module-import-not-at-top-of-file] -- ensure_sdk must run first

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

_GRAPH_SENTENCES = 7
# the defaults of the option arguments, as module-level values that nothing changes: the option classes are frozen
_DEFAULT_GRAPH_OPTIONS = GraphOptions()
_DEFAULT_LOCAL_OPTIONS = LocalOptions()
_DEFAULT_PROMPT_OPTIONS = PromptOptions()


def _feature_importances_to_text(ebm: ExplainableBoostingModel) -> str:
    """Convert the feature importances of an EBM to text.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM model.

    Returns
    -------
        str: Textual representation of the feature importances.

    """
    importances = ebm.term_importances()
    return "\n".join(
        f"{feature_name}: {importances[feature_idx]:.2f}"
        for feature_idx, feature_name in enumerate(ebm.feature_names_in_)
    )


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


def _graph_text(ebm: ExplainableBoostingModel, feature_index: int, options: GraphOptions) -> str:
    """Convert an EBM graph to text, with the BAML token budget check.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM model.
    feature_index: int
        The index of the feature to convert to text.
    options: GraphOptions
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.

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
        check_budget=baml_sdk.check_token_budget,
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


def _log_usage(what: str, counted: baml_sdk.Counted, *, truncated_attempts: int) -> str:
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


def _describe_graph_counted(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] -- the retry needs the EBM, the feature, the first text, and the two option sets
    ebm: ExplainableBoostingModel,
    feature_index: int,
    graph: str,
    num_sentences: int,
    graph_options: GraphOptions,
    opts: baml_sdk.GraphPromptOptions,
    model: baml_sdk.ModelSpec | None,
) -> _Described:
    """Describe one graph with the LLM, simplifying the graph further whenever the reply is cut off by ``max_tokens``.

    A failure that the user can fix raises an ``LLMRequestError`` (see ``ebaml.llm.provider_errors``). Any other failure
    raises the ``BamlError`` of the LLM call. A reply cut off by ``max_tokens`` is retried first, as
    ``ebaml.llm.retry.call_with_shrinking_graph`` says.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature to describe.
    graph: str
        The text of the graph, as made by ``_graph_text``.
    num_sentences: int
        The desired number of sentences for the description.
    graph_options: GraphOptions
        The options of the graph text. A retry changes ``max_tokens`` and ``overbudget_percent``.
    opts: baml_sdk.GraphPromptOptions
        The prompt options, as made by ``_prompt_options``.
    model: baml_sdk.ModelSpec | None
        The model of the requests, as made by ``_model_spec``. ``None`` selects the default clients.

    Returns
    -------
    _Described
        The description with its token usage, and the number of attempts that were cut off. The usage counts the
        attempt that succeeded only.

    """
    truncated: list[BamlError] = []
    with provider_errors():
        counted = call_with_shrinking_graph(
            lambda text: baml_sdk.describe_graph_cot(graph=text, num_sentences=num_sentences, opts=opts, model=model),
            graph,
            error_type=BamlError,
            is_truncated=_is_truncated,
            estimate_tokens=lambda text: baml_sdk.estimate_tokens(text=text),
            rebuild=lambda budget: _graph_text(
                ebm,
                feature_index,
                attrs.evolve(graph_options, max_tokens=budget, overbudget_percent=0),
            ),
            on_truncated=truncated.append,
            label=ebm.feature_names_in_[feature_index],
        )
    return _Described(counted, len(truncated))


def describe_graph(  # ruff: ignore[too-many-arguments] -- the model, the feature, the length, and three keyword-only option arguments
    ebm: ExplainableBoostingModel,
    feature_index: int,
    num_sentences: int = 7,
    *,
    graph: GraphOptions = _DEFAULT_GRAPH_OPTIONS,
    prompt: PromptOptions = _DEFAULT_PROMPT_OPTIONS,
    model: ModelOptions | str | None = None,
) -> str:
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    A reply cut off by ``max_tokens`` is retried with a simplified graph, up to 4 times. The logged usage leaves out the
    cut-off attempts, because the provider error carries no token count. It is a lower bound, and the ``usage`` log
    event has the number of those attempts in ``truncated_attempts``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature to describe.
    num_sentences: int, default=7
        The desired number of sentences for the description.
    graph: GraphOptions, default=GraphOptions()
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.
    prompt: PromptOptions, default=PromptOptions()
        The descriptions that the prompts use.
    model: ModelOptions | str | None, default=None
        The LLM of the call, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``. EBaMl sends every request of the call to this model.

    Returns
    -------
    str
        The description of the graph.

    """
    graph_text = _graph_text(ebm, feature_index, graph)
    model_spec = _model_spec(model)
    described = _describe_graph_counted(
        ebm,
        feature_index,
        graph_text,
        num_sentences,
        graph,
        _prompt_options(prompt),
        model_spec,
    )
    return _log_usage(
        f"describe_graph({ebm.feature_names_in_[feature_index]})",
        described.counted,
        truncated_attempts=described.truncated_attempts,
    )


def describe_ebm(  # ruff: ignore[too-many-arguments] -- the model, two limits, and three keyword-only option arguments
    ebm: ExplainableBoostingModel,
    num_sentences: int = 30,
    max_concurrency: int = 8,
    *,
    graph: GraphOptions = _DEFAULT_GRAPH_OPTIONS,
    prompt: PromptOptions = _DEFAULT_PROMPT_OPTIONS,
    model: ModelOptions | str | None = None,
) -> str:
    """Ask the LLM to describe an EBM.

    Each feature graph is described like in ``describe_graph`` (concurrently, at most ``max_concurrency`` at a time,
    each retried with a simplified graph if the reply is cut off by ``max_tokens``), and then the LLM summarizes the
    descriptions. The logged usage leaves out the cut-off attempts, because the provider error carries no token count.
    It is a lower bound, and the ``usage`` log event has the number of those attempts in ``truncated_attempts``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    num_sentences: int, default=30
        The desired number of sentences for the description.
    max_concurrency: int, default=8
        The maximum number of features described at the same time.
    graph: GraphOptions, default=GraphOptions()
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.
    prompt: PromptOptions, default=PromptOptions()
        The descriptions that the prompts use.
    model: ModelOptions | str | None, default=None
        The LLM of the call, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``. EBaMl sends every request of the call to this model.

    Returns
    -------
    str
        The description of the EBM.

    """
    # Convert all objects to text and check the model before prompting the LLM the first time, so that errors in
    # processing one of the graphs or in the model surface before any LLM call is paid for.
    feature_importances = _feature_importances_to_text(ebm)
    feature_names = list(ebm.feature_names_in_)
    graph_texts = [_graph_text(ebm, feature_index, graph) for feature_index in range(len(feature_names))]
    model_spec = _model_spec(model)
    opts = _prompt_options(prompt)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        described = list(
            pool.map(
                functools.partial(
                    _describe_graph_counted,
                    ebm,
                    num_sentences=_GRAPH_SENTENCES,
                    graph_options=graph,
                    opts=opts,
                    model=model_spec,
                ),
                range(len(graph_texts)),
                graph_texts,
            ),
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
            num_sentences=num_sentences,
            opts=opts,
            model=model_spec,
        )
    usage = baml_sdk.TokenUsage(
        input_tokens=summary.usage.input_tokens + sum(item.counted.usage.input_tokens for item in described),
        output_tokens=summary.usage.output_tokens + sum(item.counted.usage.output_tokens for item in described),
    )
    return _log_usage(
        "describe_ebm",
        baml_sdk.Counted(text=summary.text, usage=usage),
        truncated_attempts=sum(item.truncated_attempts for item in described),
    )


@attrs.frozen
class LocalExplanation:
    """The explanation of one prediction: three kinds of explanation in words, and the facts they come from."""

    counterfactual: str
    sufficient_reason: str
    completions: str
    facts: LocalFacts


def explain_instance(  # ruff: ignore[too-many-arguments] -- the model, the instance, three settings of the explanation, and two option sets
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, object],
    global_explanation: str,
    audience: Audience | str = Audience.LAYPERSON,
    num_sentences: int = 5,
    *,
    local: LocalOptions = _DEFAULT_LOCAL_OPTIONS,
    prompt: PromptOptions = _DEFAULT_PROMPT_OPTIONS,
    model: ModelOptions | str | None = None,
) -> LocalExplanation:
    """Explain one prediction of an EBM with three explanations, computed exactly and put in words by the LLM.

    The facts come from ``ebaml.local.compute_local_facts``: the minimum change that flips the prediction
    (counterfactual), a minimum set of inputs that fixes it (sufficient reason), and the share of completions of a
    partially known input that get the same prediction (count completions). The global explanation of the model, e.g.
    the output of ``describe_ebm``, is prepended to each prompt.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM: a binary classifier without interactions.
    instance: Mapping[str, object]
        The value of every feature, by feature name.
    global_explanation: str
        The explanation of the whole model, computed once.
    audience: Audience | str, default=Audience.LAYPERSON
        Who reads the explanation. A layperson gets no model numbers; a data scientist gets scores, probabilities and
        confidence intervals.
    num_sentences: int, default=5
        The most sentences of each explanation.
    local: LocalOptions, default=LocalOptions()
        The options of ``compute_local_facts``.
    prompt: PromptOptions, default=PromptOptions()
        The descriptions that the prompts use. The local prompts read ``expert_description`` and
        ``dataset_description``.
    model: ModelOptions | str | None, default=None
        The LLM of the call, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``. EBaMl sends every request of the call to this model.

    Returns
    -------
    LocalExplanation
        The three explanations and the facts they come from.

    """
    audience = coerce_enum(Audience, audience)
    facts = compute_local_facts(ebm, instance, local)
    texts = facts_to_texts(facts, audience)
    model_spec = _model_spec(model)
    with provider_errors():
        result = baml_sdk.explain_local(
            global_explanation=global_explanation,
            facts=baml_sdk.LocalFactTexts(**attrs.asdict(texts, recurse=False)),
            audience=audience.value,
            num_sentences=num_sentences,
            opts=_prompt_options(prompt),
            model=model_spec,
        )
    logger.info(
        "usage",
        what="explain_instance",
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )
    return LocalExplanation(result.counterfactual, result.sufficient_reason, result.completions, facts)
