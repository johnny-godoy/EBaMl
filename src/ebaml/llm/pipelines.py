"""The pipelines of the LLM calls: one implementation for the sync and the async forms.

Each pipeline is a generator that yields the steps of ``ebaml.llm.sdk_steps``. A driver runs each step, then sends
the reply into the generator, or throws the error of the step into it. ``ebaml.llm.drivers`` holds the sync and the
async driver. The graph rendering, the shrinking retry, the
vision fallback, the provider errors and the facts live here once, so a fix reaches both forms.

The module imports no ``baml_sdk``. The classes of ``baml_sdk`` that a pipeline builds reach it in ``SDKTypes``, and
the replies reach it through ``send``. So a unit test drives a pipeline with plain replies.
"""

import functools
from collections.abc import Generator, Sequence

import attrs
import structlog
from baml_bridge.errors import BamlError

from ebaml.core.exceptions import (
    InvalidOptionError,
)
from ebaml.graphs.terms import main_effect_features
from ebaml.llm import graph_conversation
from ebaml.llm.difficulties import DifficultyProposal, features_text, read_proposal
from ebaml.llm.ebm_description import EBMDescription, feature_error, settle_outcomes
from ebaml.llm.graph_conversation import GraphJob, describe_graph_counted
from ebaml.llm.graph_views import ImageSettings, VisionSwitch
from ebaml.llm.model_parts import feature_importances_text, features_without_graph, skipped_interactions
from ebaml.llm.options import ModelOptions, PromptOptions
from ebaml.llm.progress_bars import progress
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.sdk_steps import (
    FanOut,
    FeatureOutcome,
    LocalWork,
    SDKCall,
    SDKTypes,
    SDKValue,
    Step,
    TokenCounts,
)
from ebaml.llm.token_learning import RequestSample
from ebaml.local.audience import resolve_audience
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.fact_types import LocalFacts
from ebaml.local.facts import compute_local_facts

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

GRAPH_SENTENCES = 7


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
    rendered = graph_conversation.render_graph(ebm, feature_index, request.graph, model_spec)
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
    graphs = [
        graph_conversation.render_graph(ebm, feature_index, request.graph, model_spec) for feature_index in features
    ]
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
        settled = settle_outcomes(feature_names, outcomes)
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
    # the facts take up to a second for a large model, so the async driver computes them outside the event loop
    facts: LocalFacts = yield LocalWork(
        "compute_local_facts",
        functools.partial(compute_local_facts, request.ebm, request.instance, request.local),
    )
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


def sample_graph_tokens_pipeline(
    graph: str,
    model: ModelOptions | str | None,
    prompt: PromptOptions,
    types: SDKTypes,
) -> Generator[Step, SDKValue, list[RequestSample]]:
    """Send the first turn of a graph description, and keep its token samples: the pipeline of ``sample_graph_tokens``.

    Parameters
    ----------
    graph: str
        The text of the graph.
    model: ModelOptions | str | None
        The LLM of the call. ``None`` selects the default clients.
    prompt: PromptOptions
        The descriptions that the prompt uses.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    list[RequestSample]
        One sample for each request that the provider counted.

    Yields
    ------
    Step
        The checks of the model and the call of the request.

    """
    model_spec = yield from model_spec_pipeline(model, types)
    opts = yield from prompt_options_pipeline(prompt)
    with provider_errors():
        samples: list[RequestSample] = yield SDKCall(
            "sample_graph_tokens",
            {"graph": graph, "opts": opts, "model": model_spec},
        )
    return samples


def chat_with_mcp_pipeline(
    command: str,
    arguments: Sequence[str],
    model: ModelOptions | str | None,
    max_steps: int,
    types: SDKTypes,
) -> Generator[Step, SDKValue, int]:
    """Chat with an agent that reads an EBM through an MCP server: the pipeline of ``chat_with_mcp``.

    Parameters
    ----------
    command: str
        The program that starts the MCP server.
    arguments: Sequence[str]
        The arguments of the program.
    model: ModelOptions | str | None
        The LLM of the chat. ``None`` selects the default clients.
    max_steps: int
        The most rounds of tool calls for one answer.
    types: SDKTypes
        The classes of ``baml_sdk``.

    Returns
    -------
    int
        The number of questions that the chat answered.

    Yields
    ------
    Step
        The checks of the model and the call of the chat.

    """
    model_spec = yield from model_spec_pipeline(model, types)
    with provider_errors():
        answered: int = yield SDKCall(
            "chat_with_mcp",
            {"command": command, "args": list(arguments), "model": model_spec, "max_steps": max_steps},
        )
    return answered
