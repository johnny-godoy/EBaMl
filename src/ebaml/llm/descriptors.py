"""Describe EBM graphs and models using LLMs: the sync form of each call.

The prompts and the chat conversations live in ``baml_src/talk_to_ebm.baml``. The pipelines of
``ebaml.llm.pipelines`` extract the graphs from the EBM, convert them to text and say which ``baml_sdk`` functions to
call. This module drives them: it calls the ``<name>_sync`` function of ``baml_sdk`` for each step. Each function takes
one request from ``ebaml.llm.requests``, and the ``model`` field of the request chooses the LLM of the call.
"""

import contextlib
import functools
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from types import ModuleType

import attrs

from ebaml.llm.difficulties import DifficultyProposal
from ebaml.llm.options import ModelOptions, PromptOptions
from ebaml.llm.pipelines import (
    EBMDescription,
    LocalExplanation,
    describe_ebm_pipeline,
    describe_graph_pipeline,
    explain_instance_pipeline,
    model_spec_pipeline,
    prompt_options_pipeline,
    propose_difficulties_pipeline,
)
from ebaml.llm.provider_errors import provider_errors
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.sdk_setup import ensure_sdk
from ebaml.llm.sdk_steps import (
    Described,
    FanOut,
    FeatureOutcome,
    Outcome,
    PipelineRun,
    SDKCall,
    SDKTypes,
    SDKValue,
    Step,
)

# `baml_sdk` is generated and gitignored, so a checkout may lack it. Generate it before the import, or explain the fix.
ensure_sdk()

import baml_sdk  # ruff: ignore[module-import-not-at-top-of-file] -- ensure_sdk must run first

# the default of the option argument, as a module-level value that nothing changes: the option class is frozen
_DEFAULT_PROMPT_OPTIONS = PromptOptions()
# the generated module, for the async driver of ebaml.llm.async_descriptors: this module is the one that imports it
SDK_MODULE: ModuleType = baml_sdk
SDK_TYPES: SDKTypes = SDKTypes(model_spec=baml_sdk.ModelSpec, local_fact_texts=baml_sdk.LocalFactTexts)


@attrs.frozen
class _SyncDriver:
    """Runs the steps of a pipeline in the thread of the caller, and the graphs of a ``FanOut`` in a thread pool.

    Attributes
    ----------
    sdk: ModuleType
        The module whose ``<name>_sync`` functions the driver calls: ``baml_sdk``.

    """

    sdk: ModuleType

    def run[R](self, pipeline: Generator[Step, SDKValue, R]) -> R:
        """Run a pipeline to its end.

        Parameters
        ----------
        pipeline: Generator[Step, SDKValue, R]
            The pipeline.

        Returns
        -------
        R
            The value that the pipeline returns.

        """
        with contextlib.closing(PipelineRun(pipeline)) as run:
            while (step := run.step) is not None:
                run.advance(Outcome.of(functools.partial(step.dispatch, self)))
            return run.result()

    def call(self, call: SDKCall) -> SDKValue:
        """Call the ``<name>_sync`` function of ``baml_sdk``.

        Parameters
        ----------
        call: SDKCall
            The name of the function and its arguments.

        Returns
        -------
        SDKValue
            The reply.

        """
        return getattr(self.sdk, f"{call.name}_sync")(**call.arguments)

    def fan_out(self, fan_out: FanOut) -> list[FeatureOutcome]:
        """Describe the graphs at the same time, and count each one that finishes.

        The method counts the graphs in the thread of the caller, in the order in which they finish, because the
        progress bar is not thread safe. It returns the outcomes in the order of the pipelines. A graph that fails with
        an error of its feature only gives an outcome with the error. An error that stops the whole call cancels the
        graphs that have not started, and the method raises it.

        Parameters
        ----------
        fan_out: FanOut
            The pipelines, the number of graphs at the same time, and the counter of finished graphs.

        Returns
        -------
        list[FeatureOutcome]
            The outcome of each graph, in the order of the pipelines.

        """
        with ThreadPoolExecutor(max_workers=fan_out.max_concurrency) as pool:
            futures: list[Future[Described]] = [pool.submit(self.run, pipeline) for pipeline in fan_out.pipelines]
            outcomes: dict[Future[Described], FeatureOutcome] = {}
            try:
                for finished in as_completed(futures):
                    outcomes[finished] = fan_out.feature_outcome(Outcome.of(finished.result))
                    fan_out.finish_step()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            return [outcomes[future] for future in futures]


_DRIVER = _SyncDriver(baml_sdk)


def run_pipeline[R](pipeline: Generator[Step, SDKValue, R]) -> R:
    """Run a pipeline to its end with the sync functions of ``baml_sdk``.

    Parameters
    ----------
    pipeline: Generator[Step, SDKValue, R]
        The pipeline.

    Returns
    -------
    R
        The value that the pipeline returns.

    """
    return _DRIVER.run(pipeline)


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
    return run_pipeline(describe_graph_pipeline(request, SDK_TYPES))


def sample_graph_tokens(
    graph: str,
    model: ModelOptions | str | None = None,
    *,
    prompt: PromptOptions = _DEFAULT_PROMPT_OPTIONS,
) -> list[baml_sdk.TokenSample]:
    """Send one request with the text of a graph and return its token samples.

    The request is the first turn of ``describe_graph``. Each sample has the body of the request and the input tokens
    that the provider counted. Set ``max_output_tokens`` in the model options to a small number, such as 16, to keep the
    cost low: a reply that the limit cuts off is not an error here. ``scripts/llm/token_weights.py`` uses the function
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
    model_spec = run_pipeline(model_spec_pipeline(model, SDK_TYPES))
    opts = run_pipeline(prompt_options_pipeline(prompt))
    with provider_errors():
        return baml_sdk.sample_graph_tokens_sync(graph=graph, opts=opts, model=model_spec)


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
        The arguments of the program, for example ``["--model", "model.skops", "--data", "train.parquet"]``.
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
    model_spec = run_pipeline(model_spec_pipeline(model, SDK_TYPES))
    with provider_errors():
        return baml_sdk.chat_with_mcp_sync(command=command, args=list(arguments), model=model_spec, max_steps=max_steps)


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
    return run_pipeline(describe_ebm_pipeline(request, SDK_TYPES))


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
    return run_pipeline(explain_instance_pipeline(request, SDK_TYPES))


def propose_difficulties(request: ProposeDifficultiesRequest) -> DifficultyProposal:
    """Ask the LLM how hard the change of each input is for the person of an instance, as a cost strategy to check.

    The reply rates each input on the scale of ``Difficulty``, with a reason. The function runs no search: read the
    proposal, correct it if the LLM is wrong, and pass ``proposal.costs`` to ``RecourseOptions(costs=...)``. The facts
    then say that a language model proposed the difficulties, because they are assumptions.

    Parameters
    ----------
    request: ProposeDifficultiesRequest
        The EBM, the instance, the base cost strategy, the prompts and the model.

    Returns
    -------
    DifficultyProposal
        The difficulty and the reason of each input that the reply rates, and the ``ScaledCost`` that they give.

    """
    return run_pipeline(propose_difficulties_pipeline(request, SDK_TYPES))
