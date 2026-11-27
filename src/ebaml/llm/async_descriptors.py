"""Describe EBM graphs and models using LLMs: the async form of each call.

Each function awaits the async functions of ``baml_sdk``, so an application that runs an event loop (a web server, an
agent framework, a notebook) serves other work while the LLM answers. The functions run the pipelines of
``ebaml.llm.pipelines``, as the sync functions of ``ebaml.llm.descriptors`` do, so both forms give the same result
for the same request.

The steps that need no LLM run in the event loop and block it while they run: the extraction and the text of the
graphs, the images of the vision mode, and the facts of a local explanation. These steps take less than a second for a
usual model.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Generator
from types import ModuleType

import attrs

from ebaml.llm.descriptors import SDK_MODULE, SDK_TYPES
from ebaml.llm.difficulties import DifficultyProposal
from ebaml.llm.pipelines import (
    EBMDescription,
    LocalExplanation,
    describe_ebm_pipeline,
    describe_graph_pipeline,
    explain_instance_pipeline,
    propose_difficulties_pipeline,
)
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.sdk_steps import Described, FanOut, FeatureOutcome, Outcome, PipelineRun, SDKCall, SDKValue, Step


@attrs.frozen
class _AsyncDriver:
    """Awaits the steps of a pipeline, and runs the graphs of a ``FanOut`` as tasks of one task group.

    Attributes
    ----------
    sdk: ModuleType
        The module whose async functions the driver awaits: ``baml_sdk``.

    """

    sdk: ModuleType

    async def run[R](self, pipeline: Generator[Step, SDKValue, R]) -> R:
        """Await each step of a pipeline, and send its outcome back, until the pipeline returns.

        Parameters
        ----------
        pipeline: Generator[Step, SDKValue, R]
            The pipeline to run in the event loop.

        Returns
        -------
        R
            The result of the pipeline.

        """
        with contextlib.closing(PipelineRun(pipeline)) as run:
            while (step := run.step) is not None:
                run.advance(await Outcome.of_awaitable(step.dispatch(self)))
            return run.result()

    def call(self, call: SDKCall) -> Awaitable[SDKValue]:
        """Call the async function of ``baml_sdk`` that the step names.

        Parameters
        ----------
        call: SDKCall
            The name of the function and its arguments.

        Returns
        -------
        Awaitable[SDKValue]
            The coroutine of the call.

        """
        function: Callable[..., Awaitable[SDKValue]] = getattr(self.sdk, call.name)
        return function(**call.arguments)

    async def fan_out(self, fan_out: FanOut) -> list[FeatureOutcome]:
        """Describe the graphs as tasks of one task group, with at most ``max_concurrency`` of them at the same time.

        Each task counts its graph in the thread of the event loop when it finishes, so the progress bar needs no lock.
        A graph that fails with an error of its feature only gives an outcome with the error. An error that stops the
        whole call cancels the task group, and the method raises that one error, not an ``ExceptionGroup``.

        Parameters
        ----------
        fan_out: FanOut
            The pipelines, the number of graphs at the same time, the counter of finished graphs, and the function that
            settles an error.

        Returns
        -------
        list[FeatureOutcome]
            The outcome of each graph, in the order of the pipelines.

        """
        limit = asyncio.Semaphore(fan_out.max_concurrency)

        async def describe(pipeline: Generator[Step, SDKValue, Described]) -> FeatureOutcome:
            async with limit:
                outcome = fan_out.feature_outcome(await Outcome.of_awaitable(self.run(pipeline)))
            fan_out.finish_step()
            return outcome

        tasks: list[asyncio.Task[FeatureOutcome]] = []
        stopped: list[Exception] = []
        try:
            async with asyncio.TaskGroup() as group:
                tasks.extend(group.create_task(describe(pipeline)) for pipeline in fan_out.pipelines)
        except ExceptionGroup as errors:
            stopped.extend(errors.exceptions)
        if stopped:
            # raised outside the handler, so the error keeps its own cause and its fix steps lead the traceback
            raise stopped[0]
        return [task.result() for task in tasks]


_DRIVER = _AsyncDriver(SDK_MODULE)


async def describe_graph_async(request: DescribeGraphRequest) -> str:
    """Ask the LLM to describe a graph, and await the reply. The async form of ``describe_graph``.

    The function sends the same requests as ``describe_graph`` and returns the same description. The extraction and the
    text of the graph run in the event loop and block it while they run.

    Parameters
    ----------
    request: DescribeGraphRequest
        The request, as ``describe_graph`` takes it.

    Returns
    -------
    str
        The text that ``describe_graph`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import DescribeGraphRequest, describe_graph_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> print(asyncio.run(describe_graph_async(DescribeGraphRequest(ebm, feature_index=0))))
    A fake reply, which stands in for the text of the LLM.

    """
    return await _DRIVER.run(describe_graph_pipeline(request, SDK_TYPES))


async def describe_ebm_async(request: DescribeEBMRequest) -> EBMDescription:
    """Ask the LLM to describe an EBM, and await the replies. The async form of ``describe_ebm``.

    The function describes the graphs as tasks of one task group, with at most ``max_concurrency`` requests at the same
    time, and then asks for the summary. It returns the same description as ``describe_ebm``, with the same partial
    results: a feature whose request fails on its own goes into ``failed_features``, and an error that affects every
    request stops the call at once. A cancelled call cancels the requests of the graphs that have not finished. The
    extraction and the text of the graphs run in the event loop and block it while they run.

    Parameters
    ----------
    request: DescribeEBMRequest
        The request, as ``describe_ebm`` takes it. ``max_concurrency`` limits the requests in flight, and
        ``progress_bar`` shows one step for each feature and one for the summary.

    Returns
    -------
    EBMDescription
        The description that ``describe_ebm`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import DescribeEBMRequest, describe_ebm_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> description = asyncio.run(describe_ebm_async(DescribeEBMRequest(ebm, max_concurrency=2)))
    >>> print(description.text)
    A fake reply, which stands in for the text of the LLM.
    >>> description.failed_features
    []

    """
    return await _DRIVER.run(describe_ebm_pipeline(request, SDK_TYPES))


async def explain_instance_async(request: ExplainInstanceRequest) -> LocalExplanation:
    """Explain one prediction of an EBM, and await the reply. The async form of ``explain_instance``.

    The function computes the same facts and sends the same request as ``explain_instance``. The facts are computed in
    the event loop and block it while they run.

    Parameters
    ----------
    request: ExplainInstanceRequest
        The request, as ``explain_instance`` takes it.

    Returns
    -------
    LocalExplanation
        The explanation that ``explain_instance`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import ExplainInstanceRequest, explain_instance_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> request = ExplainInstanceRequest(ebm, X.iloc[0].to_dict(), "The model predicts survival.")
    >>> print(asyncio.run(explain_instance_async(request)).counterfactual)
    A fake reply, which stands in for the text of the LLM.

    """
    return await _DRIVER.run(explain_instance_pipeline(request, SDK_TYPES))


async def propose_difficulties_async(request: ProposeDifficultiesRequest) -> DifficultyProposal:
    """Ask the LLM how hard each change of an input is, and await the reply. The async form of ``propose_difficulties``.

    The function sends the same request as ``propose_difficulties`` and returns the same proposal.

    Parameters
    ----------
    request: ProposeDifficultiesRequest
        The request, as ``propose_difficulties`` takes it.

    Returns
    -------
    DifficultyProposal
        The proposal that ``propose_difficulties`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import ProposeDifficultiesRequest, propose_difficulties_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> proposal = asyncio.run(propose_difficulties_async(ProposeDifficultiesRequest(ebm, X.iloc[0].to_dict())))
    >>> [(item.feature, item.difficulty.value) for item in proposal.difficulties]
    [('fare', 'Easy'), ('age', 'VeryHard')]

    """
    return await _DRIVER.run(propose_difficulties_pipeline(request, SDK_TYPES))
