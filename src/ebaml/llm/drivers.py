"""Run the pipelines of ``ebaml.llm.pipelines`` against ``baml_sdk``: the sync driver and the async driver.

A pipeline yields its steps (``ebaml.llm.sdk_steps``). The sync driver calls the ``<name>_sync`` function of
``baml_sdk`` for each step, and runs the graphs of a ``FanOut`` in a thread pool. The async driver awaits the function
``<name>``, and runs the graphs of a ``FanOut`` as tasks of one task group. This module is the one module of ``ebaml``
that imports ``baml_sdk``, so the pipelines and the public functions run in a plain unit test.
"""

import asyncio
import contextlib
import functools
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from types import ModuleType

import attrs

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

# the classes of baml_sdk that the pipelines build
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


_SYNC_DRIVER = _SyncDriver(baml_sdk)
_ASYNC_DRIVER = _AsyncDriver(baml_sdk)


def run_sync[R](pipeline: Generator[Step, SDKValue, R]) -> R:
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
    return _SYNC_DRIVER.run(pipeline)


async def run_async[R](pipeline: Generator[Step, SDKValue, R]) -> R:
    """Run a pipeline to its end with the async functions of ``baml_sdk``, in the running event loop.

    Parameters
    ----------
    pipeline: Generator[Step, SDKValue, R]
        The pipeline.

    Returns
    -------
    R
        The value that the pipeline returns.

    """
    return await _ASYNC_DRIVER.run(pipeline)
