"""Tests for the sync and the async driver of the pipelines."""

import ast
import asyncio
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import baml_sdk
from ebaml.llm import drivers, ebm_description, pipelines
from ebaml.llm.sdk_steps import Described, FanOut, LocalWork
from tests.unit.ebaml.conftest import sdk_function_names, sdk_signature


def _no_usage() -> baml_sdk.TokenUsage:
    return baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[])


def _described_after(
    text: str,
    delay: float,
    started: list[str] | None = None,
) -> Generator[Any, Any, Described]:
    """Build a pipeline that yields no step and describes one graph after a delay.

    Returns
    -------
    Described
        A description with the text.

    Yields
    ------
    Any
        Nothing: the pipeline yields from an empty tuple and returns.

    Raises
    ------
    RuntimeError
        If the text is ``"fail"``.

    """
    if started is not None:
        started.append(text)
    if text == "fail":
        msg = "first graph failed"
        raise RuntimeError(msg)
    time.sleep(delay)
    yield from ()
    return Described(baml_sdk.Counted(text=text, usage=_no_usage()), truncated_attempts=0)


def test_fan_out_returns_the_descriptions_in_the_order_of_the_pipelines() -> None:
    finished = []
    # the first graph finishes last
    fan_out = FanOut(
        [_described_after("d0", 0.1), _described_after("d1", 0)],
        2,
        lambda: finished.append(len(finished)),
        ebm_description.feature_error,
    )
    described = drivers.run_sync(_yield_step(fan_out))
    assert [item.described.counted.text for item in described if item.described is not None] == ["d0", "d1"]
    assert finished == [0, 1]


def test_fan_out_cancels_the_graphs_that_have_not_started_after_a_failure() -> None:
    started: list[str] = []
    # the second graph runs while the driver cancels the third, which has not started
    fan_out = FanOut(
        [_described_after(text, 0.2, started) for text in ("fail", "b", "c")],
        1,
        lambda: None,
        ebm_description.feature_error,
    )
    with pytest.raises(RuntimeError, match="first graph failed"):
        drivers.run_sync(_yield_step(fan_out))
    assert "c" not in started


def _yield_step(step: FanOut) -> Generator[Any, Any, Any]:
    """Build a pipeline that yields one step and returns its reply.

    Returns
    -------
    Any
        The reply of the step.

    Yields
    ------
    FanOut
        The step.

    """
    return (yield step)


def test_the_async_driver_gives_the_same_outcomes_as_the_sync_driver() -> None:
    def fan_out() -> FanOut:
        return FanOut(
            [_described_after("d0", 0.05), _described_after("d1", 0)],
            2,
            lambda: None,
            ebm_description.feature_error,
        )

    sync_outcomes = drivers.run_sync(_yield_step(fan_out()))
    async_outcomes = asyncio.run(drivers.run_async(_yield_step(fan_out())))
    assert sync_outcomes == async_outcomes


def test_the_async_driver_raises_the_one_error_that_stops_the_call() -> None:
    fan_out = FanOut(
        [_described_after(text, 0) for text in ("fail", "b")],
        1,
        lambda: None,
        ebm_description.feature_error,
    )
    with pytest.raises(RuntimeError, match="first graph failed"):
        asyncio.run(drivers.run_async(_yield_step(fan_out)))


def _step_names() -> set[str]:
    """Read the name of each ``SDKCall`` step of the pipelines from the source.

    Returns
    -------
    set[str]
        The names of the BAML functions.

    """
    tree = ast.parse(Path(pipelines.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Call(func=ast.Name(id="SDKCall"), args=[ast.Constant(value=str(name)), *_]):
                names.add(name)
            case _:
                pass
    return names


def test_each_step_has_a_plain_sync_function_and_an_async_function() -> None:
    # A toolchain newer than the baml-bridge of uv.lock writes `async def` for every function. The fakes of the other
    # tests return a value at once, so only this check sees that a call of the sync driver would return a coroutine.
    names = _step_names()
    assert names, "the pipelines name no function of baml_sdk"
    for name in sorted(names):
        sdk_signature(f"{name}_sync")
        assert name in sdk_function_names()


def _computes(work: LocalWork) -> Generator[Any, Any, str]:
    result: str = yield work
    return result


def test_the_sync_driver_runs_local_work_in_the_thread_of_the_caller() -> None:
    caller = threading.get_ident()
    work = LocalWork("thread", lambda: "same" if threading.get_ident() == caller else "other")
    assert drivers.run_sync(_computes(work)) == "same"


def test_the_async_driver_runs_local_work_in_another_thread_while_the_event_loop_runs() -> None:
    def slow() -> str:
        time.sleep(0.2)
        return "facts"

    async def both() -> tuple[str, int]:
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(tick())
        try:
            result = await drivers.run_async(_computes(LocalWork("slow", slow)))
        finally:
            ticker.cancel()
        return result, ticks

    result, ticks = asyncio.run(both())
    assert result == "facts"
    assert ticks >= 5, "the event loop must keep running while the work runs"


def test_an_error_of_local_work_reaches_the_caller_of_each_driver() -> None:
    def broken() -> str:
        msg = "broken work"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="broken work"):
        drivers.run_sync(_computes(LocalWork("broken", broken)))
    with pytest.raises(ValueError, match="broken work"):
        asyncio.run(drivers.run_async(_computes(LocalWork("broken", broken))))
