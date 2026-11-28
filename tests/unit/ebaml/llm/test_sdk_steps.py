"""Tests for the steps of a pipeline and the run that moves a pipeline from step to step."""

import asyncio
from collections.abc import Generator
from typing import Any

import attrs
import pytest

import baml_sdk
from ebaml.core.exceptions import LLMRequestError, RateLimitedError
from ebaml.llm.sdk_steps import Described, FanOut, FeatureOutcome, LocalWork, Outcome, PipelineRun, SDKCall, Step

_COUNTED: baml_sdk.Counted = baml_sdk.Counted(
    text="t",
    usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=1, samples=[]),
)


@attrs.define
class RecordingDriver:
    """A driver that records which method each step calls."""

    calls: list[str] = attrs.field(factory=list)

    def call(self, call: SDKCall) -> str:
        """Record the call.

        Returns
        -------
        str
            A fixed reply.

        """
        self.calls.append(f"call {call.name}")
        return "reply"

    def fan_out(self, fan_out: FanOut) -> list[Described]:
        """Record the fan-out.

        Returns
        -------
        list[Described]
            No results.

        """
        self.calls.append(f"fan_out {len(fan_out.pipelines)}")
        return []

    def compute(self, work: LocalWork) -> str:
        """Record the work, and run it.

        Returns
        -------
        str
            The result of the work.

        """
        self.calls.append(f"compute {work.name}")
        result: str = work.run()
        return result


def _settle_nothing(error: Exception) -> LLMRequestError:
    raise error


def test_a_call_asks_the_driver_to_call_the_function() -> None:
    driver = RecordingDriver()
    assert SDKCall("summarize_ebm", {}).dispatch(driver) == "reply"
    assert driver.calls == ["call summarize_ebm"]


def test_local_work_asks_the_driver_to_run_it() -> None:
    driver = RecordingDriver()
    assert LocalWork("facts", lambda: "facts").dispatch(driver) == "facts"
    assert driver.calls == ["compute facts"]


def test_a_fan_out_asks_the_driver_to_run_the_pipelines() -> None:
    driver = RecordingDriver()
    assert FanOut([], 2, lambda: None, _settle_nothing).dispatch(driver) == []
    assert driver.calls == ["fan_out 0"]


def _two_steps(log: list[str]) -> Generator[Step, Any, str]:
    try:
        first = yield SDKCall("first", {})
        log.append(f"got {first}")
        try:
            yield SDKCall("second", {})
        except KeyError as error:
            log.append(f"handled {error.args[0]}")
        return f"done after {first}"
    finally:
        log.append("closed")


def test_a_run_stops_at_the_first_step() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    assert run.step == SDKCall("first", {})
    assert not log


def test_a_reply_moves_the_run_to_the_next_step() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    run.advance(Outcome(reply="one"))
    assert run.step == SDKCall("second", {})
    assert log == ["got one"]


def test_an_error_that_the_pipeline_handles_moves_the_run_on() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    run.advance(Outcome(reply="one"))
    run.advance(Outcome(error=KeyError("missing")))
    assert run.step is None
    assert run.result() == "done after one"
    assert log == ["got one", "handled missing", "closed"]


def test_an_error_that_the_pipeline_does_not_handle_leaves_the_run() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    with pytest.raises(ValueError, match="bad reply"):
        run.advance(Outcome(error=ValueError("bad reply")))
    assert log == ["closed"]


def test_a_closed_run_closes_its_pipeline() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    run.close()
    assert log == ["closed"]


def test_a_pipeline_with_no_step_returns_at_once() -> None:
    def no_step() -> Generator[Step, Any, int]:
        yield from ()
        return 3

    run = PipelineRun(no_step())
    assert run.step is None
    assert run.result() == 3


def test_an_outcome_keeps_the_reply_of_a_step() -> None:
    assert Outcome.of(lambda: "reply") == Outcome(reply="reply")


def test_an_outcome_keeps_the_error_of_a_step() -> None:
    error = KeyError("missing")

    def fail() -> str:
        raise error

    assert Outcome.of(fail) == Outcome(error=error)


async def _reply() -> str:
    await asyncio.sleep(0)
    return "reply"


async def _fail() -> str:
    await asyncio.sleep(0)
    msg = "bad"
    raise ValueError(msg)


def test_an_outcome_of_an_awaitable_keeps_its_reply() -> None:
    assert asyncio.run(Outcome.of_awaitable(_reply())) == Outcome(reply="reply")


def test_an_outcome_of_an_awaitable_keeps_its_error() -> None:
    outcome = asyncio.run(Outcome.of_awaitable(_fail()))
    assert isinstance(outcome.error, ValueError)


def test_a_fan_out_keeps_the_description_of_a_pipeline() -> None:
    described = Described(counted=_COUNTED, truncated_attempts=0)
    fan_out = FanOut([], 1, lambda: None, _settle_nothing)
    assert fan_out.feature_outcome(Outcome(reply=described)) == FeatureOutcome(described=described)


def test_a_fan_out_settles_the_error_of_a_pipeline() -> None:
    feature_error = RateLimitedError("slow down")
    fan_out = FanOut([], 1, lambda: None, lambda _error: feature_error)
    assert fan_out.feature_outcome(Outcome(error=KeyError("x"))) == FeatureOutcome(error=feature_error)


def test_a_fan_out_raises_an_error_that_its_settle_raises() -> None:
    fan_out = FanOut([], 1, lambda: None, _settle_nothing)
    with pytest.raises(KeyError):
        fan_out.feature_outcome(Outcome(error=KeyError("x")))
