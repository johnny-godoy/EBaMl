"""Tests for the steps of a pipeline and the run that moves a pipeline from step to step."""

from collections.abc import Generator
from typing import Any

import attrs
import pytest

from ebaml.llm.sdk_steps import Described, FanOut, PipelineRun, SDKCall, Step


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


def test_a_call_asks_the_driver_to_call_the_function() -> None:
    driver = RecordingDriver()
    assert SDKCall("summarize_ebm", {}).dispatch(driver) == "reply"
    assert driver.calls == ["call summarize_ebm"]


def test_a_fan_out_asks_the_driver_to_run_the_pipelines() -> None:
    driver = RecordingDriver()
    assert FanOut([], 2, lambda: None).dispatch(driver) == []
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
    run.reply("one")
    assert run.step == SDKCall("second", {})
    assert log == ["got one"]


def test_an_error_that_the_pipeline_handles_moves_the_run_on() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    run.reply("one")
    run.fail(KeyError("missing"))
    assert run.step is None
    assert run.result() == "done after one"
    assert log == ["got one", "handled missing", "closed"]


def test_an_error_that_the_pipeline_does_not_handle_leaves_the_run() -> None:
    log: list[str] = []
    run = PipelineRun(_two_steps(log))
    with pytest.raises(ValueError, match="bad reply"):
        run.fail(ValueError("bad reply"))
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
