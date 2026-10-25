"""Tests for the truncation retry policy."""

from collections.abc import Callable
from typing import Any

import pytest
from structlog.testing import capture_logs

from ebaml.core.exceptions import EBMGraphSimplificationError, EBMGraphTokenLimitError
from ebaml.llm import retry


class CutOffError(Exception):
    """The reply hit the token limit."""


class OtherError(Exception):
    """Any other failure of the LLM call."""


def _is_cut_off(error: CutOffError) -> bool:
    return str(error) == "cut"


def _run(
    call: Callable[[str], str],
    graph: str,
    rebuild: Callable[[int], str],
    estimate: Callable[[str], int] = len,
) -> str:
    return retry.call_with_shrinking_graph(
        call,
        graph,
        error_type=CutOffError,
        is_truncated=_is_cut_off,
        estimate_tokens=estimate,
        rebuild=rebuild,
        label="f0",
    )


def test_returns_the_first_reply_without_a_rebuild() -> None:
    def rebuild(_budget: int) -> str:
        pytest.fail("no rebuild is expected")

    assert _run(lambda graph: graph.upper(), "abc", rebuild) == "ABC"


def test_asks_again_with_three_quarters_of_the_estimate_rounded_down() -> None:
    sent = []
    budgets = []

    def call(graph: str) -> str:
        sent.append(graph)
        if graph == "long":
            msg = "cut"
            raise CutOffError(msg)
        return "ok"

    def rebuild(budget: int) -> str:
        budgets.append(budget)
        return "short"

    # an estimate of 301 gives 225.75, so the budget is 225
    assert _run(call, "long", rebuild, estimate=lambda _text: 301) == "ok"
    assert sent == ["long", "short"]
    assert budgets == [225]


def test_estimates_the_tokens_of_the_graph_that_failed() -> None:
    graphs = iter(["g1", "g2", "g3"])
    estimated = []

    def call(graph: str) -> str:
        if graph != "g3":
            msg = "cut"
            raise CutOffError(msg)
        return "ok"

    def estimate(text: str) -> int:
        estimated.append(text)
        return 100

    assert _run(call, "g0", lambda _budget: next(graphs), estimate=estimate) == "ok"
    assert estimated == ["g0", "g1", "g2"]


def test_stops_after_four_retries_and_logs_each() -> None:
    graphs = iter(f"g{index}" for index in range(1, 10))
    sent = []

    def call(graph: str) -> str:
        sent.append(graph)
        msg = "cut"
        raise CutOffError(msg)

    with capture_logs() as logs, pytest.raises(CutOffError):
        _run(call, "g0", lambda _budget: next(graphs))
    assert sent == ["g0", "g1", "g2", "g3", "g4"]
    assert [(log["event"], log["feature"], log["retry"], log["log_level"]) for log in logs] == [
        ("truncated_reply", "f0", retry_number, "warning") for retry_number in (1, 2, 3, 4)
    ]


def test_raises_an_error_that_is_not_a_truncation_at_once() -> None:
    sent = []

    def call(graph: str) -> str:
        sent.append(graph)
        msg = "refused"
        raise CutOffError(msg)

    with capture_logs() as logs, pytest.raises(CutOffError, match="refused"):
        _run(call, "g0", lambda _budget: "never")
    assert sent == ["g0"]
    assert logs == []


def test_does_not_catch_other_error_types() -> None:
    def call(_graph: str) -> str:
        msg = "cut"
        raise OtherError(msg)

    with pytest.raises(OtherError):
        _run(call, "g0", lambda _budget: "never")


@pytest.mark.parametrize("failure", [EBMGraphSimplificationError, EBMGraphTokenLimitError])
def test_raises_the_llm_error_when_the_graph_cannot_shrink(failure: type[Exception]) -> None:
    original = CutOffError("cut")

    def call(_graph: str) -> str:
        raise original

    def rebuild(_budget: int) -> str:
        msg = "too complex"
        raise failure(msg)

    with pytest.raises(CutOffError) as caught:
        _run(call, "g0", rebuild)
    assert caught.value is original
    assert caught.value.__suppress_context__ is True


def test_does_not_hide_other_errors_of_the_rebuild() -> None:
    def call(_graph: str) -> str:
        msg = "cut"
        raise CutOffError(msg)

    def rebuild(_budget: int) -> str:
        msg = "bug"
        raise KeyError(msg)

    with pytest.raises(KeyError):
        _run(call, "g0", rebuild)


def test_takes_the_dependencies_by_keyword_only() -> None:
    untyped: Any = retry.call_with_shrinking_graph
    with pytest.raises(TypeError):
        untyped(str, "g0", CutOffError, _is_cut_off, len, str, "f0")
