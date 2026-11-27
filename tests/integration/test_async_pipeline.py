"""The async form of the LLM calls, through the BAML runtime and a local LLM server.

The async functions await the async functions of ``baml_sdk``. These tests check what crosses the boundary: the number
of requests, the turns of a conversation, the requests in flight at the same time, and what a cancelled call sends.
"""

import asyncio
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ebaml import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ModelOptions,
    RateLimitedError,
    describe_ebm_async,
    describe_graph,
    describe_graph_async,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.unit.ebaml.conftest import PandasFrame

GRAPH_TURNS = 3
SUMMARY_TURNS = 2
DELAY_SECONDS = 0.05
# a burst of connections, and the wait of a client whose connection the kernel dropped
BURST = 64
RETRY_SECONDS = 0.9


class InFlight:
    """A handler of the local server that waits before each reply, and records the most requests in flight at once."""

    def __init__(self, delay: float = DELAY_SECONDS) -> None:
        """Wait ``delay`` seconds before each reply."""
        self.delay: float = delay
        self.peak: int = 0
        self.first_request: threading.Event = threading.Event()
        self._now = 0
        self._lock = threading.Lock()

    def __call__(self, _request: RecordedRequest) -> Reply:
        """Count the request while it waits, and answer it.

        Returns
        -------
        Reply
            The default reply.

        """
        with self._lock:
            self._now += 1
            self.peak = max(self.peak, self._now)
        self.first_request.set()
        time.sleep(self.delay)
        with self._lock:
            self._now -= 1
        return Reply()


def test_describe_ebm_async_sends_three_requests_for_each_graph_and_two_for_the_summary(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    asyncio.run(describe_ebm_async(describe_request))
    feature_count = len(describe_request.ebm.term_names_)
    assert len(fake_llm.requests) == feature_count * GRAPH_TURNS + SUMMARY_TURNS


def test_describe_graph_async_sends_the_three_turns_of_the_conversation(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    asyncio.run(describe_graph_async(DescribeGraphRequest(ebm, feature_index=0, model=llm_model)))
    assert [request.turns for request in fake_llm.requests] == [1, 2, 3]


def test_describe_ebm_async_keeps_at_most_max_concurrency_requests_in_flight(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    handler = InFlight()
    fake_llm.handler = handler
    asyncio.run(
        describe_ebm_async(DescribeEBMRequest(describe_request.ebm, model=describe_request.model, max_concurrency=2)),
    )
    # three graphs, two at a time: the requests overlap, and never more than two of them
    assert handler.peak == 2


def test_describe_ebm_async_runs_the_graphs_at_the_same_time(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    handler = InFlight()
    fake_llm.handler = handler
    asyncio.run(describe_ebm_async(describe_request))
    assert handler.peak == len(describe_request.ebm.term_names_)


async def _cancel_after_first_request(request: DescribeEBMRequest, handler: InFlight) -> None:
    task = asyncio.get_running_loop().create_task(describe_ebm_async(request))
    await asyncio.to_thread(handler.first_request.wait)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # a reply takes two delays, so a conversation that went on would send its second turn in this wait
    await asyncio.sleep(handler.delay * 5)
    assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]


def test_a_cancelled_describe_ebm_async_sends_no_further_turn(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    handler = InFlight(delay=DELAY_SECONDS * 2)
    fake_llm.handler = handler
    asyncio.run(_cancel_after_first_request(describe_request, handler))
    # the first turns of the graphs left before the cancel; no second turn and no summary followed them
    assert {request.turns for request in fake_llm.requests} == {1}
    assert len(fake_llm.requests) <= len(describe_request.ebm.term_names_)


def test_a_feature_that_the_provider_rate_limits_is_named_and_the_others_are_described(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.handler = lambda request: Reply(status=429) if "Feature Name: fare" in request.prompt else Reply()
    description = asyncio.run(describe_ebm_async(describe_request))
    assert [(failure.feature, type(failure.error)) for failure in description.failed_features] == [
        ("fare", RateLimitedError),
    ]


def test_the_summary_request_names_the_feature_that_failed(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.handler = lambda request: Reply(status=429) if "Feature Name: fare" in request.prompt else Reply()
    asyncio.run(describe_ebm_async(describe_request))
    summary_prompts = [prompt for prompt in fake_llm.prompts if "<graph_descriptions>" in prompt]
    assert summary_prompts
    assert all(
        "these features failed, so no graph description covers them: fare." in prompt for prompt in summary_prompts
    )


async def _describe_in_a_running_loop(request: DescribeGraphRequest) -> str:
    await asyncio.sleep(0)
    return describe_graph(request)


def test_the_sync_form_works_inside_a_running_event_loop(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    # the sync calls of baml_sdk run in a running loop (Jupyter, marimo): they block the loop, and they do not fail
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(text="The graph.")
    request = DescribeGraphRequest(ebm, feature_index=0, model=llm_model)
    assert asyncio.run(_describe_in_a_running_loop(request)) == "The graph."


def test_the_fake_llm_accepts_a_burst_of_connections_without_a_retry_of_one_second(fake_llm: FakeLLM) -> None:
    # A listen queue of 5 dropped connections in a burst, and each dropped client waited one second to try again.
    host, port = fake_llm.base_url.removeprefix("http://").removesuffix("/v1").split(":")
    start = threading.Barrier(BURST)

    def connect_seconds() -> float:
        start.wait()
        began = time.perf_counter()
        with socket.create_connection((host, int(port)), timeout=5):
            return time.perf_counter() - began

    with ThreadPoolExecutor(max_workers=BURST) as pool:
        waits = [future.result() for future in [pool.submit(connect_seconds) for _ in range(BURST)]]
    assert max(waits) < RETRY_SECONDS
