"""The options of ``describe_ebm``, ``describe_graph`` and ``explain_instance`` must reach the LLM or the facts.

Each test answers one adversarial question: if the code dropped this option in silence, would a test notice?
"""

import re
import threading
import time

import pytest
import structlog.testing

from ebaml import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    LocalOptions,
    ModelOptions,
    PromptOptions,
    RateLimitedError,
    compute_local_facts,
    describe_ebm,
    describe_graph,
    explain_instance,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.docs.conftest import training_data
from tests.unit.ebaml.conftest import PandasFrame

FEATURE_LINE: re.Pattern[str] = re.compile(r"Feature Name: (\w+)")
INSTANCE: dict[str, float] = {"age": 20.0, "fare": 50.0, "siblings": 1.0}

type Fitted = tuple[PandasFrame, ExplainableBoostingClassifier]


def _is_summary(request: RecordedRequest) -> bool:
    return FEATURE_LINE.search(request.prompt) is None


class _InFlight:
    """Count the requests that the server holds at the same time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __call__(self, _request: RecordedRequest) -> Reply:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.2)
        with self.lock:
            self.now -= 1
        return Reply()


@pytest.mark.parametrize("concurrency", [1, 2])
def test_the_bound_on_concurrency_is_the_peak_of_requests_in_flight(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    concurrency: int,
) -> None:
    _, ebm = binary_data_and_model
    in_flight = _InFlight()
    fake_llm.handler = in_flight
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, max_concurrency=concurrency, num_sentences=3))
    assert in_flight.peak == concurrency


def test_the_summary_lists_each_graph_description_under_its_own_feature_in_the_order_of_the_features(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model

    def answer(request: RecordedRequest) -> Reply:
        found = FEATURE_LINE.search(request.prompt)
        return Reply(text=f"described-{found.group(1)}" if found else "The summary.")

    fake_llm.handler = answer
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, num_sentences=3))
    summaries = [request.prompt for request in fake_llm.requests if _is_summary(request)]
    assert summaries
    lines = [f"{name}: described-{name}" for name in ebm.feature_names_in_]
    for prompt in summaries[:1]:
        positions = [prompt.index(line) for line in lines]
        assert positions == sorted(positions)


def test_the_number_of_sentences_reaches_the_summary_and_the_graphs_keep_their_own(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, num_sentences=4))
    summaries = [request.prompt for request in fake_llm.requests if _is_summary(request)]
    graphs = [request.prompt for request in fake_llm.requests if not _is_summary(request)]
    assert any("at most 4 sentences" in prompt for prompt in summaries)
    assert not any("at most 4 sentences" in prompt for prompt in graphs)
    assert any("at most 7 sentences" in prompt for prompt in graphs)


def test_describe_graph_describes_the_feature_that_the_index_names_and_uses_the_sentence_count(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_graph(DescribeGraphRequest(ebm, feature_index=1, model=llm_model, num_sentences=2))
    names = {
        found.group(1) for request in fake_llm.requests for found in [FEATURE_LINE.search(request.prompt)] if found
    }
    assert names == {ebm.feature_names_in_[1]}
    assert "at most 2 sentences" in fake_llm.requests[-1].prompt


def test_the_prompt_options_reach_the_prompts(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    prompt = PromptOptions(
        expert_description="EXPERT-OF-THE-TESTS",
        graph_description="GRAPH-WORDS-OF-THE-TESTS",
        dataset_description="DATASET-OF-THE-TESTS",
        task_description="TASK-OF-THE-TESTS",
    )
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, prompt=prompt, num_sentences=3))
    prompts = "\n".join(fake_llm.prompts)
    for words in ("EXPERT-OF-THE-TESTS", "GRAPH-WORDS-OF-THE-TESTS", "DATASET-OF-THE-TESTS", "TASK-OF-THE-TESTS"):
        assert words in prompts, f"{words} never reached a prompt"


def test_the_summary_names_the_interaction_terms_that_it_leaves_out(llm_model: ModelOptions, fake_llm: FakeLLM) -> None:
    features, target = training_data()
    ebm = ExplainableBoostingClassifier(interactions=2, random_state=0).fit(features, target)
    description = describe_ebm(DescribeEBMRequest(ebm, model=llm_model, num_sentences=3))
    summaries = [request.prompt for request in fake_llm.requests if _is_summary(request)]
    assert description.skipped_terms
    for prompt in summaries:
        assert "<omitted_parts>" in prompt
        assert all(term in prompt for term in description.skipped_terms)


def test_the_summary_of_a_complete_model_has_no_omitted_parts_block(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    describe_ebm(describe_request)
    assert not any("no graph description covers them" in prompt for prompt in fake_llm.prompts)


def test_the_summary_names_a_feature_without_a_main_effect(llm_model: ModelOptions, fake_llm: FakeLLM) -> None:
    features, target = training_data()
    ebm = ExplainableBoostingClassifier(interactions=0, random_state=0, exclude=["fare"]).fit(features, target)
    description = describe_ebm(DescribeEBMRequest(ebm, model=llm_model, num_sentences=3))
    summaries = [request.prompt for request in fake_llm.requests if _is_summary(request)]
    assert description.skipped_features == ["fare"]
    assert summaries
    covered = [prompt for prompt in summaries if "no graph description covers them: fare" in prompt]
    assert covered, "no summary request names the feature that has no main effect"


def test_the_usage_of_describe_ebm_adds_the_graphs_and_the_summary(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.handler = lambda _request: Reply(usage=(7, 5))
    with structlog.testing.capture_logs() as events:
        describe_ebm(describe_request)
    (usage,) = (event for event in events if event["event"] == "usage")
    assert usage["input_tokens"] == 7 * len(fake_llm.requests)
    assert usage["output_tokens"] == 5 * len(fake_llm.requests)


def test_the_local_options_reach_the_facts_that_the_prompts_carry(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    options = LocalOptions(n_samples=17, seed=3, max_steps=2)
    explanation = explain_instance(ExplainInstanceRequest(ebm, INSTANCE, "GLOBAL", model=llm_model, local=options))
    assert explanation.facts == compute_local_facts(ebm, INSTANCE, options)
    assert explanation.facts != compute_local_facts(ebm, INSTANCE, LocalOptions())
    assert fake_llm.requests


def test_a_retry_after_header_becomes_the_wait_in_the_message(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(status=429, headers={"Retry-After": "7"})
    with pytest.raises(RateLimitedError, match="Wait 7 seconds"):
        describe_graph(DescribeGraphRequest(ebm, feature_index=0, model=llm_model, num_sentences=2))
