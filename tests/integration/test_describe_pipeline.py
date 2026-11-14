"""The path of ``describe_ebm`` from a fitted EBM to the text, through the BAML runtime and a local LLM server.

One graph takes a conversation of three turns, and each turn repeats the earlier ones. The summary takes two turns.
"""

import pytest

from ebaml import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    MissingAPIKeyError,
    ModelOptions,
    describe_ebm,
    describe_graph,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.unit.ebaml.conftest import PandasFrame

GRAPH_TURNS = 3
SUMMARY_TURNS = 2


def _turns(requests: list[RecordedRequest]) -> list[int]:
    return sorted(request.turns for request in requests)


def test_each_graph_takes_three_requests_and_the_summary_two(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    describe_ebm(describe_request)
    feature_count = len(describe_request.ebm.term_names_)
    assert len(fake_llm.requests) == feature_count * GRAPH_TURNS + SUMMARY_TURNS
    expected = [*range(1, GRAPH_TURNS + 1)] * feature_count + [*range(1, SUMMARY_TURNS + 1)]
    assert _turns(fake_llm.requests) == sorted(expected)


def test_the_result_is_the_summary_text_and_names_no_skipped_part(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.handler = lambda _request: Reply(text="The summary.")
    description = describe_ebm(describe_request)
    assert description.text == "The summary."
    assert description.skipped_terms == []
    assert description.skipped_features == []


def test_each_request_goes_to_the_chat_endpoint_of_the_base_url_with_the_chosen_model(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    describe_ebm(describe_request)
    assert {request.path for request in fake_llm.requests} == {"/v1/chat/completions"}
    assert {request.body["model"] for request in fake_llm.requests} == {"fake-model"}


def test_the_graph_prompts_hold_the_graph_of_each_feature(
    describe_request: DescribeEBMRequest,
    fake_llm: FakeLLM,
) -> None:
    describe_ebm(describe_request)
    prompts = "\n".join(fake_llm.prompts)
    for name in describe_request.ebm.feature_names_in_:
        assert f"Feature Name: {name}" in prompts or name in prompts


def test_a_later_turn_repeats_the_reply_to_the_turn_before_it(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    replies = iter(f"reply-number-{number}" for number in range(100))
    fake_llm.handler = lambda _request: Reply(text=next(replies))
    describe_graph(DescribeGraphRequest(ebm, feature_index=0, model=llm_model, num_sentences=2))
    first, second, third = fake_llm.requests
    assert "reply-number-0" not in first.prompt
    assert "reply-number-0" in second.prompt
    assert "reply-number-1" not in second.prompt
    assert "reply-number-0" in third.prompt
    assert "reply-number-1" in third.prompt


def test_the_summary_prompt_carries_the_description_of_each_graph(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model

    def answer(request: RecordedRequest) -> Reply:
        return Reply(text=f"final-graph-text-{request.turns}")

    fake_llm.handler = answer
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, max_concurrency=1, num_sentences=3))
    summary = fake_llm.requests[-SUMMARY_TURNS]
    assert summary.prompt.count("final-graph-text-3") == len(ebm.term_names_)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_the_request_count_does_not_depend_on_the_concurrency(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    concurrency: int,
) -> None:
    _, ebm = binary_data_and_model
    describe_ebm(DescribeEBMRequest(ebm, model=llm_model, max_concurrency=concurrency, num_sentences=3))
    assert len(fake_llm.requests) == len(ebm.term_names_) * GRAPH_TURNS + SUMMARY_TURNS


def test_the_model_options_reach_the_wire(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    tuned = ModelOptions(
        provider=llm_model.provider,
        model=llm_model.model,
        api_key_env=llm_model.api_key_env,
        base_url=llm_model.base_url,
        temperature=0.25,
        max_output_tokens=777,
        headers={"x-ebaml-test": "yes"},
    )
    describe_ebm(DescribeEBMRequest(ebm, model=tuned, num_sentences=3))
    for request in fake_llm.requests:
        assert request.body["temperature"] == pytest.approx(0.25)
        assert request.body["max_tokens"] == 777
        assert request.headers["x-ebaml-test"] == "yes"
        assert request.headers["authorization"] == "Bearer fake-key-of-the-tests"


def test_the_shorthand_model_needs_the_key_of_its_provider_and_sends_nothing(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    del llm_model  # the fixture removes the keys of the real providers
    _, ebm = binary_data_and_model
    with pytest.raises(MissingAPIKeyError):
        describe_ebm(DescribeEBMRequest(ebm, model="openai/gpt-4o-mini"))
    assert fake_llm.requests == []


def test_describe_graph_returns_the_text_of_the_last_turn(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(text="A graph.")
    text = describe_graph(DescribeGraphRequest(ebm, feature_index=0, model=llm_model, num_sentences=2))
    assert text == "A graph."
    assert len(fake_llm.requests) == GRAPH_TURNS
