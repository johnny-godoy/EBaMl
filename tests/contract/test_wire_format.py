"""The chat completions protocol of OpenAI, which ebaml speaks through the BAML runtime, seen from the provider.

A gateway or a local server follows the same protocol, so each rule here is a rule that a provider relies on: what a
request holds, and what a reply may hold. The tests run against the local server, so they send no real request.
"""

import pytest
from baml_bridge.errors import BamlError

from ebaml import DescribeGraphRequest, EBaMlError, ModelOptions, describe_graph
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.unit.ebaml.conftest import PandasFrame

# The top-level keys of a request that the protocol defines, and that a provider accepts.
ALLOWED_KEYS = {"model", "messages", "temperature", "max_tokens", "stream", "stream_options", "n", "stop", "top_p"}


def _ask(ebm: ExplainableBoostingClassifier, model: ModelOptions) -> str:
    return describe_graph(DescribeGraphRequest(ebm, feature_index=0, model=model, num_sentences=2))


@pytest.fixture
def requests_of_one_graph(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> list[RecordedRequest]:
    _, ebm = binary_data_and_model
    _ask(ebm, llm_model)
    return fake_llm.requests


def test_a_request_is_a_post_of_json_to_the_chat_completions_path_with_a_bearer_key(
    requests_of_one_graph: list[RecordedRequest],
) -> None:
    for request in requests_of_one_graph:
        assert request.path == "/v1/chat/completions"
        assert request.headers["content-type"].startswith("application/json")
        assert request.headers["authorization"] == "Bearer fake-key-of-the-tests"


def test_a_request_body_uses_only_the_keys_of_the_protocol(requests_of_one_graph: list[RecordedRequest]) -> None:
    for request in requests_of_one_graph:
        assert set(request.body) <= ALLOWED_KEYS, set(request.body) - ALLOWED_KEYS
        assert {"model", "messages"} <= set(request.body)


def test_the_messages_open_with_a_system_message_and_then_alternate_between_user_and_assistant(
    requests_of_one_graph: list[RecordedRequest],
) -> None:
    for request in requests_of_one_graph:
        roles = [message["role"] for message in request.messages]
        assert roles[0] == "system"
        assert roles[1:] == ["user", "assistant"] * ((len(roles) - 2) // 2) + ["user"]


def test_each_message_has_text(requests_of_one_graph: list[RecordedRequest]) -> None:
    for request in requests_of_one_graph:
        for message in request.messages:
            assert str(message["content"]).strip()


def test_the_limits_are_a_positive_number_and_a_temperature_in_range(
    requests_of_one_graph: list[RecordedRequest],
) -> None:
    for request in requests_of_one_graph:
        assert isinstance(request.body["max_tokens"], int)
        assert request.body["max_tokens"] > 0
        assert isinstance(request.body["temperature"], float)
        assert 0 <= request.body["temperature"] <= 2


def test_the_request_asks_for_a_whole_reply_and_not_a_stream(requests_of_one_graph: list[RecordedRequest]) -> None:
    assert all(not request.body.get("stream", False) for request in requests_of_one_graph)


def test_a_reply_with_fields_that_the_protocol_does_not_list_is_accepted(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(
        text="With extras.",
        extra_fields={"system_fingerprint": "fp", "service_tier": "default", "provider_specific": {"a": 1}},
    )
    assert _ask(ebm, llm_model) == "With extras."


def test_the_text_of_a_reply_keeps_its_characters(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    text = "Ünïcode ✓ — “quotes”, a line\nbreak, and {braces}."
    fake_llm.handler = lambda _request: Reply(text=text)
    _, ebm = binary_data_and_model
    assert _ask(ebm, llm_model) == text


def test_a_reply_with_malformed_choices_is_an_error_and_not_a_text(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.handler = lambda _request: Reply(status=200, text="x", extra_fields={"choices": "not a list"})
    _, ebm = binary_data_and_model
    with pytest.raises((BamlError, EBaMlError)):
        _ask(ebm, llm_model)
