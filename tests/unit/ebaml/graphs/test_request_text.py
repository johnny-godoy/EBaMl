"""Tests for the text of the messages in the body of a request."""

import orjson
import pytest

from ebaml.graphs.request_text import message_text


def _body(**fields: object) -> str:
    return orjson.dumps(fields).decode()


def test_message_text_reads_the_messages_of_an_openai_body_and_leaves_out_the_parameters() -> None:
    body = _body(
        model="openai/gpt-6-luna",
        temperature=0.7,
        max_tokens=16,
        messages=[{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Describe it."}],
    )
    assert message_text(body) == "You are an expert.\nDescribe it."


def test_message_text_reads_the_parts_of_an_anthropic_body_and_skips_their_labels() -> None:
    body = _body(
        model="claude",
        system="Be brief.",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "again"}]}],
    )
    assert message_text(body) == "Be brief.\nHello\nagain"


def test_message_text_reads_the_parts_of_a_gemini_body() -> None:
    body = _body(
        system_instruction={"parts": [{"text": "Rules."}]},
        contents=[{"role": "user", "parts": [{"text": "Question?"}]}],
        generationConfig={"temperature": 0.1},
    )
    assert message_text(body) == "Rules.\nQuestion?"


def test_message_text_reads_the_instructions_of_a_responses_body() -> None:
    assert message_text(_body(instructions="Do it.", input=[{"role": "user", "content": "Now."}])) == "Do it.\nNow."


def test_message_text_keeps_the_order_of_the_body() -> None:
    body = _body(messages=[{"content": "first"}, {"content": "second"}, {"content": "third"}])
    assert message_text(body) == "first\nsecond\nthird"


def test_message_text_keeps_a_value_that_is_named_type_or_role_in_the_text_of_a_message() -> None:
    # `type` and `role` label a part only when a text key lies above them and they are keys, not words in the text
    assert message_text(_body(messages=[{"role": "user", "content": "type role"}])) == "type role"


def test_message_text_leaves_out_a_string_that_no_text_key_holds() -> None:
    assert (
        message_text(_body(model="m", metadata={"note": "not a prompt"}, messages=[{"content": "prompt"}])) == "prompt"
    )


@pytest.mark.parametrize("body", ["", "not json", "{broken"])
def test_message_text_of_a_body_that_is_not_json_is_empty(body: str) -> None:
    assert not message_text(body)


def test_message_text_of_a_body_with_no_text_is_empty() -> None:
    assert not message_text(_body(model="m", temperature=0.2))


def test_message_text_ignores_a_number_below_a_text_key() -> None:
    assert not message_text(_body(messages=[{"content": 5}]))
