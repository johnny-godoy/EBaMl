"""Tests for the conversation about one graph: its text, its description and the retry of a cut-off reply.

The tests of ``describe_graph`` and ``describe_ebm`` in ``test_descriptors.py`` run the conversation through the public
functions. The tests below call its parts directly.
"""

import pytest
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.llm import graph_conversation


def _finish_error_of(class_name: str, reason: str) -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason=reason, raw_output="")
    return BamlError(value, class_name=class_name)


FINISH_CLASS = "ai.errors.FinishReasonError"
MAX_TOKENS = "MaxTokens"


@pytest.mark.parametrize(
    ("class_name", "reason", "expected"),
    [
        (FINISH_CLASS, MAX_TOKENS, True),
        (FINISH_CLASS, "Refused", False),
        (FINISH_CLASS, "Complete", False),  # sorts before the reason
        (FINISH_CLASS, "ToolUse", False),  # sorts after the reason
        ("ai.errors.A", MAX_TOKENS, False),  # sorts before the class name
        ("ai.errors.Z", MAX_TOKENS, False),  # sorts after the class name
    ],
)
def test_is_truncated(class_name: str, reason: str, *, expected: bool) -> None:
    assert graph_conversation.is_truncated(_finish_error_of(class_name, reason)) is expected
