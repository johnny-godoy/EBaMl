"""Tests for the requests of the description functions."""

import attrs
import pytest

from ebaml.llm.options import GraphOptions, PromptOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ExplainInstanceRequest
from ebaml.local.facts import LocalOptions
from tests.unit.ebaml.conftest import fake_model

OPTION_NAMES = {"graph", "prompt", "local", "model", "language"}


@pytest.mark.parametrize("request_class", [DescribeGraphRequest, DescribeEBMRequest, ExplainInstanceRequest])
def test_option_fields_are_keyword_only(request_class: type) -> None:
    # an option set has no natural place in a positional list, so a call must name it
    kw_only = {field.name: field.kw_only for field in attrs.fields(request_class) if field.name in OPTION_NAMES}
    assert kw_only
    assert all(kw_only.values())


def test_describe_graph_request_defaults() -> None:
    request = DescribeGraphRequest(fake_model(), 3)
    assert (request.num_sentences, request.graph, request.prompt, request.model) == (
        7,
        GraphOptions(),
        PromptOptions(),
        None,
    )


def test_describe_ebm_request_defaults() -> None:
    request = DescribeEBMRequest(fake_model())
    assert (request.num_sentences, request.max_concurrency, request.graph, request.prompt, request.model) == (
        30,
        8,
        GraphOptions(),
        PromptOptions(),
        None,
    )


def test_explain_instance_request_defaults() -> None:
    request = ExplainInstanceRequest(fake_model(), {}, "global")
    assert (
        request.audience,
        request.num_sentences,
        request.language,
        request.local,
        request.prompt,
        request.model,
    ) == (
        "layperson",
        5,
        "English",
        LocalOptions(),
        PromptOptions(),
        None,
    )
