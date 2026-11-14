"""The LLM tools of the MCP server, through an in-memory client, the real describe functions and a local LLM server."""

import pytest
from mcp import Client
from mcp_types import TextContent

from ebaml import ModelOptions
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server
from tests.conftest import FakeLLM, Reply
from tests.unit.ebaml.conftest import PandasFrame, as_frame

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def served(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
) -> tuple[ModelStore, ModelOptions]:
    features, ebm = binary_data_and_model
    return ModelStore(ebm=ebm, data=as_frame(features)), llm_model


async def _call(
    store: ModelStore,
    llm: ModelOptions | None,
    name: str,
    arguments: dict[str, object],
) -> tuple[bool, str]:
    async with Client(create_server(store, llm)) as session:
        result = await session.call_tool(name, arguments)
    (content,) = result.content
    assert isinstance(content, TextContent)
    return result.is_error, content.text


async def test_describe_feature_sends_the_three_turns_of_that_feature_and_returns_the_reply(
    served: tuple[ModelStore, ModelOptions],
    fake_llm: FakeLLM,
) -> None:
    store, llm = served
    fake_llm.handler = lambda _request: Reply(text="The age graph.")
    is_error, text = await _call(store, llm, "describe_feature", {"feature": "age"})
    assert not is_error
    assert text == "The age graph."
    assert len(fake_llm.requests) == 3
    (first,) = (request for request in fake_llm.requests if request.turns == 1)
    assert "Feature Name: age" in first.prompt


async def test_describe_model_asks_once_and_keeps_the_text_for_the_revision(
    served: tuple[ModelStore, ModelOptions],
    fake_llm: FakeLLM,
) -> None:
    store, llm = served
    first = await _call(store, llm, "describe_model", {})
    sent = len(fake_llm.requests)
    second = await _call(store, llm, "describe_model", {})
    assert first == second
    assert sent > 0
    assert len(fake_llm.requests) == sent, "the second call must send no request"


async def test_an_edit_makes_the_server_describe_the_new_head_again(
    served: tuple[ModelStore, ModelOptions],
    fake_llm: FakeLLM,
) -> None:
    store, llm = served
    await _call(store, llm, "describe_model", {})
    sent = len(fake_llm.requests)
    edit = {"kind": "Shift", "feature": "age", "selection": {"kind": "WholeGraph"}, "delta": 0.5}
    is_error, text = await _call(store, llm, "apply_edit", {"edit": edit, "reason": "integration test"})
    assert not is_error, text
    await _call(store, llm, "describe_model", {})
    assert len(fake_llm.requests) > sent


async def test_a_rejected_key_is_a_tool_error_with_the_steps_to_fix_it(
    served: tuple[ModelStore, ModelOptions],
    fake_llm: FakeLLM,
) -> None:
    store, llm = served
    fake_llm.handler = lambda _request: Reply(status=401)
    is_error, text = await _call(store, llm, "describe_feature", {"feature": "age"})
    assert is_error
    assert "rejected the API key" in text
    assert "To fix it" in text


async def test_a_server_without_an_llm_has_no_describe_tools_and_sends_nothing(
    served: tuple[ModelStore, ModelOptions],
    fake_llm: FakeLLM,
) -> None:
    store, _ = served
    async with Client(create_server(store)) as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert not {"describe_feature", "describe_model"} & names
    assert fake_llm.requests == []
