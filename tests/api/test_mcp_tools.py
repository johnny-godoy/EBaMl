"""The tools of the MCP server as an agent meets them: which tools exist, how they fail, and what they change.

The unit tests of ``tests/unit/ebaml/mcp`` hold the snapshot of the argument schemas. These tests hold the rules that
every tool obeys.
"""

from pathlib import Path

import orjson
import pytest
from mcp import Client
from mcp_types import CallToolResult, TextContent, Tool

from ebaml import ModelOptions
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server
from tests.unit.ebaml.conftest import PandasFrame, as_frame

pytestmark = pytest.mark.anyio

BASE_TOOLS = {
    "get_model_summary",
    "list_features",
    "get_graph",
    "explain_prediction",
    "list_bins",
    "preview_edit",
    "apply_edit",
    "undo_edit",
    "redo_edit",
    "checkout_edit",
    "list_edits",
}
LLM_TOOLS = {"describe_feature", "describe_model"}
SAVE_TOOLS = {"save_model"}
# the tools that must leave the session as it is
READ_ONLY: dict[str, dict[str, object]] = {
    "get_model_summary": {},
    "list_features": {},
    "get_graph": {"feature": "age"},
    "explain_prediction": {"row": 0},
    "list_bins": {"feature": "age"},
    "preview_edit": {"edit": {"kind": "Shift", "feature": "age", "selection": {"kind": "WholeGraph"}, "delta": 1.0}},
    "list_edits": {},
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store(binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier], tmp_path: Path) -> ModelStore:
    features, ebm = binary_data_and_model
    return ModelStore(ebm=ebm, data=as_frame(features), output_directory=tmp_path)


def _text(result: CallToolResult) -> str:
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


async def _tools(store: ModelStore, llm: ModelOptions | None) -> dict[str, Tool]:
    async with Client(create_server(store, llm)) as session:
        return {tool.name: tool for tool in (await session.list_tools()).tools}


async def test_the_server_offers_the_base_tools_the_save_tool_and_the_llm_tools_by_option(store: ModelStore) -> None:
    llm = ModelOptions(provider="ollama", model="llama3")
    assert set(await _tools(store, llm)) == BASE_TOOLS | SAVE_TOOLS | LLM_TOOLS
    assert set(await _tools(store, None)) == BASE_TOOLS | SAVE_TOOLS
    store.output_directory = None
    assert set(await _tools(store, None)) == BASE_TOOLS


async def test_each_tool_has_a_description_and_an_object_schema(store: ModelStore) -> None:
    llm = ModelOptions(provider="ollama", model="llama3")
    for name, tool in (await _tools(store, llm)).items():
        assert tool.description, f"the tool {name} has no description"
        assert tool.input_schema["type"] == "object"


def test_each_tool_name_is_snake_case_and_starts_with_a_verb_of_the_set() -> None:
    verbs = {"get", "list", "explain", "preview", "apply", "undo", "redo", "checkout", "save", "describe"}
    for name in BASE_TOOLS | SAVE_TOOLS | LLM_TOOLS:
        assert name == name.lower()
        assert name.split("_")[0] in verbs, f"{name} does not start with an approved verb"


@pytest.mark.parametrize("name", sorted(READ_ONLY))
async def test_a_read_tool_leaves_the_revision_of_the_session_as_it_is(store: ModelStore, name: str) -> None:
    async with Client(create_server(store)) as session:
        before = orjson.loads(_text(await session.call_tool("list_edits", {})))
        result = await session.call_tool(name, READ_ONLY[name])
        after = orjson.loads(_text(await session.call_tool("list_edits", {})))
    assert not result.is_error, result.content
    assert before == after


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_graph", {"feature": "no_such_feature"}),
        ("list_bins", {"feature": "no_such_feature"}),
        ("explain_prediction", {}),
        ("explain_prediction", {"row": 10**9}),
        ("undo_edit", {}),
        ("redo_edit", {}),
        ("checkout_edit", {"revision": "no-such-revision"}),
    ],
)
async def test_a_failed_call_is_a_tool_error_with_text_and_never_a_protocol_error(
    store: ModelStore,
    name: str,
    arguments: dict[str, object],
) -> None:
    async with Client(create_server(store)) as session:
        result = await session.call_tool(name, arguments)
    assert result.is_error
    assert _text(result).strip()


async def test_an_apply_without_a_reason_is_refused_by_the_schema(store: ModelStore) -> None:
    edit = {"kind": "Shift", "feature": "age", "selection": {"kind": "WholeGraph"}, "delta": 1.0}
    async with Client(create_server(store)) as session:
        result = await session.call_tool("apply_edit", {"edit": edit})
    assert result.is_error


async def test_an_edit_that_the_agent_applies_then_undoes_returns_the_session_to_the_original(
    store: ModelStore,
) -> None:
    edit = {"kind": "Shift", "feature": "age", "selection": {"kind": "WholeGraph"}, "delta": 1.0}
    async with Client(create_server(store)) as session:
        applied = await session.call_tool("apply_edit", {"edit": edit, "reason": "api test"})
        assert not applied.is_error, applied.content
        undone = await session.call_tool("undo_edit", {})
        assert not undone.is_error, undone.content
        summary = orjson.loads(_text(await session.call_tool("get_model_summary", {})))
    assert summary["session"]["revision"] == "original"
