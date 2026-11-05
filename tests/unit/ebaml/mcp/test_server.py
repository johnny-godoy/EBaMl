"""Tests of the MCP server, through an in-memory client session of the SDK."""

from pathlib import Path

import joblib
import orjson
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent, TextResourceContents
from pydantic import AnyUrl

from ebaml.llm.options import ModelOptions
from ebaml.mcp import server as server_module
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server, main
from tests.unit.ebaml.conftest import as_frame, instance_of

pytestmark = pytest.mark.anyio

SNAPSHOT = Path(__file__).parent / "tool_schemas.json"


async def _call(server: FastMCP, name: str, arguments: dict[str, object] | None = None) -> tuple[bool, str]:
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool(name, arguments or {})
    (content,) = result.content
    assert isinstance(content, TextContent)
    return result.isError, content.text


@pytest.fixture
def binary_server(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> FastMCP:
    ebm, data = continuous_ebm
    return create_server(ModelStore(ebm=ebm, data=as_frame(data)))


async def test_summary_of_a_binary_classifier(binary_server: FastMCP) -> None:
    is_error, text = await _call(binary_server, "get_model_summary")
    summary = orjson.loads(text)
    assert not is_error
    assert summary["kind"] == "BinaryClassifier"
    assert summary["classes"] == [0, 1]
    assert summary["feature_count"] == 3
    assert summary["skipped_terms"] == []


async def test_features_are_sorted_by_importance(binary_server: FastMCP) -> None:
    _, text = await _call(binary_server, "list_features")
    features = orjson.loads(text)
    importances = [feature["importance"] for feature in features]
    assert importances == sorted(importances, reverse=True)
    assert {feature["name"] for feature in features} == {"a", "b", "c"}
    assert features[0]["name"] == "a"


async def test_get_graph_returns_the_graph_text(binary_server: FastMCP) -> None:
    is_error, text = await _call(binary_server, "get_graph", {"feature": "a", "max_tokens": 500})
    assert not is_error
    assert "a" in text
    assert "min_max" not in text


async def test_get_graph_with_an_unknown_feature_is_a_tool_error(binary_server: FastMCP) -> None:
    is_error, text = await _call(binary_server, "get_graph", {"feature": "zzz"})
    assert is_error
    assert "no feature 'zzz'" in text


async def test_graph_resource_matches_the_default_tool(binary_server: FastMCP) -> None:
    async with create_connected_server_and_client_session(binary_server._mcp_server) as session:
        resource = await session.read_resource(AnyUrl("ebm://graphs/a"))
    (content,) = resource.contents
    assert isinstance(content, TextResourceContents)
    _, text = await _call(binary_server, "get_graph", {"feature": "a"})
    assert content.text == text


async def test_explain_prediction_by_instance_and_by_row(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, data = continuous_ebm
    server = create_server(ModelStore(ebm=ebm, data=as_frame(data)))
    instance = {key: (value.item() if hasattr(value, "item") else value) for key, value in instance_of(data, 0).items()}
    _, by_instance = await _call(server, "explain_prediction", {"instance": instance})
    _, by_row = await _call(server, "explain_prediction", {"row": 0})
    assert orjson.loads(by_instance) == orjson.loads(by_row)
    result = orjson.loads(by_row)
    assert set(result) == {"facts", "texts"}
    assert result["facts"]["prediction"]["decision"] in {0, 1}
    assert set(result["texts"]) == {"prediction", "counterfactual", "sufficient_reason", "completions"}


async def test_explain_prediction_errors_are_tool_errors(binary_server: FastMCP) -> None:
    both = await _call(binary_server, "explain_prediction", {"row": 0, "instance": {"a": 1}})
    none = await _call(binary_server, "explain_prediction")
    missing_row = await _call(binary_server, "explain_prediction", {"row": 100_000})
    assert both == (True, both[1])
    assert "exactly one" in both[1]
    assert none[0]
    assert "does not exist" in missing_row[1]


async def test_explain_prediction_of_a_regressor_says_it_is_unsupported(
    regression_ebm: ExplainableBoostingRegressor,
) -> None:
    server = create_server(ModelStore(ebm=regression_ebm))
    is_error, text = await _call(server, "explain_prediction", {"instance": {"a": 0.5, "c": "x"}})
    assert is_error
    assert "binary classifier" in text


async def test_summary_of_a_regressor_and_a_multiclass_classifier(
    regression_ebm: ExplainableBoostingRegressor,
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    regressor = orjson.loads((await _call(create_server(ModelStore(ebm=regression_ebm)), "get_model_summary"))[1])
    multiclass = orjson.loads((await _call(create_server(ModelStore(ebm=multiclass_ebm)), "get_model_summary"))[1])
    assert regressor["kind"] == "Regressor"
    assert regressor["classes"] == []
    assert multiclass["kind"] == "MulticlassClassifier"
    assert sorted(multiclass["classes"]) == ["high", "low", "mid"]
    assert len(multiclass["intercept"]) == 3


async def test_llm_tools_exist_only_with_an_llm_and_call_the_llm_tools(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm

    def fake_describe_feature(_store: ModelStore, _llm: ModelOptions, feature: str) -> str:
        return f"graph {feature}"

    def fake_describe_model(_store: ModelStore, _llm: ModelOptions) -> str:
        return "model text"

    monkeypatch.setattr(server_module.llm_tools, "describe_feature", fake_describe_feature)
    monkeypatch.setattr(server_module.llm_tools, "describe_model", fake_describe_model)
    options = ModelOptions(provider="ollama", model="llama3")
    names = {}
    for label, llm in (("without", None), ("with", options)):
        async with create_connected_server_and_client_session(
            create_server(ModelStore(ebm=ebm), llm)._mcp_server,
        ) as session:
            names[label] = {tool.name for tool in (await session.list_tools()).tools}
    assert names["with"] - names["without"] == {"describe_feature", "describe_model"}
    server = create_server(ModelStore(ebm=ebm), options)
    assert await _call(server, "describe_feature", {"feature": "a"}) == (False, "graph a")
    assert await _call(server, "describe_model") == (False, "model text")


async def test_tool_schemas_match_the_snapshot(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    """The names and argument schemas of the tools are public API from 1.0.0. Regenerate the snapshot on purpose."""
    ebm, _ = continuous_ebm
    server = create_server(ModelStore(ebm=ebm), ModelOptions(provider="ollama", model="llama3"))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = (await session.list_tools()).tools
        templates = (await session.list_resource_templates()).resourceTemplates
    current = {
        "tools": {tool.name: tool.inputSchema for tool in sorted(tools, key=lambda tool: tool.name)},
        "resource_templates": [template.uriTemplate for template in templates],
    }
    expected = orjson.loads(SNAPSHOT.read_bytes())
    assert current == expected


def test_main_loads_the_model_and_runs_the_server(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, data = continuous_ebm
    joblib.dump(ebm, tmp_path / "model.joblib")
    data.to_csv(tmp_path / "data.csv", index=False)
    started: list[tuple[FastMCP, str]] = []

    def fake_run(self: FastMCP, transport: str) -> None:
        started.append((self, transport))

    monkeypatch.setattr(FastMCP, "run", fake_run)
    main(["--model", str(tmp_path / "model.joblib"), "--data", str(tmp_path / "data.csv"), "--llm", "ollama/llama3"])
    main(["--model", str(tmp_path / "model.joblib"), "--transport", "streamable-http"])
    assert [transport for _, transport in started] == ["stdio", "streamable-http"]


async def test_no_tool_takes_a_path(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    """Loading a pickle runs code, so only the user names a file, at start. No tool argument may name one."""
    ebm, _ = continuous_ebm
    server = create_server(ModelStore(ebm=ebm), ModelOptions(provider="ollama", model="llama3"))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = (await session.list_tools()).tools
    argument_names = {name for tool in tools for name in tool.inputSchema.get("properties", {})}
    assert not {name for name in argument_names if any(part in name for part in ("path", "file", "dir", "url"))}
