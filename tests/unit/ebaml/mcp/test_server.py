"""Tests of the MCP server, through an in-memory client session of the SDK."""

import ast
import asyncio
import socket
import sys
from pathlib import Path

import anyio
import joblib  # ruff: ignore[banned-api] -- the test writes the joblib file that ebaml-mcp loads
import orjson
import pandas as pd  # ruff: ignore[banned-api] -- the test builds the pandas frame that interpret fits on
import pytest
import skops.io
import structlog
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp_types import TextContent, TextResourceContents

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.llm.options import ModelOptions
from ebaml.mcp import llm_tools as llm_tools_module
from ebaml.mcp import server as server_module
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server, main
from tests.unit.ebaml.conftest import as_frame, instance_of

pytestmark = pytest.mark.anyio

SNAPSHOT = Path(__file__).parent / "tool_schemas.json"


async def _call(server: MCPServer, name: str, arguments: dict[str, object] | None = None) -> tuple[bool, str]:
    async with Client(server) as session:
        result = await session.call_tool(name, arguments or {})
    (content,) = result.content
    assert isinstance(content, TextContent)
    return result.is_error, content.text


@pytest.fixture
def binary_server(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> MCPServer:
    ebm, data = continuous_ebm
    return create_server(ModelStore(ebm=ebm, data=as_frame(data)))


async def test_summary_of_a_binary_classifier(binary_server: MCPServer) -> None:
    is_error, text = await _call(binary_server, "get_model_summary")
    summary = orjson.loads(text)
    assert not is_error
    assert summary["kind"] == "BinaryClassifier"
    assert summary["classes"] == [0, 1]
    assert summary["feature_count"] == 3
    assert summary["skipped_terms"] == []


async def test_features_are_sorted_by_importance(binary_server: MCPServer) -> None:
    _, text = await _call(binary_server, "list_features")
    features = orjson.loads(text)
    importances = [feature["importance"] for feature in features]
    assert importances == sorted(importances, reverse=True)
    assert {feature["name"] for feature in features} == {"a", "b", "c"}
    assert features[0]["name"] == "a"


async def test_get_graph_returns_the_graph_text(binary_server: MCPServer) -> None:
    is_error, text = await _call(binary_server, "get_graph", {"feature": "a"})
    assert not is_error
    assert "a" in text
    assert "min_max" not in text


async def test_get_graph_with_an_unknown_feature_is_a_tool_error(binary_server: MCPServer) -> None:
    is_error, text = await _call(binary_server, "get_graph", {"feature": "zzz"})
    assert is_error
    assert "no feature 'zzz'" in text


async def test_graph_resource_matches_the_default_tool(binary_server: MCPServer) -> None:
    async with Client(binary_server) as session:
        resource = await session.read_resource("ebm://graphs/a")
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


async def test_explain_prediction_errors_are_tool_errors(binary_server: MCPServer) -> None:
    both = await _call(binary_server, "explain_prediction", {"row": 0, "instance": {"a": 1}})
    none = await _call(binary_server, "explain_prediction")
    missing_row = await _call(binary_server, "explain_prediction", {"row": 100_000})
    assert both == (True, both[1])
    assert "exactly one" in both[1]
    assert none[0]
    assert "does not exist" in missing_row[1]


async def test_explain_prediction_of_a_regressor_gives_the_facts(regression_ebm: ExplainableBoostingRegressor) -> None:
    server = create_server(ModelStore(ebm=regression_ebm))
    is_error, text = await _call(server, "explain_prediction", {"instance": {"a": 0.5, "c": "x"}})
    assert not is_error
    assert orjson.loads(text)["facts"]["prediction"]["output_kind"] == "value"


async def test_explain_prediction_of_a_multiclass_classifier_gives_the_facts(
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    server = create_server(ModelStore(ebm=multiclass_ebm))
    is_error, text = await _call(server, "explain_prediction", {"instance": {"a": 2.0, "c": "x"}})
    assert not is_error
    assert orjson.loads(text)["facts"]["prediction"]["output_kind"] == "class_probability"


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


async def test_features_and_graphs_of_a_regressor(regression_ebm: ExplainableBoostingRegressor) -> None:
    server = create_server(ModelStore(ebm=regression_ebm))
    features = orjson.loads((await _call(server, "list_features"))[1])
    is_error, graph = await _call(server, "get_graph", {"feature": "a"})
    assert {feature["name"]: feature["type"] for feature in features} == {"a": "continuous", "c": "nominal"}
    assert features[0]["name"] == "a"
    assert not is_error
    assert "Feature Name: a" in graph


async def test_features_and_graphs_of_a_multiclass_classifier(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    server = create_server(ModelStore(ebm=multiclass_ebm))
    features = orjson.loads((await _call(server, "list_features"))[1])
    is_error, graph = await _call(server, "get_graph", {"feature": "c"})
    assert {feature["name"] for feature in features} == {"a", "c"}
    assert not is_error
    assert "Feature Name: c" in graph
    for label in ("low", "mid", "high"):
        assert label in graph


async def test_a_model_with_an_interaction_names_it_and_the_tools_skip_it(
    interaction_ebm: ExplainableBoostingClassifier,
) -> None:
    server = create_server(ModelStore(ebm=interaction_ebm))
    summary = orjson.loads((await _call(server, "get_model_summary"))[1])
    features = orjson.loads((await _call(server, "list_features"))[1])
    interaction_graph = await _call(server, "get_graph", {"feature": "p x q"})
    assert summary["skipped_terms"] == ["p x q"]
    assert summary["feature_count"] == 2
    assert {feature["name"] for feature in features} == {"p", "q"}
    assert interaction_graph[0]
    assert "no feature 'p x q'" in interaction_graph[1]


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
        async with Client(create_server(ModelStore(ebm=ebm), llm)) as session:
            names[label] = {tool.name for tool in (await session.list_tools()).tools}
    assert names["with"] - names["without"] == {"describe_feature", "describe_model"}
    server = create_server(ModelStore(ebm=ebm), options)
    assert await _call(server, "describe_feature", {"feature": "a"}) == (False, "graph a")
    assert await _call(server, "describe_model") == (False, "model text")


async def test_llm_tools_reach_the_llm_once_through_the_client(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake the two calls into ``ebaml.llm``, not the tools: the full path from the client to the LLM runs."""
    ebm, _ = continuous_ebm
    options = ModelOptions(provider="ollama", model="llama3")
    graph_calls: list[tuple[int, ModelOptions]] = []
    model_calls: list[ModelOptions] = []

    def fake_describe_graph(fitted: ExplainableBoostingModel, feature_index: int, *, model: ModelOptions) -> str:
        assert fitted is ebm
        graph_calls.append((feature_index, model))
        return f"words about feature {feature_index}"

    def fake_describe_ebm(fitted: ExplainableBoostingModel, *, model: ModelOptions) -> str:
        assert fitted is ebm
        model_calls.append(model)
        return f"words about the model, call {len(model_calls)}"

    monkeypatch.setattr(llm_tools_module, "describe_graph", fake_describe_graph)
    monkeypatch.setattr(llm_tools_module, "describe_ebm", fake_describe_ebm)
    server = create_server(ModelStore(ebm=ebm), options)
    async with Client(server) as session:
        feature = await session.call_tool("describe_feature", {"feature": "b"})
        missing = await session.call_tool("describe_feature", {"feature": "zzz"})
        first = await session.call_tool("describe_model", {})
        second = await session.call_tool("describe_model", {})
    texts = [result.content[0].text for result in (feature, first, second)]  # pyrefly: ignore[missing-attribute] -- the tools return text content
    assert texts == ["words about feature 1", "words about the model, call 1", "words about the model, call 1"]
    assert missing.is_error
    assert graph_calls == [(1, options)]
    assert model_calls == [options]


async def test_tool_schemas_match_the_snapshot(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    """The names and argument schemas of the tools are public API from 1.0.0. Regenerate the snapshot on purpose."""
    ebm, _ = continuous_ebm
    server = create_server(ModelStore(ebm=ebm), ModelOptions(provider="ollama", model="llama3"))
    async with Client(server) as session:
        tools = (await session.list_tools()).tools
        templates = (await session.list_resource_templates()).resource_templates
    current = {
        "tools": {tool.name: tool.input_schema for tool in sorted(tools, key=lambda tool: tool.name)},
        "resource_templates": [template.uri_template for template in templates],
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
    started: list[tuple[MCPServer, str]] = []

    def fake_run(self: MCPServer, transport: str, **_address: str | int) -> None:
        started.append((self, transport))

    monkeypatch.setattr(MCPServer, "run", fake_run)
    main(["--model", str(tmp_path / "model.joblib"), "--data", str(tmp_path / "data.csv"), "--llm", "ollama/llama3"])
    main(["--model", str(tmp_path / "model.joblib"), "--transport", "streamable-http"])
    assert [transport for _, transport in started] == ["stdio", "streamable-http"]


@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
def test_log_events_go_to_stderr_and_never_to_the_protocol_channel(
    transport: str,
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug: structlog printed to stdout, so a log event broke the JSON-RPC stream of the stdio server."""
    ebm, _ = continuous_ebm
    skops.io.dump(ebm, tmp_path / "model.skops")

    def fake_run(_self: MCPServer, transport: str, **_address: str | int) -> None:
        structlog.get_logger("probe").info("probe_event", transport=transport)

    monkeypatch.setattr(MCPServer, "run", fake_run)
    try:
        main(["--model", str(tmp_path / "model.skops"), "--transport", transport])
        captured = capsys.readouterr()
    finally:
        structlog.reset_defaults()  # serve configures structlog for the whole process, so undo it for the other tests
    assert not captured.out
    assert "probe_event" in captured.err


def test_main_passes_the_host_and_the_port_to_the_server(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    skops.io.dump(ebm, tmp_path / "model.skops")
    listening: list[tuple[str, str | None, int | None]] = []

    def fake_run(_self: MCPServer, transport: str, host: str | None = None, port: int | None = None) -> None:
        listening.append((transport, host, port))

    monkeypatch.setattr(MCPServer, "run", fake_run)
    model = str(tmp_path / "model.skops")
    main(["--model", model, "--transport", "streamable-http"])
    main(["--model", model, "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9123"])  # ruff: ignore[hardcoded-bind-all-interfaces] -- the test checks the option, and the server does not start
    main(["--model", model, "--host", "0.0.0.0", "--port", "9123"])  # ruff: ignore[hardcoded-bind-all-interfaces] -- the test checks that stdio ignores the address
    assert listening == [
        ("streamable-http", "127.0.0.1", 8000),
        ("streamable-http", "0.0.0.0", 9123),  # ruff: ignore[hardcoded-bind-all-interfaces] -- the value that the test passed
        ("stdio", None, None),
    ]


def test_main_rejects_an_unknown_transport(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--model", str(tmp_path / "model.joblib"), "--transport", "carrier-pigeon"])
    assert exit_info.value.code != 0
    assert "carrier-pigeon" in capsys.readouterr().err


async def test_no_tool_takes_a_path(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    """Loading a pickle runs code, so only the user names a file, at start. No tool argument may name one."""
    ebm, _ = continuous_ebm
    server = create_server(ModelStore(ebm=ebm), ModelOptions(provider="ollama", model="llama3"))
    async with Client(server) as session:
        tools = (await session.list_tools()).tools
    argument_names = {name for tool in tools for name in tool.input_schema.get("properties", {})}
    assert not {name for name in argument_names if any(part in name for part in ("path", "file", "dir", "url"))}


def test_only_the_command_loads_files() -> None:
    """Only ``serve``, which gets its paths from the command line, calls ``load_model`` and ``load_data``."""
    tree = ast.parse(Path(server_module.__file__).read_text(encoding="utf-8"))
    callers = {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef)
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and ast.unparse(call.func) in {"load_model", "load_data"}
    }
    # ast.walk also visits the functions that a function holds, so a tool inside create_server would add its parent
    assert callers == {"serve"}


async def test_a_tool_argument_with_a_path_loads_no_file(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent cannot make the server load a file, even if it adds an argument that names a path."""
    ebm, data = continuous_ebm

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a tool loaded a file")

    monkeypatch.setattr(server_module, "load_model", forbidden)
    monkeypatch.setattr(server_module, "load_data", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    server = create_server(ModelStore(ebm=ebm, data=as_frame(data)), ModelOptions(provider="ollama", model="llama3"))
    attempts: tuple[tuple[str, dict[str, object]], ...] = (
        ("get_graph", {"feature": "/etc/passwd", "path": "/etc/passwd"}),
        ("explain_prediction", {"row": 0, "model": "../evil.joblib"}),
        ("get_model_summary", {"path": "../evil.joblib"}),
    )
    for name, arguments in attempts:
        is_error, _ = await _call(server, name, arguments)
        assert is_error == (name == "get_graph")


def _command() -> str:
    """Find the ``ebaml-mcp`` script that the project installs next to the Python of the tests.

    Returns
    -------
    str
        The path of the script.

    """
    command = Path(sys.executable).parent / "ebaml-mcp"
    assert command.exists(), "install the package with `uv sync --extra mcp`"
    return str(command)


@pytest.fixture
def saved_files(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
) -> tuple[Path, Path]:
    ebm, data = continuous_ebm
    skops.io.dump(ebm, tmp_path / "model.skops")
    data.to_csv(tmp_path / "data.csv", index=False)
    return tmp_path / "model.skops", tmp_path / "data.csv"


async def test_the_real_command_serves_the_tools_over_stdio(saved_files: tuple[Path, Path]) -> None:
    """Start ``ebaml-mcp`` as a client does. This needs no LLM key, because the command has no ``--llm``."""
    model, data = saved_files
    parameters = StdioServerParameters(command=_command(), args=["--model", str(model), "--data", str(data)])
    async with Client(parameters) as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
        summary = await session.call_tool("get_model_summary", {})
        features = await session.call_tool("list_features", {})
        graph = await session.call_tool("get_graph", {"feature": "a"})
        explanation = await session.call_tool("explain_prediction", {"row": 0})
    texts = [result.content[0].text for result in (summary, features, graph, explanation)]  # pyrefly: ignore[missing-attribute] -- the tools return text content
    assert names == {"get_model_summary", "list_features", "get_graph", "explain_prediction"}
    assert not any(result.is_error for result in (summary, features, graph, explanation))
    assert orjson.loads(texts[0])["kind"] == "BinaryClassifier"
    assert orjson.loads(texts[1])[0]["name"] == "a"
    assert "Feature Name: a" in texts[2]
    assert set(orjson.loads(texts[3])) == {"facts", "texts"}


async def test_the_real_command_listens_with_streamable_http_on_the_given_port(saved_files: tuple[Path, Path]) -> None:
    model, _ = saved_files
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [_command(), "--model", str(model), "--transport", "streamable-http", "--host", "127.0.0.1"]
    async with await anyio.open_process([*command, "--port", str(port)], stderr=asyncio.subprocess.DEVNULL) as process:
        try:
            names = await _tool_names_when_ready(f"http://127.0.0.1:{port}/mcp")
        finally:
            process.terminate()
    assert "get_model_summary" in names


async def _tool_names_when_ready(url: str) -> set[str]:
    """List the tools of the server at the URL. Retry for up to 30 seconds while the server starts.

    Returns
    -------
    set[str]
        The names of the tools.

    """
    for _attempt in range(60):
        try:
            async with Client(url) as session:
                return {tool.name for tool in (await session.list_tools()).tools}
        except (OSError, ExceptionGroup):
            await asyncio.sleep(0.5)
    pytest.fail("the HTTP server did not start in 30 seconds")
