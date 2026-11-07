"""Create the MCP server, register the tools and start it from the command line.

Each tool is a thin adapter. It calls a function of a lower layer and turns the result into JSON text. A package error
(``EBaMlError``) becomes a tool error with its message.
"""

import contextlib
import sys
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Literal

import attrs
import cyclopts
import orjson
import structlog
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ebaml.core.custom_types import FeatureValue, JSONValue
from ebaml.core.exceptions import EBaMlError
from ebaml.llm.options import ModelOptions
from ebaml.mcp import explain_tools, llm_tools, read_tools
from ebaml.mcp.model_store import ModelStore, load_data, load_model

_INSTRUCTIONS = (
    "This server gives you a fitted Explainable Boosting Machine (EBM), a model that is a sum of one graph per feature."
    " Start with get_model_summary and list_features. Read a graph with get_graph. Explain one prediction with"
    " explain_prediction. Every result of these tools is exact and comes from the model, not from a language model."
)


def _json(value: JSONValue) -> str:
    """Write a value as JSON text.

    Returns
    -------
    str
        The text.

    """
    return orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode()


@contextlib.contextmanager
def _tool_errors() -> Generator[None]:
    """Turn a package error into a tool error with its message. Other errors pass on.

    Raises
    ------
    ToolError
        If the body raises an ``EBaMlError``.

    """
    try:
        yield
    except EBaMlError as error:
        raise ToolError(str(error)) from error


def create_server(
    store: ModelStore,
    llm: ModelOptions | None = None,
) -> MCPServer:
    """Create the server and register the tools.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions, optional
        The LLM. Without it, the server has no LLM tools.

    Returns
    -------
    MCPServer
        The server.

    """
    server = MCPServer("ebaml", instructions=_INSTRUCTIONS)

    @server.tool()
    def get_model_summary() -> str:
        """Summarize the model: its kind, outputs or classes, intercept and number of features.

        Call this first. It also names the interaction terms that the other tools skip.

        Returns
        -------
        str
            JSON with the kind, outputs, classes, score_meaning, intercept, feature_count and skipped_terms.

        """
        with _tool_errors():
            return _json(attrs.asdict(read_tools.model_summary(store), recurse=False))

    @server.tool()
    def list_features() -> str:
        """List the features of the model with their type and importance, the most important first.

        Call this to choose the feature to read with get_graph.

        Returns
        -------
        str
            JSON list with the name, type and importance of each feature.

        """
        with _tool_errors():
            return _json([attrs.asdict(feature) for feature in read_tools.list_features(store)])

    @server.tool()
    def get_graph(
        feature: str,
        normalization: Literal["none", "min_max", "z_score"] = "none",
        x_axis_precision: int | None = None,
        y_axis_precision: int | None = None,
        max_tokens: int = 3000,
    ) -> str:
        """Read the graph of one feature: the score of the model for each value or interval of the feature.

        Use list_features for the names. A high score raises the prediction. Lower max_tokens to get a shorter text.

        Returns
        -------
        str
            The text of the graph.

        """
        with _tool_errors():
            return read_tools.graph_text(
                store,
                feature,
                normalization=normalization,
                x_axis_precision=x_axis_precision,
                y_axis_precision=y_axis_precision,
                max_tokens=max_tokens,
            )

    @server.resource("ebm://graphs/{feature}")
    def graph_resource(feature: str) -> str:
        """Read the graph of one feature as text, with the default options of get_graph.

        Returns
        -------
        str
            The text of the graph.

        """
        with _tool_errors():
            return read_tools.graph_text(store, feature)

    @server.tool()
    def explain_prediction(
        instance: Mapping[str, FeatureValue] | None = None,
        row: int | None = None,
        confidence_level: float = 0.95,
    ) -> str:
        """Explain one prediction of a binary classifier or a regressor with exact facts.

        A regressor's explanation keeps the answer to the question "is the predicted value above the average
        prediction?". Give the value of every feature in instance, or give the number of a row of the data that the
        server loaded.

        Returns
        -------
        str
            JSON with the facts and their texts. The facts have the counterfactual (the fewest changes that flip the
            prediction), the sufficient reason (the features that fix the prediction) and the share of completions
            with the same prediction.

        """
        with _tool_errors():
            explanation = explain_tools.explain_prediction(
                store,
                instance=instance,
                row=row,
                confidence_level=confidence_level,
            )
            return _json(attrs.asdict(explanation))

    if llm is not None:
        _register_llm_tools(server, store, llm)
    return server


def _register_llm_tools(server: MCPServer, store: ModelStore, llm: ModelOptions) -> None:
    @server.tool()
    def describe_feature(feature: str) -> str:
        """Ask a language model to describe the graph of one feature in words. This costs money and takes seconds.

        Returns
        -------
        str
            The description.

        """
        with _tool_errors():
            return llm_tools.describe_feature(store, llm, feature)

    @server.tool()
    def describe_model() -> str:
        """Ask a language model to describe the whole model in words.

        The first call is slow, and the server keeps the text.

        Returns
        -------
        str
            The description.

        """
        with _tool_errors():
            return llm_tools.describe_model(store, llm)


app: cyclopts.App = cyclopts.App(
    name="ebaml-mcp",
    help="Serve a fitted EBM to an MCP client.",
    result_action="return_value",
)


@app.default
def serve(  # ruff: ignore[too-many-arguments] -- one parameter for each option of the command
    *,
    model: Path,
    data: Path | None = None,
    llm: str | None = None,
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Load the EBM and run the MCP server.

    Parameters
    ----------
    model: Path
        EBM in a skops file, which runs no code, or in a pickled or joblib file, which does.
    data: Path, optional
        CSV file with the rows for the row argument.
    llm: str, optional
        PROVIDER/MODEL of the LLM tools. Without it the server has none.
    transport: Literal["stdio", "streamable-http"]
        How the server talks to the client.
    host: str
        The address that the ``streamable-http`` transport listens on. Use ``0.0.0.0`` to accept other computers.
    port: int
        The port that the ``streamable-http`` transport listens on.

    """
    # structlog prints to stdout by default, and a stdio server speaks JSON-RPC there, so a log line would corrupt it
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    store = ModelStore(ebm=load_model(model), data=load_data(data) if data else None)
    server = create_server(store, ModelOptions.parse(llm) if llm else None)
    # The stdio transport takes no address, and the streamable-http transport takes it when it runs.
    if transport == "streamable-http":
        server.run(transport="streamable-http", host=host, port=port)
    else:
        server.run(transport="stdio")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``ebaml-mcp`` command.

    Parameters
    ----------
    argv: Sequence[str], optional
        The arguments. The default is the command line.

    """
    app(None if argv is None else list(argv))
