"""Create the MCP server, register the tools and start it from the command line.

Each tool is a thin adapter. It calls a function of a lower layer and turns the result into JSON text. A package error
(``EBaMlError``) becomes a tool error with its message.
"""

import asyncio
import contextlib
import sys
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import attrs
import cyclopts
import orjson
import structlog
from mcp.server.mcpserver import Context, Elicit, MCPServer, Resolve
from mcp.server.mcpserver.exceptions import ToolError

from ebaml.core.custom_types import FeatureValue, JSONValue
from ebaml.core.exceptions import EBaMlError
from ebaml.edit.edit_session import EditSession
from ebaml.llm.options import ModelOptions
from ebaml.mcp import edit_tools, explain_tools, llm_tools, read_tools, save_tools
from ebaml.mcp.model_store import ModelStore, load_data, load_model

_INSTRUCTIONS = (
    "This server gives you a fitted Explainable Boosting Machine (EBM), a model that is a sum of one graph per feature."
    " Start with get_model_summary and list_features. Read a graph with get_graph. Explain one prediction with"
    " explain_prediction. Every result of these tools is exact and comes from the model, not from a language model."
    " You can also edit the model so that it meets a business requirement of the person. An edit changes real"
    " predictions, so apply only what the person asked for. Work in this order: read the graph with get_graph and"
    " list_bins, call preview_edit, read the report and the effect on the data, call apply_edit with the reason, and"
    " tell the person what changed. The server edits a copy and never changes the model file. If the server has"
    " save_model, call it only when the person wants the edited model saved."
)
_EDIT_WORKFLOW = (
    "Working order: read the graph (get_graph, list_bins), call preview_edit, read the report and the effect on the"
    " data, call apply_edit with the reason, and tell the person what changed. An edit changes real predictions, so"
    " apply only what the person asked for. If several edits meet the requirement, propose the one that changes the"
    " fewest scores. A direct edit does not re-centre the graph: the report shows the shift of the weighted mean of"
    " the graph, and the edit Recenter moves that mean into the intercept with no change of any prediction."
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
    _register_read_tools(server, store)
    _register_explain_tools(server, store)
    _register_bin_tools(server, store)
    _register_history_tools(server, store)
    if store.output_directory is not None:
        _register_save_tool(server, store)
    if llm is not None:
        _register_llm_tools(server, store, llm)
    return server


def _register_read_tools(server: MCPServer, store: ModelStore) -> None:
    """Register the tools and the resource that read the model.

    Parameters
    ----------
    server: MCPServer
        The server to add them to.
    store: ModelStore
        The session.

    """

    @server.tool()
    def get_model_summary(revision: str = "current") -> str:
        """Summarize the model: its kind, outputs or classes, intercept and number of features.

        Call this first. It also names the interaction terms that the other tools skip. Give revision to read the
        model as it was at an earlier edit: ``original``, ``current`` or the id of a revision of list_edits. A read of
        a revision never changes the model.

        Returns
        -------
        str
            JSON with the kind, outputs, classes, score_meaning, intercept, feature_count, skipped_terms and session
            (the revision that the call read, the head, and how many edits the head holds).

        """
        with _tool_errors():
            return _json(attrs.asdict(read_tools.model_summary(store, revision)))

    @server.tool()
    def list_features(revision: str = "current") -> str:
        """List the features of the model with their type and importance, the most important first.

        Call this to choose the feature to read with get_graph. Give revision to read an earlier edit of the model.

        Returns
        -------
        str
            JSON with the session (the revision that the call read, the head, and how many edits the head holds) and
            the list of features, each with its name, type and importance.

        """
        with _tool_errors():
            return _json(attrs.asdict(read_tools.list_features(store, revision)))

    @server.tool()
    def get_graph(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] -- one parameter for each option of the tool
        feature: str,
        revision: str = "current",
        normalization: Literal["none", "min_max", "z_score"] = "none",
        x_axis_precision: int | None = None,
        y_axis_precision: int | None = None,
        max_tokens: int = 3000,
    ) -> str:
        """Read the graph of one feature: the score of the model for each value or interval of the feature.

        Use list_features for the names. A high score raises the prediction. Lower max_tokens to get a shorter text.
        Give revision to read the graph as it was at an earlier edit, for example ``original`` to compare it with the
        current graph.

        Returns
        -------
        str
            A line with the revision and the number of edits that the head holds, then the text of the graph.

        """
        with _tool_errors():
            return read_tools.graph_text(
                store,
                feature,
                revision=revision,
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


def _register_explain_tools(server: MCPServer, store: ModelStore) -> None:
    """Register the tool that explains one prediction.

    Parameters
    ----------
    server: MCPServer
        The server to add it to.
    store: ModelStore
        The session.

    """

    @server.tool()
    def explain_prediction(
        instance: Mapping[str, FeatureValue] | None = None,
        row: int | None = None,
        confidence_level: float = 0.95,
        revision: str = "current",
    ) -> str:
        """Explain one prediction of a classifier or a regressor with exact facts.

        A regressor's explanation keeps the answer to the question "is the predicted value above the average
        prediction?". Give the value of every feature in instance, or give the number of a row of the data that the
        server loaded. The explanation uses the model at revision, which is the current model with all edits by
        default.

        Returns
        -------
        str
            JSON with the session (the revision that the call read), the facts and their texts. The facts have the
            counterfactual (changes that flip the prediction, the fewest ones when ``minimality`` proves it), the
            sufficient reason (the features that fix the prediction) and the share of completions with the same
            prediction.

        """
        with _tool_errors():
            request = explain_tools.PredictionRequest(
                instance=instance,
                row=row,
                confidence_level=confidence_level,
                revision=revision,
            )
            explanation = explain_tools.explain_prediction(store, request)
            return _json(attrs.asdict(explanation))


def _register_bin_tools(server: MCPServer, store: ModelStore) -> None:
    """Register the tools that list the bins and that preview or apply one edit.

    Parameters
    ----------
    server: MCPServer
        The server to add them to.
    store: ModelStore
        The session.

    """
    edit_help = edit_tools.edit_kinds_help()

    @server.tool()
    def list_bins(feature: str, revision: str = "current") -> str:
        """List the bins of one feature with the training weight and the score of each bin.

        An edit must name bins that exist, so read the bins first. A continuous bin has its edges. The missing bin and
        the unseen bin have no number, and an edit addresses them by name. The result also names the pair terms that
        use the feature, because an edit changes the graph of the feature only.

        Returns
        -------
        str
            JSON with the session, the bins (number, label, edges, score, weight, weight_share), the missing bin, the
            unseen bin and the pair terms.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.list_bins(store, feature, revision)))

    @server.tool(description=f"{_EDIT_WORKFLOW}\n\nCompute the report of one edit and change nothing.\n\n{edit_help}")
    def preview_edit(edit: dict[str, JSONValue], revision: str = "current") -> str:
        """Compute the report of one edit and change nothing.

        Returns
        -------
        str
            JSON with the report (the changed bins with their scores before and after, the share of the training
            weight that they hold, the shift of the weighted mean, and the pair terms that still use the feature),
            its text, and the effect on the rows of the data. The effect is empty, with a note, if the server has no
            data.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.preview_edit(store, edit_tools.parse_edit(edit), revision)))

    @server.tool(
        description=(
            f"{_EDIT_WORKFLOW}\n\nApply one edit to the session copy of the model, and return the same result as"
            " preview_edit. The reason is the business requirement that the edit serves, in one or two sentences."
            " The history keeps it, and a person reads it later. Pass expected_revision, the head that you read, so"
            " that the call fails if another call changed the model meanwhile.\n\n"
            f"{edit_help}"
        ),
    )
    def apply_edit(edit: dict[str, JSONValue], reason: str, expected_revision: str | None = None) -> str:
        """Apply one edit with the reason for it.

        Returns
        -------
        str
            JSON with the report, its text, the effect on the data and the session (the new head).

        """
        with _tool_errors():
            result = edit_tools.apply_edit(store, edit_tools.parse_edit(edit), reason, expected_revision)
            return _json(attrs.asdict(result))


def _register_history_tools(server: MCPServer, store: ModelStore) -> None:
    """Register the tools that move in the history and that list it.

    Parameters
    ----------
    server: MCPServer
        The server to add them to.
    store: ModelStore
        The session.

    """

    @server.tool()
    def undo_edit(expected_revision: str | None = None) -> str:
        """Undo the last applied edit. The edit stays in the redo stack until a new edit drops it.

        Pass expected_revision, the head that you read, so that the call fails if another call changed the model.

        Returns
        -------
        str
            JSON with the session after the undo.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.undo(store, expected_revision)))

    @server.tool()
    def redo_edit(expected_revision: str | None = None) -> str:
        """Apply the edit that the last undo moved to the redo stack.

        Pass expected_revision, the head that you read, so that the call fails if another call changed the model.

        Returns
        -------
        str
            JSON with the session after the redo.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.redo(store, expected_revision)))

    @server.tool()
    def checkout_edit(revision: str, expected_revision: str | None = None) -> str:
        """Set the model to its state after one commit of the history. The later edits stay in the redo stack.

        Use list_edits for the ids. A read tool with the argument revision shows an earlier model without this call.

        Returns
        -------
        str
            JSON with the session after the checkout.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.checkout(store, revision, expected_revision)))

    @server.tool()
    def list_edits(*, include_replay_json: bool = False) -> str:
        """List the history of the session as a table of commits.

        Each commit has an id, a time, the feature, the type of the edit, a description, the reason and whether a
        person reviewed it. Use it to tell the person what you changed. Set include_replay_json to get the JSON that
        the option --edits of the server reads.

        Returns
        -------
        str
            JSON with the session, the commits, the ids of the undone edits and, on request, the replay JSON.

        """
        with _tool_errors():
            return _json(attrs.asdict(edit_tools.list_edits(store, include_replay_json=include_replay_json)))


def _register_save_tool(server: MCPServer, store: ModelStore) -> None:
    """Register the tool that saves the edited model. The user turns it on with ``--output-dir``.

    The head and the question are resolvers, so the SDK asks the person in the way that the protocol version needs.
    The head comes first and is fixed for the whole call, so the question and the save describe the same revision.

    Parameters
    ----------
    server: MCPServer
        The server to add it to.
    store: ModelStore
        The session.

    """

    def head() -> str:
        with _tool_errors():
            return save_tools.current_head(store)

    def ask_person(
        message: str,
        expected_revision: str | None,
        revision: Annotated[str, Resolve(head)],
        ctx: Context,
    ) -> Elicit[save_tools.Confirmation] | None:
        with _tool_errors():
            text = save_tools.question(store, message, revision, expected_revision)
        # a client with no form elicitation cannot ask, and a bare `elicitation` capability counts as form support
        capability = ctx.client_capabilities.elicitation if ctx.client_capabilities is not None else None
        cannot_ask = capability is None or (capability.form is None and capability.url is not None)
        # nobody is asked if no edit needs a review, or if the client cannot ask
        return None if text is None or cannot_ask else Elicit(text, save_tools.Confirmation)

    @server.tool(
        description=(
            "Save the edited model and its history into the output directory that the user gave at start. Tell the"
            " person what changed before you call it. The message is the change that the saved model holds, for"
            " example the business requirement that the edits serve. The server asks the person to confirm the edits"
            " and their reasons. If the person refuses, nothing is saved, so ask the person before you try again. If"
            " the client cannot ask, the server saves the model and marks each edit in the history as not reviewed by"
            " a person, and you tell the person that. A save never overwrites a file, and it never changes the model"
            " file of the server. Pass expected_revision, the head that you read, so that the call fails if the model"
            " changed meanwhile."
        ),
    )
    async def save_model(
        message: str,
        answer: Annotated[save_tools.Confirmation | None, Resolve(ask_person)],
        revision: Annotated[str, Resolve(head)],
        expected_revision: str | None = None,  # ruff: ignore[unused-function-argument] -- the resolver ask_person reads it by name
    ) -> str:
        """Save the edited model with the message of the change.

        Returns
        -------
        str
            JSON with the file names, who confirmed the save (``confirmation``), the commit of the save and the edits
            that it holds.

        """
        confirmed = None if answer is None else answer.confirm
        with _tool_errors():
            result = await asyncio.to_thread(save_tools.save_model, store, message, revision, confirmed=confirmed)
            return _json(attrs.asdict(result))


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
    output_dir: Path | None = None,
    edits: Path | None = None,
) -> None:
    """Load the EBM and run the MCP server.

    Parameters
    ----------
    model: Path
        EBM in a skops file, which runs no code, or in a pickled or joblib file, which does.
    data: Path, optional
        Parquet or CSV file with the rows for the row argument.
    llm: str, optional
        PROVIDER/MODEL of the LLM tools. Without it the server has none.
    transport: Literal["stdio", "streamable-http"]
        How the server talks to the client.
    host: str
        The address that the ``streamable-http`` transport listens on. Use ``0.0.0.0`` to accept other computers.
    port: int
        The port that the ``streamable-http`` transport listens on.
    output_dir: Path, optional
        The directory where the tool ``save_model`` writes the edited model and its history. The server creates it if it
        does not exist. Without it, the server has no ``save_model`` tool.
    edits: Path, optional
        The history file of an earlier session (``*.history.json``). The server replays it, so the work goes on. The
        server refuses a history that started from another model.

    """
    # structlog prints to stdout by default, and a stdio server speaks JSON-RPC there, so a log line would corrupt it
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    store = ModelStore(ebm=load_model(model), data=load_data(data) if data else None, output_directory=output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if edits is not None:
        store.session = EditSession.replay(store.original, edits.read_bytes())
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
