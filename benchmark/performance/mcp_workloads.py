"""The workloads of the MCP server: one call of each tool on the Titanic model, through an in-memory client.

A round starts an in-memory session with a new server and calls one tool, as an agent does for its first call. The
workload ``mcp.session`` lists the tools only, so it measures the start of the session that every other round includes.
A tool that changes the edit session gets a new store before each round, outside the measured time.
"""

import asyncio
from collections.abc import Callable, Mapping

import narwhals as nw
from mcp import Client

from benchmark.performance.llm_workloads import default_reply, difficulty_reply
from benchmark.performance.models import TARGET
from benchmark.performance.workload_types import Context, Round, Workload
from ebaml import Shift, WholeGraph
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server

MODEL = "titanic"
SHIFT = {"kind": "Shift", "feature": "age", "selection": {"kind": "WholeGraph"}, "delta": 1.0}
# the arguments of each tool that leaves the edit session as it is
READ_TOOLS: Mapping[str, dict[str, object]] = {
    "get_model_summary": {},
    "list_features": {},
    "get_graph": {"feature": "age"},
    "get_graph_image": {"feature": "age"},
    "explain_prediction": {"row": 0},
    "list_bins": {"feature": "age"},
    "preview_edit": {"edit": SHIFT},
    "list_edits": {},
    "check_constraints": {"constraints": [{"kind": "Unused", "feature": "age"}]},
    "get_edit_effect": {},
    "get_affected_profile": {},
}
LLM_TOOLS: Mapping[str, dict[str, object]] = {
    "describe_feature": {"feature": "age"},
    "describe_model": {},
    "propose_difficulties": {"row": 0},
}


def _store(context: Context, *, edits: int = 0, undone: int = 0) -> ModelStore:
    """Build the store of a server on the Titanic model and its data.

    Parameters
    ----------
    context: Context
        The context of the run.
    edits: int
        The number of shifts to apply before the round.
    undone: int
        The number of those shifts to undo before the round.

    Returns
    -------
    ModelStore
        The store, with the target as its label column and a new output directory.

    Raises
    ------
    RuntimeError
        If the model has no edit session.

    """
    fitted = context.model(MODEL)
    data = fitted.data.features.with_columns(**{TARGET: fitted.data.target})
    store = ModelStore(
        ebm=fitted.ebm,
        data=nw.from_native(data),
        output_directory=context.new_directory(),
        label_column=TARGET,
    )
    session = store.session
    if session is None:
        msg = f"The model {MODEL} has no edit session."
        raise RuntimeError(msg)
    for number in range(edits):
        session.apply(Shift("age", WholeGraph(), 1.0), f"shift {number}")
    for _ in range(undone):
        session.undo()
    return store


async def _call(store: ModelStore, context: Context, name: str | None, arguments: Mapping[str, object]) -> object:
    """Start an in-memory session and call one tool, or list the tools.

    Parameters
    ----------
    store: ModelStore
        The store of the server.
    context: Context
        The context of the run, with the model of the fake LLM.
    name: str | None
        The tool, or ``None`` to list the tools.
    arguments: Mapping[str, object]
        The arguments of the tool.

    Returns
    -------
    object
        The result of the call.

    Raises
    ------
    RuntimeError
        If the tool answers with an error, so a broken workload fails and does not measure an error path.

    """
    async with Client(create_server(store, context.llm)) as session:
        if name is None:
            return await session.list_tools()
        result = await session.call_tool(name, dict(arguments))
    if result.is_error:
        msg = f"The tool {name} failed: {result.content}"
        raise RuntimeError(msg)
    return result


def _shared_store(name: str | None, arguments: Mapping[str, object]) -> Callable[[Context], Round]:
    """Measure a tool that leaves the session as it is, on one store for every round.

    Parameters
    ----------
    name: str | None
        The tool, or ``None`` to list the tools.
    arguments: Mapping[str, object]
        The arguments of the tool.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        context.reply_with(difficulty_reply if name == "propose_difficulties" else default_reply)
        store = _store(context)
        return Round(lambda: asyncio.run(_call(store, context, name, arguments)))

    return prepare


def _new_store(name: str, arguments: Mapping[str, object], *, edits: int, undone: int) -> Callable[[Context], Round]:
    """Measure a tool that changes the session, on a new store for each round.

    Parameters
    ----------
    name: str
        The tool.
    arguments: Mapping[str, object]
        The arguments of the tool.
    edits: int
        The shifts to apply before each round.
    undone: int
        The shifts to undo before each round.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        context.reply_with(default_reply)

        def setup() -> tuple[tuple[object, ...], dict[str, object]]:
            return (_store(context, edits=edits, undone=undone),), {}

        return Round(lambda store: asyncio.run(_call(store, context, name, arguments)), setup)

    return prepare


def mcp_workloads() -> list[Workload]:
    """Make the workloads of the MCP server.

    Returns
    -------
    list[Workload]
        ``mcp.session``, and one workload for each tool.

    """
    return [
        Workload("mcp.session", "mcp", _shared_store(None, {})),
        *(Workload(f"mcp.{name}", "mcp", _shared_store(name, arguments)) for name, arguments in READ_TOOLS.items()),
        Workload("mcp.apply_edit", "mcp", _new_store("apply_edit", {"edit": SHIFT, "reason": "r"}, edits=0, undone=0)),
        Workload("mcp.undo_edit", "mcp", _new_store("undo_edit", {}, edits=1, undone=0)),
        Workload("mcp.redo_edit", "mcp", _new_store("redo_edit", {}, edits=1, undone=1)),
        Workload("mcp.checkout_edit", "mcp", _new_store("checkout_edit", {"revision": "original"}, edits=1, undone=0)),
        Workload(
            "mcp.save_model",
            "mcp",
            _new_store("save_model", {"message": "The benchmark save"}, edits=1, undone=0),
        ),
        *(Workload(f"mcp.{name}", "mcp", _shared_store(name, arguments)) for name, arguments in LLM_TOOLS.items()),
    ]
