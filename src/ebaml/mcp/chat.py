"""Chat in the terminal with an agent that explains an EBM.

The ``ebaml-chat`` command starts ``ebaml-mcp`` for the model and the data that you give, and lets an LLM answer your
questions with the tools of the server. Ask about the model, or about one row of the data, and ask follow-up questions.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts.validators import Path as PathValidator

from ebaml.llm.descriptors import chat_with_mcp
from ebaml.llm.options import ModelOptions

# `-c` runs the entry point of the server with the Python of the chat, so the server needs no script on the path. The
# program text is the first argument after `-c`, and the arguments of the server follow it.
_SERVER_PROGRAM = "from ebaml.mcp.server import main; main()"


def server_arguments(model: Path, data: Path | None) -> list[str]:
    """Give the arguments that start the server of a model, to put after the Python command.

    Parameters
    ----------
    model: Path
        The file of the EBM.
    data: Path, optional
        The Parquet or CSV file with the rows that the chat can explain.

    Returns
    -------
    list[str]
        The arguments of the Python command that runs ``ebaml-mcp``.

    """
    arguments = ["-c", _SERVER_PROGRAM, "--model", str(model)]
    if data is not None:
        arguments += ["--data", str(data)]
    return arguments


# a file that does not exist is a usage error, so the user sees it before BAML starts the server
_EXISTING_FILE = cyclopts.Parameter(validator=PathValidator(exists=True, file_okay=True, dir_okay=False))

app: cyclopts.App = cyclopts.App(
    name="ebaml-chat",
    help="Chat with an agent that explains a fitted EBM.",
    result_action="return_value",
)


@app.default
def chat(
    *,
    model: Annotated[Path, _EXISTING_FILE],
    data: Annotated[Path | None, _EXISTING_FILE] = None,
    llm: str | None = None,
    max_steps: int = 12,
) -> int:
    """Start the chat. An empty line ends it.

    Parameters
    ----------
    model: Path
        EBM in a skops file, which runs no code, or in a pickled or joblib file, which does.
    data: Path, optional
        Parquet or CSV file with the rows to explain. Ask about a row by its number.
    llm: str, optional
        PROVIDER/MODEL of the agent. The default is OpenRouter's free model, which needs OPENROUTER_API_KEY.
    max_steps: int
        The most rounds of tool calls for one answer.

    Returns
    -------
    int
        The number of questions that the chat answered.

    """
    return chat_with_mcp(
        sys.executable,
        server_arguments(model, data),
        ModelOptions.parse(llm) if llm else None,
        max_steps=max_steps,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``ebaml-chat`` command. It returns nothing, because a console script passes the result to ``sys.exit``.

    Parameters
    ----------
    argv: Sequence[str], optional
        The arguments. The default is the command line.

    """
    app(None if argv is None else list(argv))
