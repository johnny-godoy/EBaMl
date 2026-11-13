"""A marimo notebook that chats with an agent that explains an EBM, through the real ``ebaml-chat`` command."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import atexit
    import os
    import select
    import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the notebook runs the ebaml-chat command, which is the interface that it shows
    import sys
    import tempfile
    import threading
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    import skops.io
    from interpret.glassbox import ExplainableBoostingClassifier

    return (
        ExplainableBoostingClassifier,
        Path,
        atexit,
        mo,
        np,
        os,
        pl,
        select,
        skops,
        subprocess,
        sys,
        tempfile,
        threading,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Chat with an EBM

    `ebaml-chat` opens a chat with an agent that explains a fitted EBM. You ask about the model, or about one row of
    your data, and you ask follow-up questions. The agent is a language model that reads the model through the tools of
    the [MCP server](mcp-notebook.md). This notebook runs the real `ebaml-chat` command and puts a chat box in front
    of it. You learn what to ask, how the agent answers, and how to check an answer.

    - You need no data download. The notebook fits a small EBM on made-up data.
    - The notebook starts the chat without a key. Each message that you send needs an LLM, so it needs
      `OPENROUTER_API_KEY`, or a key that you type below.
    - The notebook needs the BAML toolchain (`baml`, version 0.23 or later) and the `mcp` extra:
      `pip install "ebaml[mcp]"`.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Fit and save a model

    The chat reads a saved model and, if you want to ask about rows, a CSV file with the rows. The notebook fits a
    binary classifier on three features and saves it with `skops.io`. It has no interaction terms, because the
    explain tool works for a model without them.
    """)


@app.cell
def _(ExplainableBoostingClassifier, Path, np, pl, skops, tempfile):
    workdir = Path(tempfile.mkdtemp(prefix="ebaml-chat-"))
    random_generator = np.random.default_rng(0)
    people = pl.DataFrame(
        {
            "age": random_generator.integers(18, 80, 500),
            "income": random_generator.normal(50, 15, 500).round(1),
            "city": random_generator.choice(["north", "south"], 500),
        },
    )
    score = (
        (people["age"].to_numpy() - 40) / 20
        + (people["income"].to_numpy() - 50) / 15
        + (people["city"].to_numpy() == "north")
        + random_generator.normal(size=500)
    )
    approval_threshold = 0.5
    approved = (score > approval_threshold).astype(int)
    classifier = ExplainableBoostingClassifier(feature_names=people.columns, interactions=0, random_state=0)
    classifier.fit(people, approved)
    model_file = workdir / "model.skops"
    data_file = workdir / "train.parquet"
    skops.io.dump(classifier, model_file)
    people.write_parquet(data_file)
    return data_file, model_file


@app.cell
def _(data_file, mo, model_file) -> None:
    mo.md(f"""
    ## The command

    In a terminal, the same chat starts with one command. The model and the data are the files of this notebook:

    ```bash
    ebaml-chat --model {model_file} --data {data_file}
    ```

    Add `--llm PROVIDER/MODEL` to choose another model (see "Choosing the model" in the documentation), and
    `--max-steps NUMBER` to limit the rounds of tool calls for one answer. An empty line ends the chat.
    """)


@app.cell
def _(mo, os):
    key_from_env = bool(os.environ.get("OPENROUTER_API_KEY"))
    api_key_input = None if key_from_env else mo.ui.text(kind="password", label="OpenRouter API key", full_width=True)
    key_status = (
        mo.callout("Loaded `OPENROUTER_API_KEY` from the environment.", kind="success")
        if key_from_env
        else mo.vstack(
            [
                mo.callout(
                    "`OPENROUTER_API_KEY` is not set. You can start the chat, but each message shows the error of a "
                    "missing key. Type a key and press Enter to restart the chat with it.",
                    kind="warn",
                ),
                api_key_input,
            ],
        )
    )
    key_status
    return (api_key_input,)


@app.cell
def _(api_key_input, os):
    # a typed key lives in this process only, and the chat that the notebook starts inherits it
    api_key = (
        api_key_input.value
        if api_key_input is not None and api_key_input.value
        else os.environ.get("OPENROUTER_API_KEY", "")
    )
    return (api_key,)


@app.cell
def _(atexit, os, select):
    prompt_marker = b"you> "
    reply_timeout_seconds = 300

    def read_until_prompt(process, timeout: float = reply_timeout_seconds) -> str:
        """Read the output of the chat up to its next prompt.

        Returns
        -------
        str
            The text before the prompt. A chat that ends before the prompt gives the text that it wrote.

        """
        buffer = b""
        while not buffer.endswith(prompt_marker):
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            chunk = os.read(process.stdout.fileno(), 4096) if ready else b""
            if not chunk:
                break
            buffer += chunk
        return buffer.removesuffix(prompt_marker).decode(errors="replace").strip()

    def stop_chat(process) -> None:
        """Stop a chat process, and ignore one that has already ended."""
        if process.poll() is None:
            process.terminate()

    def register_stop(process) -> None:
        """Stop the chat process when the notebook ends."""
        atexit.register(stop_chat, process)

    return read_until_prompt, register_stop


@app.cell
def _(
    api_key,
    data_file,
    mo,
    model_file,
    os,
    read_until_prompt,
    register_stop,
    subprocess,
    sys,
    threading,
):
    # the command of the terminal, started with the Python of this notebook
    chat_environment = {**os.environ, "OPENROUTER_API_KEY": api_key} if api_key else dict(os.environ)
    chat_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from ebaml.mcp.chat import main; main()",
            "--model",
            str(model_file),
            "--data",
            str(data_file),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=chat_environment,
    )
    register_stop(chat_process)
    error_lines: list[str] = []

    def collect_errors() -> None:
        """Keep each line that the chat writes to its error stream, for the tool report of the next answer."""
        error_lines.extend(raw_line.decode(errors="replace").rstrip() for raw_line in chat_process.stderr)

    threading.Thread(target=collect_errors, daemon=True).start()
    greeting = read_until_prompt(chat_process, timeout=120)
    mo.md(
        f"The chat started its server and wrote its first line, with no request to an LLM:\n\n```text\n{greeting}\n```",
    )
    return chat_process, error_lines


@app.cell
def _(chat_process, error_lines, mo, read_until_prompt):
    def answer(messages, _config) -> str:
        """Send the last message of the chat box to ``ebaml-chat`` and give its answer, with the tools that it called.

        Returns
        -------
        str
            The answer in Markdown.

        """
        error_lines.clear()
        question = " ".join(str(messages[-1].content).split())
        chat_process.stdin.write(question.encode() + b"\n")
        chat_process.stdin.flush()
        reply = read_until_prompt(chat_process)
        tools = [line.removeprefix("(tools: ").removesuffix(")") for line in error_lines if line.startswith("(tools: ")]
        return reply + (f"\n\n*Tools called: {tools[-1]}*" if tools else "")

    chat_box = mo.ui.chat(
        answer,
        prompts=[
            "Which feature matters most, and how does it change the prediction?",
            "Explain the prediction for row 3 of the data.",
            "What would flip it?",
        ],
        max_height=600,
    )
    return (chat_box,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Ask

    The chat box below talks to the command above. **Each message sends real requests** to the LLM, one for each
    round of tool calls. A message about one row usually takes two or three requests. The default model is the free
    model of OpenRouter, so the cost is zero, but it can be slow or rate limited.

    Try the three questions in this order. The second one explains a row. The third one is a follow-up: it has no
    row in it, so the agent must use the earlier messages.
    """)


@app.cell
def _(chat_box):
    chat_box


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Check the answers

    - The line *Tools called* shows the evidence. A question about a row calls `explain_prediction`. A question about a
      feature calls `get_graph`. An answer with no tool call rests on the earlier messages only.
    - The agent remembers the earlier messages, but not the earlier tool results. It calls a tool again when it needs
      a number. Because of that, a follow-up can take one more request than the first question.
    - A turn that fails prints its error, for example the missing key, and the chat goes on. Fix the cause and ask
      again.
    - Compare each number of an answer with the output of the tools in the [MCP server notebook](mcp-notebook.md).
      The tools are exact. The model only puts them in words, and it can make a mistake there.
    """)


if __name__ == "__main__":
    app.run()
