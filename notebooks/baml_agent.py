"""A marimo notebook that runs the BAML agent: a language model that explains an EBM through the ``ebaml-mcp`` tools."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shutil
    import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the notebook runs the baml command, which is the interface that it shows
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import orjson
    import polars as pl
    import skops.io
    from interpret.glassbox import ExplainableBoostingClassifier

    return (
        ExplainableBoostingClassifier,
        Path,
        mo,
        np,
        orjson,
        os,
        pl,
        shutil,
        skops,
        subprocess,
        sys,
        tempfile,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # The BAML agent

    The [MCP server](mcp-notebook.md) gives a client the tools to read an EBM. This notebook runs a client that is a
    language model. The BAML function `explain_with_mcp` starts `ebaml-mcp`, offers its tools to the model, lets the
    model call them, and returns the report together with the list of tool calls. You learn how to run the agent from
    the `baml` command, how to read what it did, and how it reports a setup problem.

    - You need no data download. The notebook fits a small EBM on made-up data.
    - You need the BAML toolchain (`baml`, version 0.23 or later) and the `mcp` extra: `pip install "ebaml[mcp]"`.
    - Every step runs without a key, except the last button. It sends real requests to an LLM. The default model
      needs `OPENROUTER_API_KEY`.
    """)


@app.cell
def _(Path, mo, os, shutil):
    project_root = Path(__file__).resolve().parent.parent
    # BAML puts its tools in ~/.baml/bin, and an older `baml` has no `describe`
    baml_path = os.pathsep.join([str(Path.home() / ".baml" / "bin"), os.environ.get("PATH", "")])
    baml_command = shutil.which("baml", path=baml_path)
    mo.stop(
        baml_command is None,
        mo.callout(
            mo.md("The `baml` command is not on the path. Install the BAML toolchain, then reopen the notebook."),
            kind="danger",
        ),
    )
    mo.callout(mo.md(f"Found `baml` at `{baml_command}`."), kind="success")
    return baml_command, baml_path, project_root


@app.cell
def _(baml_command, baml_path, os, project_root, subprocess):
    def run_baml(*arguments: str, drop_key: bool = False, timeout: float = 600) -> str:
        """Run ``baml`` in the project and give its output.

        The key of the environment is replaced by a dummy, because the offline tests need a variable that exists.
        With ``drop_key``, the variable is removed, to show the error of a missing key.

        Returns
        -------
        str
            The standard output and the standard error of the command, without BAML warnings.

        """
        environment = {**os.environ, "PATH": baml_path}
        if drop_key:
            environment.pop("OPENROUTER_API_KEY", None)
        completed = subprocess.run(
            [baml_command, *arguments],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        lines = (completed.stdout + completed.stderr).splitlines()
        return "\n".join(line for line in lines if "warning[" not in line and line.strip())

    return (run_baml,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Fit and save a model

    The agent reaches the model through the server, so the server needs a saved model and a CSV file with rows. The
    notebook fits a binary classifier on three features and saves it with `skops.io`.
    """)


@app.cell
def _(ExplainableBoostingClassifier, Path, np, pl, skops, tempfile):
    workdir = Path(tempfile.mkdtemp(prefix="ebaml-agent-"))
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
    data_file = workdir / "train.csv"
    skops.io.dump(classifier, model_file)
    people.write_csv(data_file)
    return data_file, model_file


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The function

    `baml describe` prints the signature of a BAML function. `command` and `args` start the MCP server. The model
    does not appear in the signature, because the default is the model of the BAML file. Pass `model` to choose
    another one.
    """)


@app.cell
def _(mo, run_baml) -> None:
    mo.md(f"```baml\n{run_baml('describe', 'explain_with_mcp', '--budget', '30').split('//#')[0].strip()}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Test the loop with no key

    The BAML test `explain_with_tools` runs the same loop as the agent, but with a scripted model and a fake tool.
    The scripted model asks for the tool `get_model_summary` in its first turn and writes the report in its second
    turn. The test checks that the loop ran the tool, that `tool_calls` names it, and that the usage adds up. The
    test sends no request, so it needs no key. Use it after each change of the prompts or of the loop.
    """)


@app.cell
def _(mo, run_baml) -> None:
    mo.md(f"```text\n{run_baml('test', '-i', 'root::explain_with_tools')}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Two setup problems and their fixes

    Both problems show in the first seconds, and neither costs money. The agent starts the server before it sends the
    first request, so a problem with the key shows after the server started.

    1. **A missing key.** The error says which variable the provider needs. Set `OPENROUTER_API_KEY`, or pass a
       `model` with another provider.
    2. **A server that does not start.** The error names the command. Give the path of `ebaml-mcp` in `command`.
    """)


@app.cell
def _(Path, data_file, mo, model_file, orjson, run_baml, sys):
    # the ebaml-mcp script that `pip install "ebaml[mcp]"` puts next to the Python of this notebook
    server_command = str(Path(sys.executable).parent / "ebaml-mcp")
    server_arguments = ["--model", str(model_file), "--data", str(data_file)]

    def explain_expression(
        question: str,
        command: str,
        arguments: list[str],
        model_literal: str = "null",
        max_steps: int = 12,
    ) -> str:
        """Write the BAML call of the agent. JSON text is a valid BAML string or list literal.

        Returns
        -------
        str
            The expression for ``baml run -e``.

        """
        return (
            f"root.explain_with_mcp({orjson.dumps(question).decode()}, {orjson.dumps(command).decode()}, "
            f"{orjson.dumps(arguments).decode()}, model = {model_literal}, max_steps = {max_steps})"
        )

    missing_key_output = run_baml(
        "run",
        "-e",
        explain_expression("Which features matter?", server_command, server_arguments),
        drop_key=True,
    )
    missing_server_output = run_baml(
        "run",
        "-e",
        explain_expression("Which features matter?", "no-such-server", []),
    )
    missing_key_message = missing_key_output.splitlines()[-1]
    missing_server_message = missing_server_output.splitlines()[-1]
    mo.md(
        f"**Missing key**\n\n```text\n{missing_key_message}\n```\n\n"
        f"**Server that does not start**\n\n```text\n{missing_server_message}\n```",
    )
    return explain_expression, server_arguments, server_command


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Run the agent

    The last cells send **real requests**. The agent makes one request for each step, and a step is one round of tool
    calls. A report usually takes three to six requests. The default model is the free model of OpenRouter, so the
    cost is zero, but the free model can be slow or rate limited.

    Type a question. To use another model, type `PROVIDER/MODEL` (for example `openai/gpt-4o-mini`), as the page
    "Choosing the model" explains. Leave the field empty for the default.
    """)


@app.cell
def _(mo, os):
    key_from_env = bool(os.environ.get("OPENROUTER_API_KEY"))
    api_key_input = None if key_from_env else mo.ui.text(kind="password", label="OpenRouter API key", full_width=True)
    key_status = (
        mo.callout("Loaded `OPENROUTER_API_KEY` from the environment.", kind="success")
        if key_from_env
        else mo.vstack([mo.callout("`OPENROUTER_API_KEY` is not set in the environment.", kind="warn"), api_key_input])
    )
    key_status
    return (api_key_input,)


@app.cell
def _(mo):
    question_input = mo.ui.text_area(
        value="Which features drive the model, and is any effect surprising?",
        label="Question",
        full_width=True,
    )
    model_input = mo.ui.text(placeholder="openai/gpt-4o-mini", label="Model (optional)")
    step_slider = mo.ui.slider(3, 20, value=12, label="Most steps")
    agent_button = mo.ui.run_button(label="Run the agent")
    mo.vstack([question_input, model_input, step_slider, agent_button])
    return agent_button, model_input, question_input, step_slider


@app.cell
def _(
    agent_button,
    api_key_input,
    explain_expression,
    mo,
    model_input,
    orjson,
    os,
    question_input,
    run_baml,
    server_arguments,
    server_command,
    step_slider,
):
    mo.stop(not agent_button.value, mo.md("*Press the button to send the requests.*"))
    typed_key = api_key_input.value if api_key_input is not None else ""
    if typed_key:
        # a typed key lives in this process only, and the baml child process inherits it
        os.environ["OPENROUTER_API_KEY"] = typed_key
    mo.stop(not os.environ.get("OPENROUTER_API_KEY"), mo.callout("Set `OPENROUTER_API_KEY` first.", kind="warn"))
    chosen_model = model_input.value.strip()
    provider, _, model_name = chosen_model.partition("/")
    model_literal = (
        f"ModelSpec {{ provider: {orjson.dumps(provider).decode()}, model: {orjson.dumps(model_name).decode()} }}"
        if model_name
        else "null"
    )
    agent_output = run_baml(
        "run",
        "-e",
        explain_expression(
            question_input.value,
            server_command,
            server_arguments,
            model_literal,
            max_steps=step_slider.value,
        ),
        "--output-format",
        "json",
    )
    return (agent_output,)


@app.cell
def _(agent_output, mo, orjson):
    try:
        agent_result = orjson.loads(agent_output[agent_output.index("{") :])
    except (ValueError, orjson.JSONDecodeError):
        mo.stop(
            predicate=True,
            output=mo.callout(mo.md(f"The agent failed:\n\n```text\n{agent_output}\n```"), kind="danger"),
        )
    mo.vstack(
        [
            mo.md("### Tools that the model called, in order"),
            mo.md("\n".join(f"{step}. `{name}`" for step, name in enumerate(agent_result["tool_calls"], start=1))),
            mo.md(
                f"The model sent **{agent_result['usage']['input_tokens']}** input tokens and wrote "
                f"**{agent_result['usage']['output_tokens']}** output tokens.",
            ),
            mo.md("### Report"),
            mo.md(agent_result["report"]),
        ],
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Read the result

    - `tool_calls` shows the evidence. A good run calls `get_model_summary` and `list_features` first, then
      `get_graph` for the features that matter. A report with no tool call rests on nothing, so do not trust it.
    - Compare each number of the report with the output of the tool in the
      [MCP server notebook](mcp-notebook.md). The tools are exact, and the model only puts them in words.
    - The same loop runs on tools that you supply. `explain_with_tools` takes a list of `ai.tools.Tool`, so a later
      server with edit tools needs no change of the loop.
    """)


if __name__ == "__main__":
    app.run()
