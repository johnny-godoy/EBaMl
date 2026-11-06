"""A marimo notebook that serves an EBM to an agent with ``ebaml-mcp`` and calls the tools like a client does."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import contextlib
    import os
    import socket
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import orjson
    import polars as pl
    import skops.io
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    return (
        ClientSession,
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
        Path,
        StdioServerParameters,
        asyncio,
        contextlib,
        mo,
        np,
        orjson,
        os,
        pl,
        skops,
        socket,
        stdio_client,
        streamable_http_client,
        sys,
        tempfile,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # The MCP server

    `ebaml-mcp` gives an agent a fitted EBM through the
    [Model Context Protocol](https://modelcontextprotocol.io). This notebook starts the real server, connects to it
    as a client does, and calls each tool. You learn which tools exist, what they return, how to choose the
    transport, how the server reports a problem, and how to connect Claude Code or Claude Desktop.

    - You need no data download and no key: the notebook fits a small EBM on made-up data.
    - Everything outside the last button runs without a key and sends no request to an LLM.
    - The last section sends one real request. The default model needs `OPENROUTER_API_KEY`.
    - The notebook needs the `mcp` extra: `pip install "ebaml[mcp]"`.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Fit and save a model

    The server loads a saved model and, if you want, a CSV file with rows to explain. The notebook fits a binary
    classifier on three features and saves it with `skops.io`. A skops file runs no code when the server loads it,
    and the server trusts only the EBM classes. Pass `interactions=0`, because `explain_prediction` works for a
    model without interaction terms.
    """)


@app.cell
def _(ExplainableBoostingClassifier, Path, np, pl, skops, tempfile):
    workdir = Path(tempfile.mkdtemp(prefix="ebaml-mcp-"))
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
    return approved, data_file, model_file, people, workdir


@app.cell
def _(approved, data_file, model_file, people, mo) -> None:
    mo.md(f"""
    The notebook saved **{len(people)}** rows in `{data_file.name}` and the model in `{model_file.name}`.
    The share of approved people is **{approved.mean():.2f}**.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Start the server and list the tools

    A client starts `ebaml-mcp` as a child process and talks to it over standard input and output (the `stdio`
    transport). The helper below does the same with the SDK of the protocol. It starts a new server for each call
    and stops it when the call ends, so every call shows what a fresh client sees.
    """)


@app.cell
def _(ClientSession, Path, StdioServerParameters, contextlib, stdio_client, sys):
    # the ebaml-mcp script that `pip install "ebaml[mcp]"` puts next to the Python of this notebook
    command = str(Path(sys.executable).parent / "ebaml-mcp")

    class open_session:  # ruff: ignore[invalid-class-name] -- it reads like a function in the cells: `async with open_session(...)`
        """Start ebaml-mcp with the arguments, and give an initialized client session. Stop the server on exit."""

        def __init__(self, *arguments: str, env: dict[str, str] | None = None) -> None:
            self.parameters = StdioServerParameters(command=command, args=list(arguments), env=env)
            self.stack = contextlib.AsyncExitStack()

        async def __aenter__(self) -> ClientSession:
            read, write = await self.stack.enter_async_context(stdio_client(self.parameters))
            session = await self.stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session

        async def __aexit__(self, *exception_details: object) -> bool | None:
            return await self.stack.__aexit__(*exception_details)

    return command, open_session


@app.cell
async def _(data_file, mo, model_file, open_session):
    async with open_session("--model", str(model_file), "--data", str(data_file)) as listing_session:
        listing = await listing_session.list_tools()
    mo.md(
        "The server offers **{count}** tools without `--llm`:\n\n{rows}".format(
            count=len(listing.tools),
            rows="\n".join(f"- `{tool.name}`: {tool.description.splitlines()[0]}" for tool in listing.tools),
        ),
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Read the model

    An agent starts with `get_model_summary` and `list_features`. The summary names the kind of the model and the
    interaction terms that the tools skip. The list shows the features, the most important first.
    """)


@app.cell
async def _(data_file, model_file, open_session, orjson):
    async with open_session("--model", str(model_file), "--data", str(data_file)) as read_session:
        summary_result = await read_session.call_tool("get_model_summary", {})
        features_result = await read_session.call_tool("list_features", {})
    summary = orjson.loads(summary_result.content[0].text)
    features = orjson.loads(features_result.content[0].text)
    summary
    return features, summary


@app.cell
def _(features, mo):
    mo.ui.table(features, selection=None, label="list_features")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    Pick a feature and a token budget. `get_graph` returns the text that an LLM reads. For a continuous feature,
    the server merges bins until the estimated size is near the budget. A budget that is too small for the graph
    gives a tool error.
    """)


@app.cell
def _(features, mo):
    feature_picker = mo.ui.dropdown(
        [feature["name"] for feature in features],
        value=features[0]["name"],
        label="Feature",
    )
    budget_slider = mo.ui.slider(1000, 4000, step=500, value=2000, label="max_tokens", show_value=True)
    mo.hstack([feature_picker, budget_slider], justify="start")
    return budget_slider, feature_picker


@app.cell
async def _(budget_slider, data_file, feature_picker, mo, model_file, open_session):
    async with open_session("--model", str(model_file), "--data", str(data_file)) as graph_session:
        graph_result = await graph_session.call_tool(
            "get_graph",
            {"feature": feature_picker.value, "max_tokens": budget_slider.value},
        )
    graph_text = graph_result.content[0].text
    mo.callout(graph_text, kind="danger") if graph_result.isError else mo.md(
        f"**{len(graph_text)}** characters.\n\n```text\n{graph_text}\n```",
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Explain one prediction

    `explain_prediction` takes the number of a row of the data file (`row`), or the value of every feature
    (`instance`). It computes the facts exactly from the scores of the model: the counterfactual, the sufficient
    reason and the share of completions with the same prediction. The server calls no LLM.
    """)


@app.cell
def _(mo, people):
    row_slider = mo.ui.slider(0, len(people) - 1, value=0, label="row", show_value=True)
    confidence_slider = mo.ui.slider(0.5, 0.99, step=0.01, value=0.95, label="confidence_level", show_value=True)
    mo.hstack([row_slider, confidence_slider], justify="start")
    return confidence_slider, row_slider


@app.cell
async def _(confidence_slider, data_file, model_file, open_session, orjson, row_slider):
    async with open_session("--model", str(model_file), "--data", str(data_file)) as explain_session:
        explain_result = await explain_session.call_tool(
            "explain_prediction",
            {"row": row_slider.value, "confidence_level": confidence_slider.value},
        )
    explanation = orjson.loads(explain_result.content[0].text)
    facts = explanation["facts"]
    return explanation, facts


@app.cell
def _(facts, mo):
    prediction = facts["prediction"]
    changes = facts["counterfactual"]["changes"]
    mo.md(f"""
    The model predicts the class **{prediction["decision"]}** with a probability of **{prediction["output"]:.3f}**
    (interval {prediction["lower"]:.3f} to {prediction["upper"]:.3f}).

    - **Counterfactual.** The fewest changes that flip the prediction:
      {"; ".join(f"{c['feature']} from {c['from_value']} to {c['to_value']}" for c in changes) or "none exists"}.
    - **Sufficient reason.** These features fix the prediction: {", ".join(facts["sufficient_reason"]["features"])}.
    - **Completions.** Fix none of the features, and the share of completions with the same prediction is
      **{facts["completions"][0]["share"]:.2f}**.
    """)


@app.cell
def _(facts, mo):
    mo.ui.table(facts["contributions"], selection=None, label="Contribution of each feature")


@app.cell
def _(explanation, mo):
    mo.accordion(
        {
            "The same facts as the texts of the prompts": mo.md(
                "\n\n".join(f"**{name}**\n\n```json\n{text}\n```" for name, text in explanation["texts"].items()),
            ),
        },
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Choose the transport

    The `stdio` transport suits a client on the same computer. Use `--transport streamable-http` when the client
    runs elsewhere, for example in a container. Then choose the address with `--host` and the port with `--port`.
    The server has no login. The default host `127.0.0.1` accepts this computer only, so change it only behind a
    firewall or a proxy with authentication.

    The cell below starts the server on a free port and connects to it over HTTP.
    """)


@app.cell
async def _(
    asyncio,
    ClientSession,
    command,
    data_file,
    mo,
    model_file,
    socket,
    streamable_http_client,
):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    http_server = await asyncio.create_subprocess_exec(
        command,
        "--model",
        str(model_file),
        "--data",
        str(data_file),
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        str(free_port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    http_names: list[str] = []
    try:
        for _attempt in range(60):
            try:
                async with (
                    streamable_http_client(f"http://127.0.0.1:{free_port}/mcp") as (read, write, _),
                    ClientSession(
                        read,
                        write,
                    ) as http_session,
                ):
                    await http_session.initialize()
                    http_names = [tool.name for tool in (await http_session.list_tools()).tools]
                break
            except (OSError, ExceptionGroup):
                await asyncio.sleep(0.5)
    finally:
        http_server.terminate()
        await http_server.wait()
    mo.md(f"The HTTP server listened on `http://127.0.0.1:{free_port}/mcp` and offered: {', '.join(http_names)}.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Serve each kind of model

    The read tools work for the three kinds of EBM. `explain_prediction` works for a binary classifier without
    interaction terms only. The notebook fits a regressor, a multiclass classifier (low, mid and high income) and a
    classifier with one interaction term. Then it asks each server for the summary, and for an explanation.
    """)


@app.cell
def _(
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    Path,
    np,
    people,
    skops,
    workdir,
):
    older_than = 45
    income_target = people["income"].to_numpy()
    income_band = np.array(["low", "mid", "high"])[np.digitize(people["income"].to_numpy(), [40, 60])]
    other_models = {
        "regressor": ExplainableBoostingRegressor(feature_names=["age", "city"], interactions=0, random_state=0).fit(
            people.drop("income"),
            income_target,
        ),
        "multiclass classifier": ExplainableBoostingClassifier(
            feature_names=["age", "city"],
            interactions=0,
            random_state=0,
        ).fit(people.drop("income"), income_band),
        "classifier with an interaction": ExplainableBoostingClassifier(
            feature_names=people.columns,
            interactions=1,
            random_state=0,
        ).fit(people, (people["age"].to_numpy() > older_than) ^ (people["city"].to_numpy() == "north")),
    }
    other_instances = {
        "regressor": people.drop("income").row(0, named=True),
        "multiclass classifier": people.drop("income").row(0, named=True),
        "classifier with an interaction": people.row(0, named=True),
    }
    other_files: dict[str, Path] = {}
    for kind_name, fitted in other_models.items():
        other_files[kind_name] = workdir / f"{kind_name.replace(' ', '_')}.skops"
        skops.io.dump(fitted, other_files[kind_name])
    return other_files, other_instances


@app.cell
async def _(mo, open_session, orjson, other_files, other_instances):
    kind_rows = []
    for row_label, row_file in other_files.items():
        async with open_session("--model", str(row_file)) as kind_session:
            kind_summary = orjson.loads((await kind_session.call_tool("get_model_summary", {})).content[0].text)
            kind_features = orjson.loads((await kind_session.call_tool("list_features", {})).content[0].text)
            kind_explain = await kind_session.call_tool(
                "explain_prediction",
                {"instance": other_instances[row_label]},
            )
        kind_rows.append(
            {
                "model": row_label,
                "kind": kind_summary["kind"],
                "features with a graph": ", ".join(feature["name"] for feature in kind_features),
                "skipped terms": ", ".join(kind_summary["skipped_terms"]) or "none",
                "explain_prediction": "tool error: " + kind_explain.content[0].text.split(".")[0]
                if kind_explain.isError
                else "works",
            },
        )
    mo.ui.table(kind_rows, selection=None, label="One server for each model", wrapped_columns=["explain_prediction"])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## When something is wrong

    A problem that the user can fix gives a message with the fix. The server reports a mistake of a tool call as a
    tool error, so the agent can read it and try again. A problem at the start stops the server, and the client
    sees a closed connection.
    """)


@app.cell
async def _(data_file, mo, model_file, open_session, other_files):
    problems = {}
    async with open_session("--model", str(model_file), "--data", str(data_file)) as problem_session:
        problems["Unknown feature"] = await problem_session.call_tool("get_graph", {"feature": "height"})
        problems["Row outside the data"] = await problem_session.call_tool("explain_prediction", {"row": 9999})
        problems["Both row and instance"] = await problem_session.call_tool(
            "explain_prediction",
            {"row": 0, "instance": {"age": 30}},
        )
    async with open_session("--model", str(model_file)) as nodata_session:
        problems["Row without --data"] = await nodata_session.call_tool("explain_prediction", {"row": 0})
    async with open_session("--model", str(other_files["regressor"])) as regressor_session:
        problems["Explain a regressor"] = await regressor_session.call_tool(
            "explain_prediction",
            {"instance": {"age": 30, "city": "north"}},
        )
    problem_rows = [
        {"situation": situation, "tool error": result.isError, "message": result.content[0].text}
        for situation, result in problems.items()
    ]
    mo.ui.table(problem_rows, selection=None, label="Tool errors", wrapped_columns=["message"])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    Two more problems stop the server at the start. The server never loads a file that a tool names: only the
    command line of the user names the files. A file that holds an object other than an EBM fails with a message,
    and a skops file with another type fails because the loader trusts only the EBM classes.
    """)


@app.cell
async def _(asyncio, command, mo, skops, workdir):
    not_a_model = workdir / "not_a_model.skops"
    skops.io.dump({"this": "is a dict"}, not_a_model)
    failed_server = await asyncio.create_subprocess_exec(
        command,
        "--model",
        str(not_a_model),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, error_output = await failed_server.communicate()
    start_failure = error_output.decode().strip().splitlines()[-1]
    mo.md(f"The last line of the output of the server:\n\n```text\n{start_failure}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    The LLM tools need `--llm PROVIDER/MODEL`. A server that starts without a key for the provider answers a call
    with the steps to fix it. The cell below starts the server with `--llm openrouter/openrouter/free` and no key,
    so it sends no request.
    """)


@app.cell
async def _(data_file, mo, model_file, open_session):
    async with open_session(
        "--model",
        str(model_file),
        "--data",
        str(data_file),
        "--llm",
        "openrouter/openrouter/free",
    ) as keyless_session:
        keyless_tools = [tool.name for tool in (await keyless_session.list_tools()).tools]
        keyless_result = await keyless_session.call_tool("describe_feature", {"feature": "age"})
    mo.md(
        f"The server offered **{len(keyless_tools)}** tools. The call of `describe_feature` answered:\n\n"
        f"```text\n{keyless_result.content[0].text}\n```",
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Connect a client

    Claude Code starts the server for you. Replace the paths with the files that you saved:

    ```bash
    claude mcp add ebaml -- ebaml-mcp --model /path/to/model.skops --data /path/to/train.csv
    ```

    Claude Desktop reads an entry in `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "ebaml": {
          "command": "ebaml-mcp",
          "args": ["--model", "/path/to/model.skops", "--data", "/path/to/train.csv"]
        }
      }
    }
    ```

    A client usually starts the server with a small environment. To use `--llm`, give the client the key. In Claude
    Code, add `--env OPENROUTER_API_KEY=...` to the command.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    The last cell sends **one real request** to the LLM through `describe_feature`. The server needs
    `OPENROUTER_API_KEY`. Pass another `PROVIDER/MODEL` to `--llm` to use another model, as the documentation page
    "Choosing the model" shows.
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
def _(api_key_input, os):
    # a typed key only lives in this process; the notebook passes it to the server that it starts
    api_key = (
        api_key_input.value
        if api_key_input is not None and api_key_input.value
        else os.environ.get("OPENROUTER_API_KEY", "")
    )
    return (api_key,)


@app.cell
def _(mo):
    describe_button = mo.ui.run_button(label="Ask the LLM to describe the feature")
    describe_button
    return (describe_button,)


@app.cell
async def _(api_key, data_file, describe_button, feature_picker, mo, model_file, open_session, os):
    mo.stop(not describe_button.value, mo.md("*Press the button to send the request.*"))
    mo.stop(not api_key, mo.callout("Set `OPENROUTER_API_KEY` first.", kind="warn"))
    # the server needs the key in its own environment, because a client starts it with a small one
    async with open_session(
        "--model",
        str(model_file),
        "--data",
        str(data_file),
        "--llm",
        "openrouter/openrouter/free",
        env={"PATH": os.environ["PATH"], "OPENROUTER_API_KEY": api_key},
    ) as llm_session:
        llm_result = await llm_session.call_tool("describe_feature", {"feature": feature_picker.value})
    mo.callout(llm_result.content[0].text, kind="danger" if llm_result.isError else "neutral")


if __name__ == "__main__":
    app.run()
