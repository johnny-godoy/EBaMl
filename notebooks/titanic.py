"""A marimo notebook that talks to an EBM about the Titanic: the port of TalkToEBM's example notebook."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    from ebaml.descriptors import describe_ebm, describe_graph
    from ebaml.graphs import EBMGraph

    return (
        EBMGraph,
        ExplainableBoostingClassifier,
        describe_ebm,
        describe_graph,
        fetch_openml,
        mo,
        os,
        pl,
        train_test_split,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Talk to an EBM about the Titanic

    A port of TalkToEBM's *Spaceship Titanic* notebook. It uses the classic
    [Titanic dataset](https://www.openml.org/d/40945) from OpenML, which downloads without an account, instead of the
    Kaggle one.

    The LLM cells call [OpenRouter](https://openrouter.ai) through BAML and only run when you press their button. They
    need an OpenRouter API key: `OPENROUTER_API_KEY` from the environment that started the notebook (see the README), or
    the password field below when it is not set. The model is configured in
    `baml_src/talk_to_ebm.baml`.
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
    # a typed key only lives in this process; the BAML clients read the variable when they send a request
    if api_key_input is not None and api_key_input.value:
        os.environ["OPENROUTER_API_KEY"] = api_key_input.value
    has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    return (has_api_key,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Load the data and train an EBM
    """)


@app.cell
def _(fetch_openml, pl):
    titanic = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame)
    # `boat` and `body` only exist for the survivors and the dead, so they would leak the label; the rest of the
    # dropped columns are free text or have too many distinct values to fit in the context window of an LLM
    features = titanic.select("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked").with_columns(
        pl.col("sex", "embarked").cast(pl.String),
    )
    survived = titanic["survived"].cast(pl.String).cast(pl.Int8)
    return features, survived


@app.cell
def _(ExplainableBoostingClassifier, features, survived, train_test_split):
    x_train, x_test, y_train, y_test = train_test_split(features, survived, test_size=0.2, random_state=42)
    # no pairwise interactions: the LLM reads one graph per feature
    ebm = ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=42)
    ebm.fit(x_train, y_train)
    return ebm, x_test, y_test


@app.cell
def _(ebm, mo, x_test, y_test) -> None:
    mo.md(f"""
    Test accuracy: **{ebm.score(x_test, y_test):.3f}**. The features, in the order of `feature_index`:
    {", ".join(f"`{i}` {name}" for i, name in enumerate(ebm.feature_names_in_))}.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## High-level API: describe a graph

    `describe_graph` extracts the graph of one feature, turns it into text, and asks the LLM three questions in a row
    (a first description, the surprising regions, a final summary).
    """)


@app.cell
def _(ebm, mo):
    feature = mo.ui.dropdown(
        options={name: i for i, name in enumerate(ebm.feature_names_in_)},
        value="age",
        label="Feature",
    )
    with_context = mo.ui.checkbox(label="Tell the LLM about the dataset and the y-axis")
    describe_graph_button = mo.ui.run_button(label="Describe the graph")
    mo.hstack([feature, with_context, describe_graph_button], justify="start")
    return describe_graph_button, feature, with_context


@app.cell
def _():
    # context for the LLM: what the data is, and what the y-axis of the graphs means
    dataset_description = """The RMS Titanic sank in the North Atlantic in April 1912, after hitting an iceberg.
    Each row is a passenger, and the outcome is whether they survived.

    Features:
    - pclass - Ticket class: 1 (first), 2 (second) or 3 (third).
    - sex - The sex of the passenger.
    - age - The age in years. Missing for some passengers.
    - sibsp - The number of siblings and spouses aboard.
    - parch - The number of parents and children aboard.
    - fare - The price of the ticket, in 1912 pounds.
    - embarked - The port of embarkation: C (Cherbourg), Q (Queenstown) or S (Southampton)."""
    graph_description = "The y-axis of the graph shows contributions in log-odds towards survival."
    return dataset_description, graph_description


@app.cell
def _(
    dataset_description,
    describe_graph,
    describe_graph_button,
    ebm,
    feature,
    graph_description,
    has_api_key,
    mo,
    with_context,
) -> None:
    mo.stop(not describe_graph_button.value, mo.md("Press the button to ask the LLM."))
    mo.stop(not has_api_key, mo.callout("Enter your OpenRouter API key above first.", kind="warn"))
    context = {"dataset_description": dataset_description, "graph_description": graph_description}
    with mo.status.spinner("Asking the LLM (three turns)..."):
        description = describe_graph(ebm, feature.value, **(context if with_context.value else {}))
    mo.md(description)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## High-level API: describe the whole model

    `describe_ebm` describes every feature graph (concurrently), then asks the LLM to summarize the model, with the
    feature importances. The free models are rate limited, so this lowers the concurrency.
    """)


@app.cell
def _(mo):
    describe_ebm_button = mo.ui.run_button(label="Describe the model")
    describe_ebm_button
    return (describe_ebm_button,)


@app.cell
def _(
    dataset_description,
    describe_ebm,
    describe_ebm_button,
    ebm,
    graph_description,
    has_api_key,
    mo,
) -> None:
    mo.stop(not describe_ebm_button.value, mo.md("Press the button to ask the LLM."))
    mo.stop(not has_api_key, mo.callout("Enter your OpenRouter API key above first.", kind="warn"))
    with mo.status.spinner("Asking the LLM (one chain per feature, then a summary)..."):
        summary = describe_ebm(
            ebm,
            max_concurrency=2,
            dataset_description=dataset_description,
            graph_description=graph_description,
        )
    mo.md(summary)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Low-level API: graphs and text

    The prompts are built from `EBMGraph`s. This part makes no LLM calls.

    `EBMGraph.from_ebm` extracts the graph of a feature from the EBM: the bins of the x-axis, the scores and their
    standard deviations. `simplify` merges neighbouring bins whose scores differ by less than a fraction of the range
    of the graph, which makes the text shorter.
    """)


@app.cell
def _(EBMGraph, ebm, mo):
    age = ebm.feature_names_in_.index("age")
    graph = EBMGraph.from_ebm(ebm, age)
    simplified = graph.simplify(min_variation_per_cent=0.05)
    mo.md(
        f"The graph of `age` has **{len(graph.x_vals)}** bins, and **{len(simplified.x_vals)}** after simplifying it.",
    )
    return graph, simplified


@app.cell
def _(mo) -> None:
    mo.md(r"""
    `to_text` writes the graph as the LLM reads it. There is no tokenizer before the call, so the size is only
    estimated. `to_text` simplifies the graph until the estimate is within `overbudget_percent` (50% by default) of
    `max_tokens`, and warns if it is still above `max_tokens`. The provider's own error is the real limit.
    """)


@app.cell
def _(graph, mo) -> None:
    mo.md(f"```\n{graph.to_text(max_tokens=500)}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    `from_text` is the inverse: it reads a graph back from its text (or from a prompt that contains it). The standard
    deviations are estimated from the confidence bounds, so they are only as precise as the rounding of the text.
    """)


@app.cell
def _(EBMGraph, graph, mo) -> None:
    parsed = EBMGraph.from_text(graph.to_text())
    largest_error = max(abs(a - b) for a, b in zip(parsed.scores, graph.scores, strict=True))
    mo.md(
        f"Parsed back **{len(parsed.x_vals)}** bins of `{parsed.feature_name}`. "
        f"The largest difference of a score is **{largest_error:.4f}**.",
    )


if __name__ == "__main__":
    app.run()
