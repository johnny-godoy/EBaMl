"""A marimo notebook that explains several predictions of an EBM at the same time with the async API."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import os
    import time

    import marimo as mo
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.datasets import load_breast_cancer

    from ebaml import (
        DescribeEBMRequest,
        ExplainInstanceRequest,
        compute_local_facts,
        describe_ebm_async,
        explain_instance_async,
    )

    return (
        asyncio,
        compute_local_facts,
        describe_ebm_async,
        DescribeEBMRequest,
        explain_instance_async,
        ExplainableBoostingClassifier,
        ExplainInstanceRequest,
        load_breast_cancer,
        mo,
        os,
        pl,
        time,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Explain several predictions at the same time

    This notebook shows the async API of EBaMl. Each function that calls an LLM has an async form with the suffix
    `_async`: `describe_ebm_async` and `explain_instance_async` are two of them. An async call waits for the LLM
    without blocking the event loop, so several calls wait at the same time.

    You learn how to:

    1. describe the model with `describe_ebm_async`, and read the features whose description failed,
    2. explain several predictions at the same time in one `asyncio.TaskGroup`,
    3. compare the time of the calls one after the other with the time of the calls at the same time.

    The data is the breast cancer data of scikit-learn, so the notebook needs no download. The facts of each prediction
    need no API key. The cells that call the LLM run only when you press their button. The default model needs
    `OPENROUTER_API_KEY`: set it in the environment that starts the notebook, or type it in the password field below.
    To use another model, pass `model=` in the requests (see the documentation page "Choosing the model").

    marimo runs each cell that uses `await` in its event loop, so the cells below await the calls directly. In a
    script, run the coroutine with `asyncio.run`.
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
    ## Train an EBM

    Five features of the data keep the graphs and the explanations short. The EBM is a binary classifier without
    interactions, because the local explanations support that kind of model.
    """)


@app.cell
def _(ExplainableBoostingClassifier, load_breast_cancer, pl):
    cancer = load_breast_cancer(as_frame=True)
    feature_names = ["mean radius", "mean texture", "mean smoothness", "mean concavity", "mean symmetry"]
    features = pl.from_pandas(cancer.data[feature_names])
    ebm = ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=0)
    ebm.fit(features, cancer.target.to_numpy())
    return ebm, features


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The facts, without any LLM

    `compute_local_facts` computes the facts of each prediction from the model. It sends no request. Choose how many
    rows to explain. The table shows the predicted class and the number of changes of each counterfactual.
    """)


@app.cell
def _(mo):
    row_count = mo.ui.slider(2, 8, value=4, label="Rows to explain", show_value=True)
    row_count
    return (row_count,)


@app.cell
def _(compute_local_facts, ebm, features, mo, row_count):
    instances = features.head(row_count.value).to_dicts()
    row_facts = [compute_local_facts(ebm, instance) for instance in instances]
    mo.ui.table(
        [
            {
                "row": row,
                "class": facts.prediction.decision,
                "probability": round(facts.prediction.output, 3),
                "changes that flip it": len(facts.counterfactuals[0].changes) if facts.counterfactuals else None,
            }
            for row, facts in enumerate(row_facts)
        ],
        selection=None,
    )
    return (instances,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Describe the model with the async form

    `describe_ebm_async` describes the graphs as tasks of one task group, with at most `max_concurrency` requests in
    flight. If the request of one feature fails on its own, for example after a rate limit, the description covers the
    other features, and `failed_features` names the failed ones. An error that affects every request, such as a missing
    key, stops the call.
    """)


@app.cell
def _(mo):
    describe_button = mo.ui.run_button(label="Describe the model")
    describe_button
    return (describe_button,)


@app.cell
async def _(describe_button, describe_ebm_async, DescribeEBMRequest, ebm, has_api_key, mo):
    mo.stop(not describe_button.value, mo.md("Press the button to describe the model first."))
    mo.stop(not has_api_key, mo.callout("Enter your OpenRouter API key above first.", kind="warn"))
    with mo.status.spinner("Describing the model..."):
        description = await describe_ebm_async(DescribeEBMRequest(ebm, max_concurrency=2))
    failed_view = (
        mo.callout(
            "These features failed: " + ", ".join(failure.feature for failure in description.failed_features),
            kind="warn",
        )
        if description.failed_features
        else mo.callout("Every feature has a description.", kind="success")
    )
    mo.vstack([mo.md(description.text), failed_view])
    return (description,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Explain the rows at the same time

    One task for each row runs in one `asyncio.TaskGroup`. The tasks wait for the LLM at the same time, so the total
    time is close to the time of the slowest call. The second button runs the same calls one after the other, for a
    comparison. A provider limits the rate of the requests, so keep the number of rows small with a free model.
    """)


@app.cell
def _(mo):
    together_button = mo.ui.run_button(label="Explain the rows at the same time")
    one_by_one_button = mo.ui.run_button(label="Explain the rows one after the other")
    mo.hstack([together_button, one_by_one_button], justify="start")
    return one_by_one_button, together_button


@app.cell
async def _(
    asyncio,
    description,
    ebm,
    explain_instance_async,
    ExplainInstanceRequest,
    instances,
    mo,
    one_by_one_button,
    time,
    together_button,
):
    mo.stop(
        not (together_button.value or one_by_one_button.value),
        mo.md("Press a button to explain the rows."),
    )
    requests = [ExplainInstanceRequest(ebm, instance, description.text) for instance in instances]
    started = time.perf_counter()
    if together_button.value:
        with mo.status.progress_bar(total=len(requests), title="Explaining the rows") as bar:

            async def explain(request):
                explanation = await explain_instance_async(request)
                bar.update()
                return explanation

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(explain(request)) for request in requests]
        explanations = [task.result() for task in tasks]
    else:
        explanations = []
        for request in mo.status.progress_bar(requests, title="Explaining the rows"):
            explanations.append(await explain_instance_async(request))
    elapsed = time.perf_counter() - started
    mo.vstack([
        mo.md(f"**{len(explanations)}** explanations in **{elapsed:.1f}** seconds."),
        mo.ui.table(
            [{"row": row, "counterfactual": item.counterfactual} for row, item in enumerate(explanations)],
            selection=None,
        ),
    ])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## When a call fails

    A failed call raises an error whose message lists the steps that fix it. In a task group, the first error cancels
    the other tasks, and the group raises an `ExceptionGroup`. To keep the explanations that succeeded, catch the error
    in each task, as `describe_ebm_async` does for the features.

    The steps that need no LLM, such as the facts of a prediction, run in the event loop and block it while they run.
    They take less than a second for this model.
    """)


if __name__ == "__main__":
    app.run()
