"""A marimo notebook that shows the graph images and the text, vision and mixed modes of the graph descriptions."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import marimo as mo
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from sklearn.datasets import load_iris

    from ebaml import (
        DescribeGraphRequest,
        EBMGraph,
        GraphImageOptions,
        GraphOptions,
        InvalidOptionError,
        ModelOptions,
        VisionFallbackWarning,
        VisionUnsupportedError,
        check_token_budget,
        describe_graph,
        graph_png,
        model_kind,
    )

    return (
        check_token_budget,
        describe_graph,
        DescribeGraphRequest,
        EBMGraph,
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
        graph_png,
        GraphImageOptions,
        GraphOptions,
        InvalidOptionError,
        load_iris,
        mo,
        model_kind,
        ModelOptions,
        pl,
        VisionFallbackWarning,
        VisionUnsupportedError,
        warnings,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Graphs for vision models

    EBaMl sends each graph of an EBM to the LLM as text, as an image, or as both. This notebook shows the image of each
    model kind, the three modes of `describe_graph`, and what a call does when the model reads no images.

    - The images need the extra `vision`: `pip install "ebaml[vision]"`.
    - You need no data download: the iris data comes with scikit-learn.
    - Everything outside the buttons runs without a key and sends no request.
    - The buttons send real requests. The default model, `openrouter/free`, needs `OPENROUTER_API_KEY`, and it sends
      a request with an image to a free model that reads images.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Three EBMs

    The three models read the same features of the iris data: two measures, and the sepal width as a category
    (`narrow` or `wide`), so that you see a continuous and a categorical graph. Only the target changes: a binary
    classifier (is the flower a virginica?), a multiclass classifier (the species) and a regressor (the petal width).
    """)


@app.cell
def _(ExplainableBoostingClassifier, ExplainableBoostingRegressor, load_iris, model_kind, pl):
    iris = load_iris(as_frame=True)
    measures = pl.from_pandas(iris.data).rename(lambda name: name.removesuffix(" (cm)").replace(" ", "_"))
    species = pl.Series("species", iris.target_names[iris.target])
    narrow_below_cm = 3.0
    width_class = pl.when(measures["sepal_width"] < narrow_below_cm).then(pl.lit("narrow")).otherwise(pl.lit("wide"))
    features = measures.select("sepal_length", "petal_length", sepal_width_class=width_class)
    settings = {"feature_names": features.columns, "interactions": 0, "random_state": 42}
    models = {
        "binary": ExplainableBoostingClassifier(**settings).fit(features, species == "virginica"),
        "multiclass": ExplainableBoostingClassifier(**settings).fit(features, species),
        "regressor": ExplainableBoostingRegressor(**settings).fit(features, measures["petal_width"]),
    }
    petal_index = features.columns.index("petal_length")
    width_index = features.columns.index("sepal_width_class")
    axes = {name: model_kind(ebm).score_axis for name, ebm in models.items()}
    return axes, models, petal_index, width_index


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The image of each kind

    `graph_png` draws a graph as a PNG image of 640 by 360 pixels. A continuous feature is a step line with its 95%
    confidence band. A categorical feature is a bar for each value with error bars. A multiclass graph has one series
    for each class and a legend. `ModelKind.score_axis` names the score axis: log-odds for a classifier, and the units
    of the target for a regressor.
    """)


@app.cell
def _(EBMGraph, GraphImageOptions, axes, graph_png, mo, models, petal_index, width_index) -> None:
    mo.ui.tabs({
        name: mo.hstack(
            [
                mo.image(graph_png(EBMGraph.from_ebm(ebm, index), GraphImageOptions(score_axis=axes[name])))
                for index in (petal_index, width_index)
            ],
            wrap=True,
        )
        for name, ebm in models.items()
    })


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## One graph, two views

    Mixed mode sends the text and the image of the same graph. The text budget can merge bins first, so the image of
    mixed mode shows the simplified graph that the text renders: the two views agree bin for bin. `EBMGraph.to_text`
    gives that graph to `on_render`. The cell uses a small budget, so that the text merges bins.
    """)


@app.cell
def _(EBMGraph, GraphImageOptions, axes, check_token_budget, graph_png, mo, models, petal_index) -> None:
    extracted = EBMGraph.from_ebm(models["binary"], petal_index)
    shown = []
    text = extracted.to_text(
        max_tokens=600,
        overbudget_percent=0,
        check_budget=check_token_budget,
        on_render=shown.append,
    )
    mo.hstack(
        [
            mo.md(f"```\n{text}\n```"),
            mo.vstack([
                mo.md(f"The image of the text: {len(shown[0].x_vals)} bins, from {len(extracted.x_vals)}."),
                mo.image(graph_png(shown[0], GraphImageOptions(score_axis=axes["binary"]))),
            ]),
        ],
        widths=[1, 1],
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The three modes

    `GraphOptions(mode=...)` chooses what the requests send. Each button sends the same call, `describe_graph` of the
    petal length graph of the multiclass model, in one mode. Each call makes three requests.

    | Mode | The request sends |
    | --- | --- |
    | `text` | The graph text. |
    | `vision` | The image of the graph, with no text. The prompt asks the LLM to call each value an estimate. |
    | `mixed` | The text and the image. The prompt takes every number from the text and the shape from the image. |

    A mode that is not one of the three fails at once, before any request:
    """)


@app.cell
def _(GraphOptions, InvalidOptionError, mo) -> None:
    try:
        GraphOptions(mode="picture")
    except InvalidOptionError as mode_error:
        mo.output.replace(mo.callout(mo.md(f"`InvalidOptionError`: {mode_error}"), kind="warn"))


@app.cell
def _(mo):
    mode_buttons = {mode: mo.ui.run_button(label=f"Describe in {mode} mode") for mode in ("text", "vision", "mixed")}
    mo.hstack(list(mode_buttons.values()), justify="start")
    return (mode_buttons,)


@app.cell
def _(DescribeGraphRequest, GraphOptions, describe_graph, mo, mode_buttons, models, petal_index) -> None:
    pressed = [mode for mode, button in mode_buttons.items() if button.value]
    mo.stop(not pressed, mo.md("Press a button to describe the graph in that mode."))
    with mo.status.progress_bar(total=len(pressed), title="Asking the LLM (three turns each)") as bar:
        descriptions = {}
        for mode in pressed:
            request = DescribeGraphRequest(
                models["multiclass"],
                petal_index,
                num_sentences=4,
                graph=GraphOptions(mode=mode),
            )
            descriptions[mode] = describe_graph(request)
            bar.update()
    mo.vstack([mo.md(f"### {mode}\n\n{description}") for mode, description in descriptions.items()])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## A model that reads no images

    `ModelOptions(vision=False)` says that a model reads no images. `GraphOptions(vision_fallback=...)` chooses what a
    call in vision or mixed mode does then. With `"raise"`, the default, the call raises `VisionUnsupportedError` before
    any request. Its message lists the steps that fix the cause. The cell sends no request.
    """)


@app.cell
def _(
    DescribeGraphRequest,
    GraphOptions,
    ModelOptions,
    VisionUnsupportedError,
    describe_graph,
    mo,
    models,
    petal_index,
):
    text_only = ModelOptions(provider="openrouter", model="liquid/lfm-3-1.7b:free", vision=False)
    try:
        describe_graph(
            DescribeGraphRequest(models["binary"], petal_index, graph=GraphOptions(mode="vision"), model=text_only),
        )
    except VisionUnsupportedError as vision_error:
        mo.output.replace(mo.callout(mo.md(f"`VisionUnsupportedError`:\n\n```\n{vision_error}\n```"), kind="danger"))
    return (text_only,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    With `vision_fallback="text"`, the call describes the graph from its text and emits one `VisionFallbackWarning`.
    A provider that rejects the image gives the same two behaviors, and the rejected request is not retried. The button
    sends the call to the text-only model.
    """)


@app.cell
def _(mo):
    fallback_button = mo.ui.run_button(label="Describe with the text fallback")
    fallback_button
    return (fallback_button,)


@app.cell
def _(
    DescribeGraphRequest,
    GraphOptions,
    VisionFallbackWarning,
    describe_graph,
    fallback_button,
    mo,
    models,
    petal_index,
    text_only,
    warnings,
) -> None:
    mo.stop(not fallback_button.value, mo.md("Press the button to send the call with the text fallback."))
    fallback = GraphOptions(mode="vision", vision_fallback="text")
    with warnings.catch_warnings(record=True) as caught, mo.status.spinner("Asking the LLM (three turns)..."):
        warnings.simplefilter("always", VisionFallbackWarning)
        fallback_text = describe_graph(
            DescribeGraphRequest(models["binary"], petal_index, graph=fallback, model=text_only),
        )
    mo.vstack([
        mo.callout(mo.md("\n\n".join(f"`{item.category.__name__}`: {item.message}" for item in caught)), kind="warn"),
        mo.md(fallback_text),
    ])


if __name__ == "__main__":
    app.run()
