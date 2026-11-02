"""A marimo notebook that shows the graphs and the descriptions of a multiclass EBM."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.datasets import load_iris

    from ebaml import (
        EBMGraph,
        GraphOptions,
        InvalidOptionError,
        Normalization,
        UnsupportedModelError,
        describe_ebm,
        describe_graph,
        model_kind,
    )

    return (
        EBMGraph,
        ExplainableBoostingClassifier,
        GraphOptions,
        InvalidOptionError,
        Normalization,
        UnsupportedModelError,
        describe_ebm,
        describe_graph,
        load_iris,
        mo,
        model_kind,
        np,
        pd,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Multiclass classifiers

    This notebook shows how EBaMl reads an EBM with three or more classes: the kind of the model, the graph with one
    series per class, the options that select and shift the classes, and the description by the LLM.

    - You need no data download: the iris data comes with scikit-learn.
    - Everything outside the buttons runs without a key and sends no request.
    - The buttons at the end send real requests. The default model needs `OPENROUTER_API_KEY`. Pass `model=` to use
      another model, as the documentation page "Choosing the model" shows.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## A multiclass EBM

    The iris data has three classes of flower and four features. `model_kind` reads the kind of the fitted EBM.
    """)


@app.cell
def _(ExplainableBoostingClassifier, load_iris, mo, model_kind, pd):
    iris = load_iris(as_frame=True)
    features = iris.data.rename(columns=lambda name: name.removesuffix(" (cm)").replace(" ", "_"))
    species = pd.Series(iris.target_names[iris.target], name="species")
    # no pairwise interactions: the LLM reads one graph per feature
    ebm = ExplainableBoostingClassifier(interactions=0, random_state=42).fit(features, species)
    kind = model_kind(ebm)
    petal_index = list(features.columns).index("petal_length")
    mo.md(f"`model_kind(ebm)` is `{kind!r}`, with the outputs `{kind.outputs}`.\n\n{kind.score_meaning}")
    return ebm, features, kind, petal_index


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## One series per class

    The graph of a feature has one score per bin and per class. Softmax gives the same probabilities when the same
    number is added to the scores of all the classes of a bin. So `EBMGraph.from_ebm` centres each bin: the mean of
    its scores over the classes is zero. The text has one series of means and bounds for each class, after a
    `Class:` line.
    """)


@app.cell
def _(EBMGraph, ebm, mo, np, petal_index):
    graph = EBMGraph.from_ebm(ebm, petal_index)
    largest_mean = float(np.abs(np.asarray(graph.scores).mean(axis=1)).max())
    text = graph.to_text(y_axis_precision=2, confidence_bounds=False)
    mo.vstack([
        mo.md(f"The largest mean of a bin over the classes is `{largest_mean:.1e}`."),
        mo.md(f"```\n{text}\n```"),
    ])
    return (graph,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The centring keeps the predictions

    The intercept plus the centred score of each feature gives the same probabilities as the EBM, because the centring
    only adds one number per bin to all the classes. The cell compares them for the first five flowers.
    """)


@app.cell
def _(EBMGraph, ebm, features, kind, mo, np):
    scores = np.tile(np.asarray(ebm.intercept_, dtype=float), (5, 1))
    for index in range(len(features.columns)):
        feature_graph = EBMGraph.from_ebm(ebm, index, use_feature_bounds=False)
        # the bin of each value, from the cut points of the EBM (one bin per interval)
        cuts = np.asarray(ebm.bins_[index][0], dtype=float)
        bins = np.searchsorted(cuts, features.iloc[:5, index].to_numpy(), side="right")
        scores += np.asarray(feature_graph.scores)[bins]
    difference = float(np.abs(kind.predict(scores) - ebm.predict_proba(features.iloc[:5])).max())
    mo.md(f"The largest difference from `ebm.predict_proba` is `{difference:.1e}`.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Select the classes

    The text of a model with many classes is long. `classes` keeps some classes, in the order that you give. The
    scores stay centred over all the classes of the model, so a bin of the selected classes need not have a mean of
    zero. `describe_graph` and `describe_ebm` take the same option as `GraphOptions(classes=[...])`.
    """)


@app.cell
def _(EBMGraph, ebm, mo, petal_index) -> None:
    two = EBMGraph.from_ebm(ebm, petal_index, classes=["virginica", "setosa"])
    mo.md(f"```\n{two.to_text(y_axis_precision=2, confidence_bounds=False, include_description=False)}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Shift the scores

    `Normalization.MIN_MAX` and `Normalization.Z_SCORE` shift all the scores of the graph by one number. The
    differences between the classes of a bin, which carry the meaning, do not change. Here the smallest score becomes
    zero.
    """)


@app.cell
def _(EBMGraph, Normalization, ebm, graph, mo, np, petal_index) -> None:
    shifted = EBMGraph.from_ebm(ebm, petal_index, Normalization.MIN_MAX)
    gaps = np.diff(np.asarray(graph.scores), axis=1)
    shifted_gaps = np.diff(np.asarray(shifted.scores), axis=1)
    mo.md(
        f"The smallest score is `{np.min(shifted.scores):.2f}`. The largest change of a gap between two classes is"
        f" `{float(np.abs(gaps - shifted_gaps).max()):.1e}`.",
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## When the options are wrong

    The cells below show the errors for an unknown class, for one string in place of a list, and for `classes` on a
    model with one output. They send no request.
    """)


@app.cell
def _(EBMGraph, InvalidOptionError, ebm, mo, petal_index) -> None:
    try:
        EBMGraph.from_ebm(ebm, petal_index, classes=["setosa", "rose"])
        unknown_class = mo.md("No error.")
    except InvalidOptionError as unknown:
        unknown_class = mo.callout(mo.md(f"`InvalidOptionError`: {unknown}"), kind="danger")
    unknown_class


@app.cell
def _(EBMGraph, InvalidOptionError, ebm, mo, petal_index) -> None:
    try:
        EBMGraph.from_ebm(ebm, petal_index, classes="setosa")
        string_classes = mo.md("No error.")
    except InvalidOptionError as one_string:
        string_classes = mo.callout(mo.md(f"`InvalidOptionError`: {one_string}"), kind="danger")
    string_classes


@app.cell
def _(EBMGraph, ExplainableBoostingClassifier, UnsupportedModelError, features, mo, petal_index, species) -> None:
    binary = ExplainableBoostingClassifier(interactions=0, random_state=42).fit(features, species == "setosa")
    try:
        EBMGraph.from_ebm(binary, petal_index, classes=["True"])
        binary_classes = mo.md("No error.")
    except UnsupportedModelError as one_output:
        binary_classes = mo.callout(mo.md(f"`UnsupportedModelError`: {one_output}"), kind="danger")
    binary_classes


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    Each button sends real requests. `describe_graph` makes three requests for one graph. `describe_ebm` makes three
    for each feature and two for the summary.
    """)


@app.cell
def _(mo):
    graph_button = mo.ui.run_button(label="Describe the petal length graph")
    model_button = mo.ui.run_button(label="Describe the whole model")
    mo.hstack([graph_button, model_button], justify="start")
    return graph_button, model_button


@app.cell
def _(GraphOptions, describe_graph, ebm, graph_button, mo, petal_index) -> None:
    mo.stop(not graph_button.value, mo.md("Press the first button to describe the graph."))
    with mo.status.spinner("Asking the LLM (three turns)..."):
        graph_description = describe_graph(
            ebm,
            petal_index,
            num_sentences=4,
            graph=GraphOptions(classes=["setosa", "virginica"]),
        )
    mo.md(graph_description)


@app.cell
def _(describe_ebm, ebm, mo, model_button) -> None:
    mo.stop(not model_button.value, mo.md("Press the second button to describe the model."))
    with mo.status.spinner("Asking the LLM (three turns per feature, then the summary)..."):
        model_description = describe_ebm(ebm, num_sentences=8)
    mo.md(model_description)


if __name__ == "__main__":
    app.run()
