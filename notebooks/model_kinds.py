"""A marimo notebook that compares the three kinds of EBM: binary classifier, multiclass classifier and regressor."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import attrs
    import marimo as mo
    import numpy as np
    import pandas as pd
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from sklearn.datasets import load_iris

    from ebaml import EBMGraph, UnsupportedModelError, model_kind
    from ebaml.llm.descriptors import describe_graph
    from ebaml.local.facts import LocalOptions, compute_local_facts

    return (
        EBMGraph,
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
        LocalOptions,
        UnsupportedModelError,
        attrs,
        compute_local_facts,
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
    # Model kinds

    This notebook puts the three kinds of EBM side by side: a binary classifier, a multiclass classifier and a
    regressor. You learn how EBaMl reads the kind of a model, what a score means for each kind, how the graph text
    differs, and which fields the local explanations give.

    - You need no data download: the iris data comes with scikit-learn.
    - Everything outside the buttons runs without a key and sends no request.
    - The buttons at the end send real requests. The default model needs `OPENROUTER_API_KEY`. Pass `model=` to use
      another model, as the documentation page "Choosing the model" shows.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Three EBMs on the same features

    The three models read the same three features of the iris data. Only the target changes:

    | Model | Target | Kind |
    | --- | --- | --- |
    | `binary` | Is the flower a virginica? | Binary classifier |
    | `multiclass` | The species of the flower | Multiclass classifier |
    | `regressor` | The petal width, in centimetres | Regressor |

    `model_kind` reads the kind of each fitted EBM from its link function. Each kind gives the names of the outputs
    and the meaning of a score.
    """)


@app.cell
def _(ExplainableBoostingClassifier, ExplainableBoostingRegressor, load_iris, mo, model_kind, pd):
    iris = load_iris(as_frame=True)
    measures = iris.data.rename(columns=lambda name: name.removesuffix(" (cm)").replace(" ", "_"))
    species = pd.Series(iris.target_names[iris.target], name="species")
    features = measures.drop(columns="petal_width")
    # no pairwise interactions: the graphs and the local explanations read one effect per feature
    models = {
        "binary": ExplainableBoostingClassifier(interactions=0, random_state=42).fit(features, species == "virginica"),
        "multiclass": ExplainableBoostingClassifier(interactions=0, random_state=42).fit(features, species),
        "regressor": ExplainableBoostingRegressor(interactions=0, random_state=42).fit(features, measures.petal_width),
    }
    kinds = {name: model_kind(ebm) for name, ebm in models.items()}
    petal_index = list(features.columns).index("petal_length")
    mo.ui.table(
        [
            {"model": name, "kind": repr(kind), "outputs": ", ".join(kind.outputs), "score": kind.score_meaning}
            for name, kind in kinds.items()
        ],
        selection=None,
    )
    return features, kinds, models, petal_index


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## From scores to the output

    An EBM adds the intercept and one score per feature. `ModelKind.predict` converts that sum to the output of the
    model: the sigmoid for a binary classifier, softmax for a multiclass classifier, and no change for a regressor. The
    cell sums the scores of the graphs for the first five flowers and compares the result with the EBM.
    """)


@app.cell
def _(EBMGraph, features, kinds, mo, models, np):
    first = features.iloc[:5]
    # the output of each EBM, as its kind defines it: one column per output
    model_outputs = {
        "binary": models["binary"].predict_proba(first)[:, 1:],
        "multiclass": models["multiclass"].predict_proba(first),
        "regressor": models["regressor"].predict(first).reshape(-1, 1),
    }
    differences = []
    for model_name, model in models.items():
        total = np.asarray(model.intercept_, dtype=float) + np.zeros((len(first), len(kinds[model_name].outputs)))
        for feature_index in range(len(features.columns)):
            graph_scores = np.asarray(EBMGraph.from_ebm(model, feature_index, use_feature_bounds=False).scores)
            # the bin of each value, from the cut points of the EBM (one bin per interval)
            cuts = np.asarray(model.bins_[feature_index][0], dtype=float)
            bins = np.searchsorted(cuts, first.iloc[:, feature_index].to_numpy(), side="right")
            total += graph_scores.reshape(len(graph_scores), -1)[bins]
        difference = float(np.abs(kinds[model_name].predict(total) - model_outputs[model_name]).max())
        differences.append({"model": model_name, "largest difference": f"{difference:.1e}"})
    mo.ui.table(differences, selection=None)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The graph of each kind

    The tabs show the text of the petal length graph for each model. A binary classifier and a regressor have one
    series of means. A multiclass classifier has one series for each class, after a `Class:` line, and the text says
    how to read it. The scores of a binary classifier are log-odds, and the scores of the regressor are centimetres.
    """)


@app.cell
def _(EBMGraph, mo, models, petal_index) -> None:
    mo.ui.tabs({
        name: mo.md(
            f"```\n{EBMGraph.from_ebm(ebm, petal_index).to_text(y_axis_precision=2, confidence_bounds=False)}\n```",
        )
        for name, ebm in models.items()
    })


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Local explanations of the binary classifier

    The local explanations take a binary classifier. Their result types have field names that fit every kind:

    - `Prediction.decision` is the predicted class, and `Prediction.output` is its probability.
    - `Prediction.score` is the sum of the scores on the scale of the link: the log-odds of the second class.
    - `Prediction.margin` is the log-odds of the predicted class. It is positive while the decision holds, so it means
      the same thing for every decision. When the decision is `False`, the margin is the opposite of the score.
    - `Counterfactual.new_decision` is the class that the prediction changes to, and `new_output` is its probability.
    - `SufficientReason.minimum` tells whether no smaller set of features fixes the decision. It is always true for a
      binary classifier.

    The iris data has only 150 flowers, so the confidence intervals of the scores are wide. At a high confidence level,
    most flowers have no counterfactual and no sufficient reason. Flower 70 is a versicolor near the border with the
    virginica.
    """)


@app.cell
def _(features, mo):
    flower = mo.ui.slider(0, len(features) - 1, value=70, label="Flower", show_value=True)
    confidence = mo.ui.slider(0.0, 0.99, step=0.01, value=0.0, label="Confidence level", show_value=True)
    mo.vstack([flower, confidence])
    return confidence, flower


@app.cell
def _(LocalOptions, compute_local_facts, confidence, features, flower, models):
    instance = features.iloc[flower.value].to_dict()
    facts = compute_local_facts(models["binary"], instance, LocalOptions(confidence_level=confidence.value))
    return facts, instance


@app.cell
def _(attrs, facts, instance, mo) -> None:
    prediction = facts.prediction
    counterfactual = facts.counterfactual
    sufficient = facts.sufficient_reason
    sections = [
        mo.md(
            "The flower has "
            + ", ".join(f"**{name}** = {value}" for name, value in instance.items())
            + f". The decision is **{prediction.decision}** with output **{prediction.output:.2f}**. The score is"
            f" **{prediction.score:.2f}** and the margin is **{prediction.margin:.2f}**.",
        ),
        mo.ui.table([attrs.asdict(prediction)], selection=None),
    ]
    if counterfactual is None:
        sections.append(mo.md("**No counterfactual** at this confidence level."))
    else:
        sections.extend([
            mo.md(
                f"The counterfactual changes the decision to **{counterfactual.new_decision}**, with output"
                f" **{counterfactual.new_output:.2f}**:",
            ),
            mo.ui.table([attrs.asdict(change) for change in counterfactual.changes], selection=None),
        ])
    if sufficient is None:
        sections.append(mo.md("**No sufficient reason** at this confidence level."))
    else:
        sections.append(
            mo.md(
                f"The sufficient reason is **{', '.join(sufficient.features)}**, with a guaranteed margin of"
                f" **{sufficient.guaranteed_margin:.2f}**. `minimum` is **{sufficient.minimum}**.",
            ),
        )
    mo.vstack(sections)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The other kinds

    The local explanations of regressors and multiclass classifiers are not available yet. The documentation page
    "Local explanations" defines what they will mean. A call with such a model raises `UnsupportedModelError` before any
    LLM call, and the message names the kind of the model.
    """)


@app.cell
def _(UnsupportedModelError, compute_local_facts, instance, mo, models) -> None:
    errors = []
    for other_name in ("multiclass", "regressor"):
        try:
            compute_local_facts(models[other_name], instance)
            errors.append(mo.md(f"`{other_name}`: no error."))
        except UnsupportedModelError as unsupported:
            errors.append(mo.callout(mo.md(f"`{other_name}`: `UnsupportedModelError`: {unsupported}"), kind="danger"))
    mo.vstack(errors)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    Each button sends the same call, `describe_graph` of the petal length graph, for one model. The prompt tells the LLM
    what a score of that kind means. Each call makes three requests.
    """)


@app.cell
def _(mo, models):
    buttons = {name: mo.ui.run_button(label=f"Describe the {name} graph") for name in models}
    mo.hstack(list(buttons.values()), justify="start")
    return (buttons,)


@app.cell
def _(buttons, describe_graph, mo, models, petal_index) -> None:
    pressed = [name for name, button in buttons.items() if button.value]
    mo.stop(not pressed, mo.md("Press a button to describe the graph of that model."))
    with mo.status.spinner("Asking the LLM (three turns)..."):
        descriptions = {name: describe_graph(models[name], petal_index, num_sentences=4) for name in pressed}
    mo.vstack([mo.md(f"### {name}\n\n{description}") for name, description in descriptions.items()])


if __name__ == "__main__":
    app.run()
