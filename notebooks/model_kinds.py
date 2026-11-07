"""A marimo notebook that compares the three kinds of EBM: binary classifier, multiclass classifier and regressor."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import attrs
    import marimo as mo
    import numpy as np
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from sklearn.datasets import load_iris

    from ebaml import EBMGraph, LocalOptions, OptionRangeError, compute_local_facts, describe_graph, model_kind

    return (
        EBMGraph,
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
        LocalOptions,
        OptionRangeError,
        attrs,
        compute_local_facts,
        describe_graph,
        load_iris,
        mo,
        model_kind,
        np,
        pl,
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
def _(ExplainableBoostingClassifier, ExplainableBoostingRegressor, load_iris, mo, model_kind, pl):
    iris = load_iris(as_frame=True)
    measures = pl.from_pandas(iris.data).rename(lambda name: name.removesuffix(" (cm)").replace(" ", "_"))
    species = pl.Series("species", iris.target_names[iris.target])
    features = measures.drop("petal_width")
    # no pairwise interactions: the graphs and the local explanations read one effect per feature
    models = {
        "binary": ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=42).fit(
            features,
            species == "virginica",
        ),
        "multiclass": ExplainableBoostingClassifier(
            feature_names=features.columns,
            interactions=0,
            random_state=42,
        ).fit(features, species),
        "regressor": ExplainableBoostingRegressor(feature_names=features.columns, interactions=0, random_state=42).fit(
            features,
            measures["petal_width"],
        ),
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
    first = features.head(5)
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
            bins = np.searchsorted(cuts, first[:, feature_index].to_numpy(), side="right")
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
    ## Local explanations of each kind

    The result types of the local explanations have field names that fit every kind. Pick a kind below to see the
    fields for that kind. The facts come from Python and send no request.

    - `Prediction.decision` is the predicted class. For a regressor, it is the answer to a question, such as "above 1".
    - `Prediction.output` is the probability of that class. For a regressor, it is the predicted value.
    - `Prediction.margin` is positive while the decision holds. It is the log-odds of the predicted class for a binary
      classifier, the smallest score gap to another class for a multiclass classifier, and the distance to the threshold
      for a regressor.
    - `Counterfactual.new_decision` is the class that is predicted after the changes. For a regressor, it is the other
      answer.
    - `SufficientReason.minimum` tells whether no smaller set of features fixes the decision. It is always true for a
      binary classifier. For the other kinds, it depends on the search that `LocalOptions.reason_search` names.

    A multiclass classifier has one margin for each other class: the score of the predicted class minus the score of
    that class. The option `target_class` asks for the changes that lead to a chosen class. Its set of changes is never
    marked as the smallest.

    The iris data has only 150 flowers, so the confidence intervals of the scores are wide. At a high confidence level,
    most flowers have no counterfactual and no sufficient reason. Flower 70 is a versicolor near the border with the
    virginica.
    """)


@app.cell
def _(features, mo, models):
    kind_choice = mo.ui.dropdown(list(models), value="binary", label="Model")
    flower = mo.ui.slider(0, len(features) - 1, value=70, label="Flower", show_value=True)
    confidence = mo.ui.slider(0.0, 0.99, step=0.01, value=0.0, label="Confidence level", show_value=True)
    target = mo.ui.dropdown(
        ["no target", *models["multiclass"].classes_],
        value="no target",
        label="Target class (multiclass only)",
    )
    mo.vstack([kind_choice, flower, confidence, target])
    return confidence, flower, kind_choice, target


@app.cell
def _(
    LocalOptions,
    OptionRangeError,
    compute_local_facts,
    confidence,
    features,
    flower,
    kind_choice,
    mo,
    models,
    target,
):
    instance = features.row(flower.value, named=True)
    chosen_target = target.value if kind_choice.value == "multiclass" and target.value != "no target" else None
    try:
        facts = compute_local_facts(
            models[kind_choice.value],
            instance,
            LocalOptions(confidence_level=confidence.value, target_class=chosen_target),
        )
    except OptionRangeError as option_error:
        mo.stop(predicate=True, output=mo.callout(mo.md(f"`OptionRangeError`: {option_error}"), kind="danger"))
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
                f" **{counterfactual.new_output:.2f}**. `minimum` is **{counterfactual.minimum}**:",
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
    ## A setup error and its fix

    A `target_class` needs a multiclass classifier, and it must be another class of the model. Each other call raises
    an `OptionRangeError` before any LLM call. The message names the cause. To fix it, pass a class of the model that
    is not the predicted class, or leave the option out.
    """)


@app.cell
def _(LocalOptions, OptionRangeError, compute_local_facts, instance, mo, models) -> None:
    attempts = {
        "binary classifier, target `virginica`": (models["binary"], "virginica"),
        "multiclass classifier, target `tulip`": (models["multiclass"], "tulip"),
    }
    errors = []
    for label, (attempted_model, target_class) in attempts.items():
        try:
            compute_local_facts(attempted_model, instance, LocalOptions(target_class=target_class))
            errors.append(mo.md(f"{label}: no error."))
        except OptionRangeError as option_error:
            errors.append(mo.callout(mo.md(f"{label}: `OptionRangeError`: {option_error}"), kind="danger"))
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
