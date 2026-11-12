"""A marimo notebook that edits the graphs of an EBM, with the report after each edit."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier

    from ebaml import (
        Align,
        Bins,
        Categories,
        EBMGraph,
        EditSession,
        Interpolate,
        Monotonize,
        SelectionError,
        SetScores,
        ValueInterval,
    )

    return (
        Align,
        Bins,
        Categories,
        EBMGraph,
        EditSession,
        ExplainableBoostingClassifier,
        Interpolate,
        Monotonize,
        SelectionError,
        SetScores,
        ValueInterval,
        mo,
        np,
        pl,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Editing a model

    This notebook edits the graphs of an EBM with `EditSession`, as the study of GAM Changer (Wang et al., KDD 2022)
    edits a model of the risk of pneumonia: smooth a jump, flatten a dip, make a range monotone, and neutralize a
    category.

    - You need no key and no download. The data are synthetic.
    - No cell sends a request to an LLM. The edits are plain computations.
    - Each edit shows its report first (`preview`), and then applies it with the reason that a person gives.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The data and the model

    The risk of death grows with the age. Two defects of the data make the graph of the age wrong: a jump near 60, and
    a dip for the oldest patients, who are few and were treated with special care. The model also has a category
    `unknown` of the feature `ward` with no real meaning.
    """)


@app.cell
def _(ExplainableBoostingClassifier, np, pl):
    random_generator = np.random.default_rng(7)
    patient_count = 3000
    age = random_generator.uniform(20, 100, patient_count).round()
    ward = random_generator.choice(["general", "intensive", "unknown"], patient_count, p=[0.55, 0.3, 0.15])
    jump_age = np.isin(age, np.arange(58, 66))  # a jump of the risk near the age of 60
    dip_age = age > np.quantile(age, 0.88)  # the oldest patients, who are few and had special care
    logit = 0.05 * (age - 60) + 0.8 * (ward == "intensive") - 0.5 + 1.0 * jump_age
    logit = logit - 2.0 * dip_age + 0.6 * (ward == "unknown")
    died = (random_generator.uniform(size=patient_count) < 1 / (1 + np.exp(-logit))).astype(int)
    patients = pl.DataFrame({"age": age, "ward": ward})
    ebm = ExplainableBoostingClassifier(
        feature_names=patients.columns,
        feature_types=["continuous", "nominal"],
        interactions=0,
        random_state=0,
    ).fit(patients, died)
    return died, ebm, patients


@app.cell
def _(EBMGraph, ebm, mo):
    age_graph = EBMGraph.from_ebm(ebm, 0).to_text(include_description=False, confidence_bounds=False)
    mo.md(f"The graph of the age before any edit:\n\n```\n{age_graph}\n```")


@app.cell
def _(EditSession, ebm, mo):
    session = EditSession(ebm)
    mo.md(
        "A session works on a copy of the model, so `ebm` does not change. The head of the session is "
        f"`{session.revision}`.",
    )
    return (session,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Smooth a jump

    The graph jumps up and down near the age of 60. `Interpolate` draws a straight line from the first to the last
    selected score, and it keeps the bins. `ValueInterval` selects every bin that overlaps the ages 55 to 70, and the
    report names the bins that it chose.
    """)


@app.cell
def _(Interpolate, ValueInterval, mo, session):
    jump_edit = Interpolate("age", ValueInterval(55, 70))
    jump_report = session.preview(jump_edit)
    mo.md(f"```\n{jump_report.to_text()}\n```")
    return (jump_edit,)


@app.cell
def _(jump_edit, mo, session):
    session.apply(jump_edit, "The jump near 60 has no cause in the data")
    mo.md(f"Applied. The head is now `{session.revision}`.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Flatten a dip

    The oldest patients are few, and the graph falls for them. `Align` with `"left"` copies the score of the bin
    before the selection to the selected bins, so the graph stays flat there. The report shows how little of the
    training weight these bins hold.
    """)


@app.cell
def _(Align, Bins, ebm, mo, session):
    last_bin = len(ebm.term_scores_[0]) - 3
    dip_edit = Align("age", Bins(last_bin - 5, last_bin), "left")
    mo.md(f"```\n{session.preview(dip_edit).to_text()}\n```")
    return (dip_edit,)


@app.cell
def _(dip_edit, mo, session):
    session.apply(dip_edit, "Too few patients over 90 to trust the fall of the graph")
    mo.md(f"Applied. The head is now `{session.revision}`.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Make a range monotone

    The risk must not fall when the age goes up. `Monotonize` fits a weighted isotonic regression on the selected
    bins. The selection can be a part of the graph, and the rest stays as it is. The edit keeps the weighted mean of
    the selected scores.
    """)


@app.cell
def _(Monotonize, mo, session):
    monotone_edit = Monotonize("age", "increasing")
    monotone_report = session.preview(monotone_edit)
    mo.md(
        f"The edit changes {monotone_report.bins_changed} bins and reaches {monotone_report.weight_share:.1%} of the "
        "training weight.",
    )
    return (monotone_edit,)


@app.cell
def _(monotone_edit, mo, session):
    session.apply(monotone_edit, "The risk of death must not fall with the age")
    mo.md(f"Applied. The head is now `{session.revision}`.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Neutralize a category

    The category `unknown` of the ward has no real meaning, so the model must not score it. `SetScores` sets its bin to
    0 and names the category. A category that does not exist raises a `SelectionError` with the categories that do.
    """)


@app.cell
def _(Categories, SelectionError, SetScores, mo, session):
    try:
        session.preview(SetScores("ward", Categories(["emergency"]), 0.0))
    except SelectionError as error:
        failure = str(error)
    mo.md(f"A wrong category fails with the fix steps:\n\n> {failure}")


@app.cell
def _(Categories, SetScores, mo, session):
    ward_edit = SetScores("ward", Categories(["unknown"]), 0.0)
    mo.md(f"```\n{session.preview(ward_edit).to_text()}\n```")
    return (ward_edit,)


@app.cell
def _(mo, session, ward_edit):
    session.apply(ward_edit, "The ward unknown is a gap in the records, and it must not move the risk")
    mo.md(f"Applied. The head is now `{session.revision}`.")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The effect on the predictions

    An edit changes the predictions of the model, so look at the effect before you use the model. The table shows the
    change of the predicted risk for a few patients. A check on the data and on each subgroup belongs here too.
    """)


@app.cell
def _(ebm, mo, np, patients, pl, session):
    edited = session.model
    comparison = pl.DataFrame({
        "age": patients["age"],
        "ward": patients["ward"],
        "risk before": ebm.predict_proba(patients)[:, 1].round(3),
        "risk after": edited.predict_proba(patients)[:, 1].round(3),
    }).with_columns((pl.col("risk after") - pl.col("risk before")).alias("change"))
    largest = comparison.sort(pl.col("change").abs(), descending=True).head(8)
    mo.vstack([
        mo.md(f"The mean absolute change of the risk is `{np.abs(comparison['change']).mean():.4f}`."),
        mo.ui.table(largest, selection=None),
    ])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The history

    Each edit is a revision with its reason, a time and a flag for the review. `model_at` reads any revision without
    a checkout. A save is a commit too: it needs a message and refuses edits that nobody reviewed.
    """)


@app.cell
def _(mo, pl, session):
    rows = pl.DataFrame([
        {
            "id": row.short_id,
            "edit": row.edit_type,
            "feature": row.feature,
            "reason": row.reason,
            "reviewed": row.reviewed,
        }
        for row in session.history_table()
    ])
    mo.ui.table(rows, selection=None)


if __name__ == "__main__":
    app.run()
