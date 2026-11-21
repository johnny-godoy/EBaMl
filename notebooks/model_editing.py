"""A marimo notebook that edits the graphs of an EBM, with the report after each edit."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import narwhals as nw
    import numpy as np
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier

    from ebaml import (
        Align,
        Bins,
        Bounded,
        Categories,
        EBMGraph,
        EditSession,
        ExtendRange,
        FixedScore,
        Interpolate,
        MaxStep,
        Monotone,
        Monotonize,
        SelectionError,
        SetScores,
        Subgroup,
        ValueInterval,
        affected_profile,
        check_constraints,
        edit_effect,
        load_gamchanger,
        save_gamchanger,
    )

    return (
        Align,
        Bins,
        Bounded,
        Categories,
        EBMGraph,
        EditSession,
        ExplainableBoostingClassifier,
        ExtendRange,
        FixedScore,
        Interpolate,
        MaxStep,
        Monotone,
        Monotonize,
        Path,
        SelectionError,
        SetScores,
        Subgroup,
        ValueInterval,
        affected_profile,
        check_constraints,
        edit_effect,
        load_gamchanger,
        mo,
        np,
        nw,
        pl,
        save_gamchanger,
        tempfile,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Editing a model

    This notebook edits the graphs of an EBM with `EditSession`, as the study of GAM Changer (Wang et al., KDD 2022)
    edits a model of the risk of pneumonia: smooth a jump, flatten a dip, make a range monotone, and neutralize a
    category. It then checks business constraints, adds a bin where the model saw no data, measures the effect of the
    edits on the data and on a subgroup, and saves the edits as a file that GAM Changer opens.

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
    ## Business constraints

    A constraint states a business requirement. `check_constraints` reads the model and changes nothing. It gives one
    violation for each constraint that the model breaks, with the bins that break it and the edit that repairs it. Here
    the risk must rise with the age, the ward `unknown` must score 0, the risk must not jump by more than 0.3 between
    two neighbour ages, and no score of the age may leave the range from -3 to 3.
    """)


@app.cell
def _(Bounded, Categories, FixedScore, MaxStep, Monotone, check_constraints, mo, pl, session):
    constraints = [
        Monotone("age", "increasing"),
        FixedScore("ward", Categories(["unknown"]), 0.0),
        MaxStep("age", 0.3),
        Bounded("age", -3.0, 3.0),
    ]
    violations = check_constraints(session.model, constraints)
    violation_rows = pl.DataFrame(
        [{"violation": item.description, "repair": type(item.repair).__name__} for item in violations],
        schema={"violation": pl.String, "repair": pl.String},
    )
    mo.vstack([
        mo.md(f"The model breaks {len(violations)} of {len(constraints)} constraints."),
        mo.ui.table(violation_rows, selection=None),
    ])
    return constraints, violations


@app.cell
def _(check_constraints, constraints, mo, session, violations):
    for violation in violations:
        session.apply(violation.repair, f"Meet the business constraint: {violation.description}")
    remaining = check_constraints(session.model, constraints)
    mo.md(
        f"Applied the repair of each of the {len(violations)} violations. A repair of one constraint can break another"
        f" one, so check again: the model breaks {len(remaining)} of the constraints now.",
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Change the bins

    The model saw no patient above the age of 100, so the last bin covers every higher age. `ExtendRange` adds an empty
    bin above the oldest patient, with the score of the last bin, so no prediction changes. A later edit gives the new
    bin its own score: here the expert knows that the risk keeps growing.
    """)


@app.cell
def _(Bins, ExtendRange, SetScores, mo, session):
    oldest = float(session.model.feature_bounds_[0][1])
    session.apply(ExtendRange("age", high=120.0), "Ages above the data must have their own score")
    new_bin = len(session.model.bins_[0][0])
    last_score = float(session.model.term_scores_[0][new_bin])
    extension_edit = SetScores("age", Bins(new_bin, new_bin), last_score + 0.5)
    extension_report = session.preview(extension_edit)
    session.apply(extension_edit, "The risk keeps growing above the oldest patient")
    mo.md(f"The oldest patient was {oldest:.0f}. The new bin:\n\n```\n{extension_report.to_text()}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The effect on the predictions

    An edit changes the predictions of the model, so look at the effect before you use the model. The table shows the
    change of the predicted risk for a few patients. The next two tables check the data and the subgroups, as the
    Metric Panel and the Feature Panel of GAM Changer do.
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
def _(Subgroup, died, edit_effect, mo, nw, patients, pl, session):
    effect = edit_effect(session, nw.from_native(patients), died, Subgroup("ward", "intensive"))
    scope_rows = pl.DataFrame([
        {
            "scope": scope.scope,
            "rows": scope.row_count,
            "accuracy, original": None if scope.original is None else round(scope.original.accuracy, 3),
            "accuracy, before the last edit": None if scope.last is None else round(scope.last.accuracy, 3),
            "accuracy, current": None if scope.current is None else round(scope.current.accuracy, 3),
        }
        for scope in effect.scopes
    ])
    mo.vstack([
        mo.md(
            "A global metric hardly moves when a few bins change, so read the rows that the last edit changes and a"
            " subgroup too. The last edit here is the score of the new bin, which reaches no patient of the data.",
        ),
        mo.ui.table(scope_rows, selection=None),
    ])


@app.cell
def _(affected_profile, mo, nw, patients, pl, session):
    profile = affected_profile(session, nw.from_native(patients), revision=session.history_table()[0].id)
    profile_rows = pl.DataFrame([
        {"feature": feature.feature, "bins": ", ".join(feature.bins), "l1 distance": round(feature.distance, 3)}
        for feature in profile.features
    ])
    mo.vstack([
        mo.md(
            f"The first edit (the jump near 60) changes {profile.affected_rows} of {profile.total_rows} patients. For"
            " each other feature, the l1 distance compares its mix among these patients with all patients. A large"
            " distance means that the edit reaches one subgroup more than the others.",
        ),
        mo.ui.table(profile_rows, selection=None),
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


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Open the edits in GAM Changer

    `save_gamchanger` writes the original model and the edits as a file that the GAM Changer widget opens, so a person
    can check the edits there. Without data, the file holds synthetic rows made from the bins, because the widget needs
    rows to start. The result lists what the file cannot hold, and `load_gamchanger` reads the file back.
    """)


@app.cell
def _(Path, ebm, load_gamchanger, mo, pl, save_gamchanger, session, tempfile):
    export = save_gamchanger(session, Path(tempfile.mkdtemp()) / "pneumonia.gamchanger")
    loaded = load_gamchanger(export.path, ebm)
    commit_rows = pl.DataFrame([
        {
            "feature": commit.feature,
            "type": commit.commit_type,
            "edits": len(commit.edits),
            "description": commit.description,
        }
        for commit in loaded
    ])
    mo.vstack([
        mo.md(
            f"The file holds {export.commit_count} commits and {export.sample_count} synthetic rows. It leaves out:"
            f" {'; '.join(export.left_out) or 'nothing'}. Read back, the newest commit of each feature:",
        ),
        mo.ui.table(commit_rows, selection=None),
    ])


if __name__ == "__main__":
    app.run()
