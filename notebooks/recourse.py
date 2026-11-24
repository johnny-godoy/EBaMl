"""A marimo notebook that turns the counterfactual of a loan decision into plans that a person can carry out."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import attrs
    import marimo as mo
    import numpy as np
    import orjson
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier

    from ebaml import (
        DataCost,
        Levels,
        LocalOptions,
        ProposeDifficultiesRequest,
        RecourseOptionError,
        RecourseOptions,
        ScaledCost,
        UnitCost,
        UserCost,
        ValueRange,
        compute_local_facts,
        facts_to_texts,
        propose_difficulties,
    )

    return (
        attrs,
        DataCost,
        ExplainableBoostingClassifier,
        Levels,
        LocalOptions,
        ProposeDifficultiesRequest,
        RecourseOptionError,
        RecourseOptions,
        ScaledCost,
        UnitCost,
        UserCost,
        ValueRange,
        compute_local_facts,
        facts_to_texts,
        mo,
        np,
        orjson,
        os,
        pl,
        propose_difficulties,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Recourse plans: changes that a person can carry out

    A counterfactual says what would have to change for a prediction to be different. For a person whose loan was
    denied, the fewest changes are not always a useful answer: the person cannot change their age, and an extra degree
    takes years. A **recourse plan** keeps to the limits of the person and prefers the changes that are easy for them.

    This notebook shows each option of `RecourseOptions` on one applicant of a synthetic loan model:

    1. The fewest changes, with no limit.
    2. The limits of the person: the features that must not change (`immutable`) and the values that a feature may take
       (`allowed`, with `ValueRange` and `Levels`).
    3. One section for each cost strategy, on the same applicant: `UnitCost`, `DataCost`, `UserCost` and `ScaledCost`.
    4. Several diverse plans (`plans`), and the cap on the number of changes (`max_changes`).
    5. The text that the LLM reads, and the errors of options that do not fit the model.
    6. The difficulties that an LLM proposes (`propose_difficulties`).

    Everything up to the last section runs without an API key, because Python computes every plan. The last section
    sends one request to the LLM, and only when you press its button. The default model needs `OPENROUTER_API_KEY`.
    See the documentation page "Local explanations", section "Recourse limits", for the ideas, and the page "Proof of
    the recourse plans" for what is proved.
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
    ## A synthetic loan model

    The data has six inputs. Income, savings and the number of past defaults are numbers. Housing and education are
    categories. Age is a number too. The bank approves a loan more often with a high income, high savings, no past
    default, an owned home and more education. The EBM has no interaction terms (`interactions=0`), because the local
    explanations need a sum of one effect per feature.
    """)


@app.cell
def _(ExplainableBoostingClassifier, np, pl):
    random_generator = np.random.default_rng(7)
    row_count = 3000
    applicants = pl.DataFrame({
        "age": random_generator.integers(20, 70, row_count).astype(float),
        "income": random_generator.integers(20, 120, row_count).astype(float),
        "savings": random_generator.integers(0, 50, row_count).astype(float),
        "past_defaults": random_generator.integers(0, 4, row_count).astype(float),
        "housing": random_generator.choice(["rent", "family", "own"], row_count),
        "education": random_generator.choice(["school", "bachelor", "master"], row_count),
    })
    log_odds = (
        0.04 * (applicants["income"] - 60).to_numpy()
        + 0.06 * (applicants["savings"] - 20).to_numpy()
        - 0.9 * applicants["past_defaults"].to_numpy()
        + 0.6 * (applicants["housing"] == "own").to_numpy()
        + 0.5 * (applicants["education"] == "master").to_numpy()
        + random_generator.normal(0, 0.8, row_count)
    )
    approved = (log_odds > 0).astype(int)
    ebm = ExplainableBoostingClassifier(feature_names=applicants.columns, interactions=0, random_state=7)
    ebm.fit(applicants, approved)
    return applicants, ebm


@app.cell
def _(applicants, ebm, mo, pl):
    # the first applicant whom the model denies and who has a past default, so that the limits matter
    denied = applicants.filter(pl.Series(ebm.predict(applicants) == 0)).filter(pl.col("past_defaults") >= 1)
    applicant = denied.row(0, named=True)
    mo.vstack([mo.md("## The applicant\n\nThe model denies this applicant:"), mo.ui.table([applicant], selection=None)])
    return (applicant,)


@app.cell
def _():
    def plan_rows(plans):
        """Write one row for each plan: its changes, its cost and the strength of its claim.

        Returns
        -------
        list[dict]
            The rows, in the order of the plans.

        """
        return [
            {
                "plan": number,
                "changes": "; ".join(
                    f"{change.feature}: {change.from_value} to {change.to_value}" for change in plan.changes
                ),
                "number of changes": len(plan.changes),
                "cost": round(plan.cost, 3),
                "minimality": str(plan.minimality),
            }
            for number, plan in enumerate(plans, start=1)
        ]

    return (plan_rows,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 1. The fewest changes

    With no recourse option, the counterfactual is the smallest set of changes after which the model approves. Sorting
    finds it exactly (`minimality` is `sorting_proof`). It may change the age or the past defaults, which the applicant
    cannot change.
    """)


@app.cell
def _(LocalOptions, applicant, compute_local_facts, ebm, mo, plan_rows):
    free_facts = compute_local_facts(ebm, applicant, LocalOptions(confidence_level=0.0))
    mo.ui.table(plan_rows(free_facts.counterfactuals), selection=None)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 2. The limits of the person

    The applicant cannot change their age or their past, so both are `immutable`. They can raise their income by at
    most 20, and they can move to their family's home but cannot buy one. A `ValueRange` gives the values of a number,
    and `Levels` gives the categories. A range and a bin can overlap in part: `match="overlap"` (the default) allows
    each bin with some values in the range, and `match="inside"` only each bin with all of its values in the range.
    """)


@app.cell
def _(Levels, RecourseOptions, ValueRange, applicant):
    def limits(match):
        """Build the limits of the applicant, with one rule for the range of the income.

        Returns
        -------
        RecourseOptions
            The limits.

        """
        return RecourseOptions(
            immutable=["age", "past_defaults"],
            allowed={
                "income": ValueRange(high=applicant["income"] + 20, match=match),
                "housing": Levels(["rent", "family"]),
            },
        )

    return (limits,)


@app.cell
def _(LocalOptions, applicant, compute_local_facts, ebm, limits, mo, plan_rows):
    limit_rows = []
    for match_rule in ("overlap", "inside"):
        limited_facts = compute_local_facts(
            ebm,
            applicant,
            LocalOptions(confidence_level=0.0, recourse=limits(match_rule)),
        )
        limit_rows.extend({"match": match_rule} | row for row in plan_rows(limited_facts.counterfactuals))
    mo.ui.table(limit_rows, selection=None)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 3. One section for each cost strategy

    Each strategy gives a cost to each change, and the counterfactual becomes the cheapest plan under the limits above.
    With equal costs, sorting still finds the plan exactly. With other costs, the MIP solver of `LocalOptions.search`
    finds it and proves it (`solver_proof`), because sorting is not exact there.

    - `UnitCost()` gives every change the cost 1: the fewest changes.
    - `DataCost()` gives a change its distance in the training data: a large jump of a number, or a rare category,
      costs more.
    - `UserCost(...)` takes the costs from you: here savings are cheap and education is expensive.
    - `ScaledCost(base, multipliers)` multiplies the costs of another strategy: here the data costs, with income twice
      as hard.
    """)


@app.cell
def _(DataCost, ScaledCost, UnitCost, UserCost):
    strategies = {
        "UnitCost": UnitCost(),
        "DataCost": DataCost(),
        "UserCost": UserCost(per_feature={"savings": 0.5, "education": 5.0}),
        "ScaledCost": ScaledCost(DataCost(), {"income": 2.0}),
    }
    return (strategies,)


@app.cell
def _(LocalOptions, applicant, attrs, compute_local_facts, ebm, limits, mo, plan_rows, strategies):
    strategy_rows = []
    for strategy_name, strategy in strategies.items():
        strategy_options = LocalOptions(confidence_level=0.0, recourse=attrs.evolve(limits("overlap"), costs=strategy))
        strategy_facts = compute_local_facts(ebm, applicant, strategy_options)
        strategy_rows.extend({"strategy": strategy_name} | row for row in plan_rows(strategy_facts.counterfactuals))
    mo.ui.table(strategy_rows, selection=None)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 4. Several plans and a cap

    `plans=3` asks for three plans. As in GAM Coach, each plan after the first uses no pair of a feature and a bin of an
    earlier plan, so it can move a feature of an earlier plan to another value. The costs of the plans do not go down.
    `max_changes=2` keeps every plan to at most two changes. A shorter list means that no further plan exists.
    """)


@app.cell
def _(DataCost, LocalOptions, applicant, attrs, compute_local_facts, ebm, limits, mo, plan_rows):
    diverse_recourse = attrs.evolve(limits("overlap"), costs=DataCost(), plans=3, max_changes=2)
    diverse_facts = compute_local_facts(ebm, applicant, LocalOptions(confidence_level=0.0, recourse=diverse_recourse))
    mo.ui.table(plan_rows(diverse_facts.counterfactuals), selection=None)
    return (diverse_facts,)


@app.cell
def _(diverse_facts, facts_to_texts, mo, orjson):
    counterfactual_text = orjson.dumps(
        orjson.loads(facts_to_texts(diverse_facts).counterfactual),
        option=orjson.OPT_INDENT_2,
    )
    mo.md(f"""
    ## 5. What the LLM reads

    The facts name the limits, the costs, the cap and the other plans. The prompt then calls a plan the cheapest one
    only when `is_optimal` is true, and it never calls a plan with costs the fewest changes.

    ```json
    {counterfactual_text.decode()}
    ```
    """)


@app.cell
def _(Levels, LocalOptions, RecourseOptionError, RecourseOptions, ValueRange, applicant, compute_local_facts, ebm, mo):
    error_cases = {
        "a feature that the model does not have": RecourseOptions(immutable=["height"]),
        "a value range on a category": RecourseOptions(allowed={"housing": ValueRange(high=1.0)}),
        "levels on a number": RecourseOptions(allowed={"income": Levels(["high"])}),
    }
    error_rows = []
    for case_name, case_options in error_cases.items():
        try:
            compute_local_facts(ebm, applicant, LocalOptions(recourse=case_options))
        except RecourseOptionError as error:
            error_rows.append({"options": case_name, "error": str(error)})
    mo.vstack([
        mo.md(
            "### Options that do not fit the model\n\nEach one raises a `RecourseOptionError` that says what to fix:",
        ),
        mo.ui.table(error_rows, selection=None),
    ])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 6. Difficulties that an LLM proposes

    When the applicant cannot say which changes are easy, `propose_difficulties` asks the LLM to rate each input on a
    fixed scale, with a reason. It runs no search: read the proposal and correct it, then pass `proposal.costs` to
    `RecourseOptions`. The facts then say that a language model proposed the difficulties, so the explanation calls them
    guesses. Press the button to send one request.
    """)


@app.cell
def _(mo):
    propose_button = mo.ui.run_button(label="Ask the LLM for the difficulties")
    propose_button
    return (propose_button,)


@app.cell
def _(
    LocalOptions,
    ProposeDifficultiesRequest,
    applicant,
    attrs,
    compute_local_facts,
    ebm,
    has_api_key,
    limits,
    mo,
    plan_rows,
    propose_button,
    propose_difficulties,
):
    mo.stop(not propose_button.value, mo.md("Press the button to ask the LLM."))
    mo.stop(not has_api_key, mo.callout("Set `OPENROUTER_API_KEY` first.", kind="warn"))
    proposal = propose_difficulties(ProposeDifficultiesRequest(ebm, applicant))
    proposal_rows = [
        {
            "feature": item.feature,
            "difficulty": str(item.difficulty),
            "multiplier": item.multiplier,
            "reason": item.reason,
        }
        for item in proposal.difficulties
    ]
    guided_options = LocalOptions(confidence_level=0.0, recourse=attrs.evolve(limits("overlap"), costs=proposal.costs))
    guided_facts = compute_local_facts(ebm, applicant, guided_options)
    mo.vstack([
        mo.ui.table(proposal_rows, selection=None),
        mo.md("The cheapest plan under these difficulties:"),
        mo.ui.table(plan_rows(guided_facts.counterfactuals), selection=None),
    ])


if __name__ == "__main__":
    app.run()
