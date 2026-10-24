"""A marimo notebook that explains single predictions of an EBM: counterfactual, sufficient reason, completions."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import attrs
    import marimo as mo
    import numpy as np
    import polars as pl
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    from ebaml.custom_types import Audience
    from ebaml.descriptors import describe_ebm, explain_instance
    from ebaml.local import LocalOptions, compute_local_facts
    from ebaml.local_text import facts_to_texts

    return (
        Audience,
        ExplainableBoostingClassifier,
        LocalOptions,
        attrs,
        compute_local_facts,
        describe_ebm,
        explain_instance,
        facts_to_texts,
        fetch_openml,
        mo,
        np,
        os,
        pl,
        train_test_split,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Explain one prediction of an EBM

    The global explanation of an Explainable Boosting Machine (EBM) says how the model works *in general*: see
    `titanic.py` for how to ask an LLM for it. This notebook is about the *local* question: **why did the model predict
    this for this passenger?**

    The answer has two parts, and the split is the point of the design:

    1. **The facts are computed, not generated.** Python reads the scores and the confidence intervals of the EBM and
       works out three things exactly. No LLM is involved, so they cannot be hallucinated and they are reproducible.
    2. **The LLM only puts the facts in words.** It gets the global explanation of the model, the facts, and an
       *audience*, and writes the explanation.

    The three kinds of fact are the explainability queries of
    [Barceló et al., *Model Interpretability through the Lens of Computational Complexity*](https://arxiv.org/abs/2010.12265):

    | Query | Question it answers | Explanation |
    | --- | --- | --- |
    | Minimum change required | Fewest inputs to change so that the prediction flips? | Counterfactual |
    | Minimum sufficient reason | Fewest inputs that settle it, whatever the others are? | Sufficient reason |
    | Count completions | With only some inputs known, what share of completions get this prediction? | Completions |

    The first two are exact for an EBM without interactions, because the model is a sum of one effect per feature. The
    third is estimated by sampling. See the documentation page "Local explanations" for the details and the limits.

    The cells that call the LLM only run when you press their button and need an OpenRouter API key:
    `OPENROUTER_API_KEY` from the environment that started the notebook (see the README), or the password field below
    when it is not set. Everything before them works without a key.
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

    The same Titanic model as in `titanic.py`. Local explanations need *every* input of the passenger, so the passengers
    to explain are the ones with no missing value. The EBM itself is trained on all the rows. It has to be a binary
    classifier without pairwise interactions (`interactions=0`).
    """)


@app.cell
def _(ExplainableBoostingClassifier, fetch_openml, pl, train_test_split):
    titanic = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame)
    features = titanic.select("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked").with_columns(
        pl.col("sex", "embarked").cast(pl.String),
    )
    survived = titanic["survived"].cast(pl.String).cast(pl.Int8)
    x_train, x_test, y_train, y_test = train_test_split(features, survived, test_size=0.2, random_state=42)
    ebm = ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=42)
    ebm.fit(x_train, y_train)
    complete = x_test.drop_nulls()
    return complete, ebm, x_test, y_test


@app.cell
def _(complete, ebm, mo, x_test, y_test) -> None:
    mo.md(f"""
    Test accuracy: **{ebm.score(x_test, y_test):.3f}**. **{len(complete)}** of the {len(x_test)} test passengers have
    no missing value and can be explained.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Pick a passenger

    Any passenger works. Passenger 0 is a good start; then look for one the model is unsure about, and compare.
    """)


@app.cell
def _(complete, mo):
    passenger = mo.ui.slider(0, len(complete) - 1, value=0, label="Passenger", show_value=True)
    passenger
    return (passenger,)


@app.cell
def _(complete, ebm, mo, passenger):
    instance = complete.row(passenger.value, named=True)
    survival = ebm.predict_proba(complete[passenger.value : passenger.value + 1])[0][1]
    mo.md(
        "Passenger "
        + ", ".join(f"**{name}** = {value}" for name, value in instance.items())
        + f".\n\nThe EBM gives a survival probability of **{survival:.2f}**.",
    )
    return (instance,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The facts, without any LLM

    `compute_local_facts` computes the prediction and the three explanations. Its options (a `LocalOptions`) are:

    - `confidence_level`: how sure a claim has to be. Every score is moved by that many standard deviations *against*
      the claim, so a counterfactual still flips the prediction, and a sufficient reason still holds, when the model's
      effects sit at the unfavourable edge of their confidence interval. `0` ignores the uncertainty.
    - `min_support`: the share of the training data a value needs to be offered as an alternative. It keeps freak
      values (a 500-year-old passenger) out of the counterfactual.

    Move the slider and watch the explanations get harder to prove: more changes are needed to flip the prediction,
    and more inputs are needed to settle it.
    """)


@app.cell
def _(mo):
    confidence = mo.ui.slider(0.0, 0.99, step=0.01, value=0.95, label="Confidence level", show_value=True)
    confidence
    return (confidence,)


@app.cell
def _(LocalOptions, compute_local_facts, confidence, ebm, instance):
    facts = compute_local_facts(ebm, instance, LocalOptions(confidence_level=confidence.value))
    return (facts,)


@app.cell
def _(attrs, facts, mo) -> None:
    prediction = facts.prediction
    mo.vstack([
        mo.md(f"""
    ### Prediction and contributions

    The model predicts class **{prediction.label}** with probability **{prediction.probability:.2f}**, and a
    {facts.confidence_level:.0%} confidence interval of **[{prediction.lower:.2f}, {prediction.upper:.2f}]**. The
    interval comes from adding the variances of the effects of the features, which assumes they are independent.

    The contributions are in log-odds: a positive score pushes towards class 1 (survived). Largest first:
    """),
        mo.ui.table([attrs.asdict(c) for c in facts.contributions], selection=None),
    ])


@app.cell
def _(attrs, facts, mo) -> None:
    counterfactual = facts.counterfactual
    if counterfactual is None:
        counterfactual_view = mo.md("""
        ### Counterfactual

        **No change flips the prediction** at this confidence level, even when every input is changed to its most
        favourable realistic value.
        """)
    else:
        counterfactual_view = mo.vstack([
            mo.md(f"""
            ### Counterfactual (minimum change required)

            **{len(counterfactual.changes)}** change(s) are enough to flip the prediction, and **no smaller set of
            changes does**. After them the other class has probability **{counterfactual.new_probability:.2f}**.
            """),
            mo.ui.table([attrs.asdict(c) for c in counterfactual.changes], selection=None),
        ])
    counterfactual_view


@app.cell
def _(facts, mo) -> None:
    sufficient = facts.sufficient_reason
    if sufficient is None:
        sufficient_view = mo.md("""
        ### Sufficient reason

        **There is none** at this confidence level: even with every input known, the model's uncertainty is too large to
        guarantee the prediction. Lower the confidence level to see one.
        """)
    else:
        sufficient_view = mo.md(f"""
        ### Sufficient reason (minimum sufficient reason)

        Once **{", ".join(sufficient.features)}** are known, the prediction is settled: it cannot change whatever the
        other inputs are, and **no smaller set of inputs** does this. The guaranteed margin, in log-odds, is
        **{sufficient.guaranteed_margin:.2f}**.
        """)
    sufficient_view


@app.cell
def _(attrs, facts, mo) -> None:
    mo.vstack([
        mo.md("""
    ### Completions (count completions)

    Reveal the inputs one by one, in the order of how much each guarantees, and count how many of the possible
    completions of the unknown inputs get the same prediction. The unknown inputs are sampled from the training data
    (10,000 samples, fixed seed), so these shares are estimates, not guarantees.
    """),
        mo.ui.table([attrs.asdict(s) for s in facts.completions], selection=None),
    ])


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Check the guarantees

    The first two explanations are claims about *every* possible input, so they can be tested. Take the sufficient
    reason, keep the passenger's values for those features, and fill all the other features with the values of
    randomly chosen other passengers. The prediction must not change, ever. (This uses `confidence_level=0` and
    `min_support=0`, so that the claim covers every value, even rare ones.)
    """)


@app.cell
def _(LocalOptions, compute_local_facts, complete, ebm, instance, mo, np):
    strict = compute_local_facts(ebm, instance, LocalOptions(confidence_level=0, min_support=0))
    original = strict.prediction.label
    if strict.sufficient_reason is None:
        check_view = mo.md("No sufficient reason to check.")
    else:
        check_rng = np.random.default_rng(0)
        check_pool = complete.to_dicts()
        check_same = 0
        check_trials = 500
        for _ in range(check_trials):
            check_other = check_pool[int(check_rng.integers(len(check_pool)))]
            check_mixed = {
                name: value if name in strict.sufficient_reason.features else check_other[name]
                for name, value in instance.items()
            }
            check_same += int(compute_local_facts(ebm, check_mixed).prediction.label == original)
        check_view = mo.md(
            f"With **{', '.join(strict.sufficient_reason.features)}** fixed, **{check_same}** of "
            f"**{check_trials}** random completions kept the prediction ({original}). It should be all of them.",
        )
    check_view


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## What the LLM reads

    `facts_to_texts` turns the facts into one JSON text per explanation. This is the *audience* at work, and it happens
    before the LLM sees anything:

    - A **layperson** (the default) knows no numbers. Scores, probabilities, log-odds and intervals are replaced by
      words (`strongly`, `very likely`), so the LLM has no model number to repeat. The prompt also forbids numbers and
      jargon.
    - A **data scientist** gets the scores, the probabilities, the confidence interval and the confidence level, and is
      told to quote them.
    """)


@app.cell
def _(Audience, facts, facts_to_texts, mo) -> None:
    audience_texts = {audience: facts_to_texts(facts, audience) for audience in Audience}
    mo.ui.tabs({
        kind: mo.hstack(
            [mo.md(f"**{audience.value}**\n```json\n{audience_texts[audience][kind]}\n```") for audience in Audience],
            align="start",
            justify="start",
        )
        for kind in ("prediction", "counterfactual", "sufficient_reason", "completions")
    })


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    `explain_instance` needs the **global explanation** of the model, computed once and prepended to every prompt, so
    the LLM knows what the features mean and which effects are surprising. Get it with `describe_ebm` (this is the slow
    step: one chain per feature, then a summary).
    """)


@app.cell
def _(mo):
    describe_button = mo.ui.run_button(label="Describe the model")
    describe_button
    return (describe_button,)


@app.cell
def _(describe_button, describe_ebm, ebm, has_api_key, mo):
    mo.stop(not describe_button.value, mo.md("Press the button to describe the model first."))
    mo.stop(not has_api_key, mo.callout("Enter your OpenRouter API key above first.", kind="warn"))
    dataset_description = (
        "The RMS Titanic sank in April 1912. Each row is a passenger; the outcome is whether they survived. "
        "pclass is the ticket class (1 first, 3 third), sibsp and parch count relatives aboard, "
        "fare is in 1912 pounds, embarked is the port (C, Q or S). "
        "The graphs show contributions in log-odds towards survival."
    )
    with mo.status.spinner("Describing the model..."):
        global_explanation = describe_ebm(ebm, max_concurrency=2, dataset_description=dataset_description)
    mo.md(global_explanation)
    return dataset_description, global_explanation


@app.cell
def _(Audience, mo):
    audience = mo.ui.dropdown(
        options={a.value: a for a in Audience},
        value=Audience.LAYPERSON.value,
        label="Audience",
    )
    explain_button = mo.ui.run_button(label="Explain this prediction")
    mo.hstack([audience, explain_button], justify="start")
    return audience, explain_button


@app.cell
def _(
    audience,
    confidence,
    dataset_description,
    ebm,
    explain_button,
    explain_instance,
    global_explanation,
    instance,
    mo,
):
    mo.stop(not explain_button.value, mo.md("Press the button to explain the prediction of the passenger above."))
    with mo.status.spinner("Asking the LLM (three explanations, concurrently)..."):
        explanation = explain_instance(
            ebm,
            instance,
            global_explanation,
            audience=audience.value,
            confidence_level=confidence.value,
            dataset_description=dataset_description,
        )
    mo.md(f"""
    ### What would have to change

    {explanation.counterfactual}

    ### Which inputs settle it

    {explanation.sufficient_reason}

    ### How certain it gets as inputs become known

    {explanation.completions}
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    Run it again with the other audience and compare. The facts are the same; the words are not. Because the facts are
    computed and not generated, `explanation.facts` holds the numbers behind the text, so any sentence can be checked.
    """)


if __name__ == "__main__":
    app.run()
