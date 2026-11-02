"""A marimo notebook that writes custom audiences for the local explanations of an EBM."""

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

    from ebaml import (
        Audience,
        DataScientist,
        InvalidOptionError,
        Layperson,
        PromptOptions,
        compute_local_facts,
        describe_ebm,
        explain_instance,
        facts_to_texts,
        likelihood_word,
        points_towards,
        round_value,
        strength_word,
    )
    from ebaml.core.custom_types import JSONValue

    return (
        Audience,
        DataScientist,
        ExplainableBoostingClassifier,
        InvalidOptionError,
        JSONValue,
        Layperson,
        PromptOptions,
        compute_local_facts,
        describe_ebm,
        explain_instance,
        facts_to_texts,
        fetch_openml,
        mo,
        os,
        pl,
        likelihood_word,
        points_towards,
        round_value,
        strength_word,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Write your own audience

    A local explanation says *why the model predicted this for this person*. Who reads it changes what a good
    explanation is. A layperson needs words in place of log-odds. A data scientist wants the log-odds. A regulator wants
    the direction of every effect and no probability at all. EBaMl ships two audiences, `Layperson` and
    `DataScientist`, and the base class `Audience` lets you add your own.

    An audience does two jobs:

    1. **It decides which facts the LLM gets.** Seven methods (`contribution`, `prediction`, `change`, `flip`,
       `class_output`, `margin` and `share`) return the keys that the audience adds to one kind of fact. A number that
       the audience leaves out is a number that the LLM cannot repeat.
    2. **It tells the LLM how to write.** The method `guidance` returns the instruction, with its reason.

    All eight methods are abstract, so an author decides on every model number. There are two ways in. **Subclass a
    built-in audience** and override the one method that differs: this is the short way. **Subclass `Audience`** and
    write all eight methods: this is the way for an audience with new rules.

    This notebook builds six audiences for six readers (a child, a chief executive, a reader of Simplified Technical
    English, a loan officer, a machine learning engineer and a regulator). It shows the facts that each one produces,
    and the errors you meet when an audience is wrong. To change only the language of the explanation, you need no
    audience: pass `language=` (see the languages notebook).

    Everything works without a key until the last cells. Those cells call the LLM, and they only run when you press
    their button. The default model needs `OPENROUTER_API_KEY`. To use another model, pass `model=`:
    see the documentation page "Choosing the model".
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
    ## The model and the facts

    The same Titanic EBM as in `local_explanations.py`: a binary classifier without interactions. The facts of one
    passenger come from `compute_local_facts`. No LLM is involved, and the facts do not depend on the audience.
    """)


@app.cell
def _(ExplainableBoostingClassifier, fetch_openml, pl):
    titanic = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame)
    features = titanic.select("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked").with_columns(
        pl.col("sex", "embarked").cast(pl.String),
    )
    survived = titanic["survived"].cast(pl.String).cast(pl.Int8)
    ebm = ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=42)
    ebm.fit(features, survived)
    complete = features.drop_nulls()
    return complete, ebm


@app.cell
def _(complete, mo):
    passenger = mo.ui.slider(0, len(complete) - 1, value=0, label="Passenger", show_value=True)
    passenger
    return (passenger,)


@app.cell
def _(compute_local_facts, complete, ebm, passenger):
    instance = complete.row(passenger.value, named=True)
    facts = compute_local_facts(ebm, instance)
    return facts, instance


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 1. The short way: subclass a built-in audience

    Some readers need the facts of a layperson (words, no model numbers) and a different instruction. For each of them,
    the class below subclasses `Layperson` and overrides `guidance`. The other seven methods stay as they are. The
    readers differ in role and style:

    - `Eli5` is a child. The text uses short sentences and one everyday comparison.
    - `Executive` is a chief executive. The text opens with the bottom line and then gives the inputs that matter most.
    - `SimplifiedEnglish` is a reader of Simplified Technical English (ASD-STE100), for example a technician or a reader
      who learns English. The text uses one idea in each sentence and the same word for the same thing.

    Each instruction gives the reason ("because a child follows an image better than a rule") as well as the order. A
    model follows a rule better when it knows the motive.
    """)


@app.cell
def _(Layperson):
    from typing import override as _override

    class Eli5(Layperson):
        """A layperson who is a child: short sentences and one everyday comparison."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader is a child of about ten, so use short sentences, ordinary words and one comparison from "
                "everyday life, because a child follows an image better than a rule. Express strength and likelihood "
                "in plain words, and mention only values of the person's own inputs, because model numbers mean "
                "nothing to this reader."
            )

    class Executive(Layperson):
        """A layperson who decides: the bottom line first, then the inputs that matter."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader is a chief executive who has one minute, so open with the bottom line in one sentence. "
                "Then name the one or two inputs that matter most, because the reader decides on the result and not "
                "on the method. Use business words, and mention only values of the person's own inputs, because model "
                "numbers do not help this decision."
            )

    class SimplifiedEnglish(Layperson):
        """A layperson who reads Simplified Technical English (ASD-STE100)."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader needs text with one meaning, so write in Simplified Technical English (ASD-STE100): one "
                "idea in each sentence, at most 20 words in a sentence, the active voice, the simple present tense, "
                "and the same word for the same thing every time. Express strength and likelihood in plain words, and "
                "mention only values of the person's own inputs, because model numbers mean nothing to this reader."
            )

    return Eli5, Executive, SimplifiedEnglish


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 2. Change one rule: subclass `DataScientist`

    Two readers start from the data scientist and change a rule. A machine learning engineer wants one more number for
    each input. A loan officer reads figures, but only one: the probability, as a percentage. The scores, the standard
    deviations and the confidence interval are noise for this reader. The class `LoanOfficer` subclasses
    `DataScientist` and overrides three methods:

    - `contribution` returns no key, so a contribution shows the feature and its value only.
    - `prediction` returns the probability as a percentage, in place of the probability, the interval and the logit.
    - `guidance` asks for percentages and no statistics.

    The other methods keep the rules of the data scientist. The helper `round_value` rounds to the precision of the
    prompts.

    The machine learning engineer debugs the model. An effect that is small against its standard deviation can be noise
    from the training data. The class `MLEngineer` adds the key `signal_to_noise` (the size of the score divided by the
    standard deviation) to each contribution, and keeps every other rule of the data scientist.
    """)


@app.cell
def _(DataScientist, JSONValue, round_value):
    from typing import override as _override

    class LoanOfficer(DataScientist):
        """A reader who needs the chance of the outcome as a percentage, and nothing else about the model."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader is a loan officer who decides with percentages, so quote the percentage from the facts "
                "and skip the statistics of the model, because scores and intervals do not help the decision."
            )

        @_override
        def contribution(self, contribution, prediction) -> dict[str, JSONValue]:
            return {}

        @_override
        def prediction(self, prediction, confidence_level) -> dict[str, JSONValue]:
            return {"percent": round_value(100 * prediction.output)}

    class MLEngineer(DataScientist):
        """A reader who debugs the model and wants the signal-to-noise ratio of each effect."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader is a machine learning engineer who debugs the model, so quote the score and the "
                "signal_to_noise of each input. Point out an input with a low signal_to_noise, because such an effect "
                "may be noise from the training data."
            )

        @_override
        def contribution(self, contribution, prediction) -> dict[str, JSONValue]:
            ratio = abs(contribution.score) / contribution.std if contribution.std > 0 else None
            return {
                **super().contribution(contribution, prediction),
                "signal_to_noise": None if ratio is None else round_value(ratio),
            }

    return LoanOfficer, MLEngineer


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## 3. New rules: subclass `Audience`

    A regulator checks that the model uses the right inputs in the right direction. This reader wants the direction and
    the size of each effect, in words, and no probability at all, because a probability depends on the threshold of the
    bank. The class below subclasses `Audience` itself, so it writes all eight methods. Each one answers the question
    "which number does this reader get?". The answers are in the table:

    | Method | Keys of this audience |
    | --- | --- |
    | `contribution` | The direction and the size, in words, with the helpers `points_towards` and `strength_word`. |
    | `prediction` | None. |
    | `change` | The direction of the change of the score, in words. |
    | `flip` | Whether the flip holds under the uncertainty of the model. |
    | `class_output` | The likelihood of the class after the changes, in words. |
    | `margin` | None. |
    | `share` | A rough word, `most` or `few`, because the share is a sampling estimate. |
    """)


@app.cell
def _(Audience, JSONValue, likelihood_word, points_towards, strength_word):
    from typing import override as _override

    class Regulator(Audience):
        """A reader who checks the direction of the effects and wants no probability."""

        @_override
        def guidance(self) -> str:
            return (
                "The reader is a model regulator who checks the direction of every effect, so say for each input "
                "whether it pushes towards or against the prediction, and how strongly. State no probability and no "
                "number of the model, because the regulator judges the logic of the model and not its calibration."
            )

        @_override
        def contribution(self, contribution, prediction) -> dict[str, JSONValue]:
            direction = "towards" if points_towards(contribution, prediction) else "against"
            return {"direction": direction, "size": strength_word(contribution.score)}

        @_override
        def prediction(self, prediction, confidence_level) -> dict[str, JSONValue]:
            return {}

        @_override
        def change(self, change) -> dict[str, JSONValue]:
            return {"direction": "up" if change.score_change > 0 else "down"}

        @_override
        def flip(self, confidence_level) -> dict[str, JSONValue]:
            return {"flip_holds_even_if_the_model_is_somewhat_off": confidence_level > 0}

        @_override
        def class_output(self, output) -> dict[str, JSONValue]:
            return {"likelihood": likelihood_word(output)}

        @_override
        def margin(self, reason) -> dict[str, JSONValue]:
            return {}

        @_override
        def share(self, share: float) -> str:
            return "most" if share > 1 / 2 else "few"

    return (Regulator,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The facts that each audience produces

    `facts_to_texts` renders the same facts for each audience. Compare the tabs. `Eli5`, `Executive` and
    `SimplifiedEnglish` get the same facts as `Layperson`, because they only change `guidance`, so the tabs show
    `Layperson` for all four. The machine learning engineer gets `signal_to_noise` in addition to the data scientist's
    numbers. The loan officer gets `percent`. The regulator gets `direction` and `size`, and no probability.
    The LLM reads exactly these texts, so an audience that leaves a number
    out makes sure that the LLM never repeats it.
    """)


@app.cell
def _(
    DataScientist,
    Eli5,
    Executive,
    Layperson,
    LoanOfficer,
    MLEngineer,
    Regulator,
    SimplifiedEnglish,
    facts,
    facts_to_texts,
    mo,
) -> None:
    audience_objects = {
        "Layperson": Layperson(),
        "Eli5": Eli5(),
        "Executive": Executive(),
        "SimplifiedEnglish": SimplifiedEnglish(),
        "DataScientist": DataScientist(),
        "MLEngineer": MLEngineer(),
        "LoanOfficer": LoanOfficer(),
        "Regulator": Regulator(),
    }
    # the three subclasses of Layperson that change only the guidance give the facts of Layperson
    facts_audiences = {
        name: audience
        for name, audience in audience_objects.items()
        if name not in {"Eli5", "Executive", "SimplifiedEnglish"}
    }
    audience_texts = {name: facts_to_texts(facts, audience) for name, audience in facts_audiences.items()}
    mo.ui.tabs({
        kind: mo.hstack(
            [mo.md(f"**{name}**\n```json\n{getattr(texts, kind)}\n```") for name, texts in audience_texts.items()],
            align="start",
            justify="start",
        )
        for kind in ("prediction", "counterfactual", "sufficient_reason", "completions")
    })
    return (audience_objects,)


@app.cell
def _(audience_objects, mo) -> None:
    mo.md(
        "## The instruction of each audience\n\n"
        + "\n\n".join(f"**{name}**\n\n> {audience.guidance()}" for name, audience in audience_objects.items()),
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## When an audience is wrong

    Two mistakes are easy to make, and both fail early, before any LLM call and before you pay for one.

    **An unknown name.** A string selects a built-in audience only. A custom audience is passed as an object.
    """)


@app.cell
def _(InvalidOptionError, mo) -> None:
    from ebaml import resolve_audience

    try:
        resolve_audience("child")
    except InvalidOptionError as unknown_name_error:
        unknown_name_view = mo.callout(f"`InvalidOptionError`: {unknown_name_error}", kind="danger")
    unknown_name_view
    return (resolve_audience,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    **A missing method.** The eight methods are abstract. If you forget one, Python refuses to build the audience, and
    the error names the method. This is the reason for the design: an audience cannot leak a model number by omission.
    """)


@app.cell
def _(Audience, mo) -> None:
    from typing import override as _override

    class Forgetful(Audience):
        @_override
        def guidance(self) -> str:
            return "The reader is a manager."

    try:
        Forgetful()
    except TypeError as missing_method_error:
        missing_method_view = mo.callout(f"`TypeError`: {missing_method_error}", kind="danger")
    missing_method_view


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    Now call the LLM. `explain_instance` takes the audience as an object. It needs the global explanation of the model,
    computed once with `describe_ebm`. That is the slow step: one chain per feature, then a summary. Both buttons ask
    the LLM, and nothing runs before you press one.
    """)


@app.cell
def _(mo):
    describe_button = mo.ui.run_button(label="Describe the model")
    describe_button
    return (describe_button,)


@app.cell
def _(PromptOptions, describe_button, describe_ebm, ebm, has_api_key, mo):
    mo.stop(not describe_button.value, mo.md("Press the button to describe the model first."))
    mo.stop(not has_api_key, mo.callout("Enter your OpenRouter API key above first.", kind="warn"))
    dataset_description = (
        "The RMS Titanic sank in April 1912. Each row is a passenger; the outcome is whether they survived. "
        "pclass is the ticket class (1 first, 3 third), sibsp and parch count relatives aboard, "
        "fare is in 1912 pounds, embarked is the port (C, Q or S). "
        "The graphs show contributions in log-odds towards survival."
    )
    with mo.status.spinner("Describing the model..."):
        global_explanation = describe_ebm(
            ebm,
            max_concurrency=2,
            prompt=PromptOptions(dataset_description=dataset_description),
        )
    mo.md(global_explanation)
    return dataset_description, global_explanation


@app.cell
def _(audience_objects, mo):
    audience_choice = mo.ui.dropdown(options=list(audience_objects), value="Regulator", label="Audience")
    explain_button = mo.ui.run_button(label="Explain this prediction")
    mo.hstack([audience_choice, explain_button], justify="start")
    return audience_choice, explain_button


@app.cell
def _(
    PromptOptions,
    audience_choice,
    audience_objects,
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
            audience=audience_objects[audience_choice.value],
            prompt=PromptOptions(dataset_description=dataset_description),
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
    Run it again with another audience and compare. The facts are the same. The numbers that the LLM sees, and the
    instruction that it follows, are the ones of the audience. To add a reader of your own, write a subclass like the
    six above and pass an object of it as `audience=`. To get the explanation in another language, add `language=`.
    """)


if __name__ == "__main__":
    app.run()
