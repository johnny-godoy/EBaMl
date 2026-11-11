"""A marimo notebook that gets the local explanations of an EBM in several languages."""

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
        DescribeEBMRequest,
        ExplainInstanceRequest,
        Layperson,
        MissingAPIKeyError,
        ModelOptions,
        PromptOptions,
        compute_local_facts,
        describe_ebm,
        explain_instance,
        facts_to_texts,
    )

    return (
        compute_local_facts,
        describe_ebm,
        DescribeEBMRequest,
        explain_instance,
        ExplainableBoostingClassifier,
        ExplainInstanceRequest,
        facts_to_texts,
        fetch_openml,
        Layperson,
        MissingAPIKeyError,
        mo,
        ModelOptions,
        os,
        pl,
        PromptOptions,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Explanations in your language

    A local explanation says *why the model predicted this for this person*. The reader of that text may not read
    English. The argument `language=` of `explain_instance` asks the LLM to write the three explanations in the language
    of the reader. It is one string, and it works with every audience.

    What you learn here:

    1. The facts do not depend on the language. Python computes them, and the LLM only puts them in words.
    2. How to name a language, and how to use `language=` with the audiences of the package.
    3. The error that you meet when the key is missing.

    What you need: nothing up to the section "Put it in words". After that, you need an LLM. The default model needs
    `OPENROUTER_API_KEY`. Each call to the LLM sits behind a button, so you never pay by accident. To use another model,
    pass `model=`: see the documentation page "Choosing the model".
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

    The same Titanic EBM as in the other notebooks: a binary classifier without interactions. Pick a passenger. The
    facts below come from `compute_local_facts` and `facts_to_texts`. Neither function takes a language, because a
    number is the same in every language. The language only changes the words around the facts.
    """)


@app.cell
def _(ExplainableBoostingClassifier, fetch_openml, pl):
    # The liac-arff parser reads the categories as text. The pandas parser of scikit-learn 1.9 fails on pandas 3.1.
    titanic = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff").frame)
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
def _(Layperson, compute_local_facts, complete, ebm, facts_to_texts, mo, passenger):
    instance = complete.row(passenger.value, named=True)
    facts = compute_local_facts(ebm, instance)
    texts = facts_to_texts(facts, Layperson())
    mo.md(f"**The facts that the LLM reads** (for the layperson)\n\n```json\n{texts.counterfactual}\n```")
    return (instance,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## How to name a language

    Pass any name that the LLM knows. The package does not keep a list of languages, so a new language needs no new
    release. The table shows names that work well:

    | `language=` | Result |
    | --- | --- |
    | `"English"` | The default. |
    | `"French"`, `"German"`, `"Japanese"` | The name of the language in English. This is the clearest form. |
    | `"Deutsch"`, `"日本語"` | The name of the language in that language. |
    | `"Brazilian Portuguese"`, `"Simplified Chinese"` | A variety of a language, where the variety matters. |
    | `"formal German"` | A language with a register. |

    The package does not check the name. A misspelt name goes to the LLM as it is, and the LLM may guess the language or
    answer in English. If the text comes back in the wrong language, check the spelling first.

    The facts and the numbers stay as they are. The instruction asks the LLM to keep every number and to translate the
    names of the inputs only where the reader needs them.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Put it in words

    Now call the LLM. `explain_instance` needs the global explanation of the model, computed once with `describe_ebm`.
    That is the slow step: one chain per feature, then a summary. The global explanation is written in English here, and
    the local explanations are not: the argument `language=` sets the language of the local ones only. If you want the
    global explanation in another language too, write it in that language and pass it as it is.
    """)


@app.cell
def _(mo):
    describe_button = mo.ui.run_button(label="Describe the model")
    describe_button
    return (describe_button,)


@app.cell
def _(describe_button, describe_ebm, DescribeEBMRequest, ebm, has_api_key, mo, PromptOptions):
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
            DescribeEBMRequest(
                ebm,
                max_concurrency=2,
                prompt=PromptOptions(dataset_description=dataset_description),
            ),
        ).text
    mo.md(global_explanation)
    return dataset_description, global_explanation


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ### One passenger, several languages

    Choose the languages and press the button. Each language is one call of `explain_instance`, which sends three
    requests. Three languages send nine requests, so keep the list short. The tabs show the three explanations of each
    language, side by side with the same facts.
    """)


@app.cell
def _(mo):
    language_choice = mo.ui.multiselect(
        options=["English", "French", "German", "Spanish", "Japanese", "Brazilian Portuguese", "Simplified Chinese"],
        value=["French", "German", "Japanese"],
        label="Languages",
    )
    languages_button = mo.ui.run_button(label="Explain in these languages")
    mo.hstack([language_choice, languages_button], justify="start")
    return language_choice, languages_button


@app.cell
def _(
    dataset_description,
    ebm,
    explain_instance,
    ExplainInstanceRequest,
    global_explanation,
    instance,
    language_choice,
    languages_button,
    mo,
    PromptOptions,
):
    mo.stop(not languages_button.value, mo.md("Press the button to explain the passenger above."))
    mo.stop(not language_choice.value, mo.callout("Choose at least one language.", kind="warn"))
    explanations = {}
    for chosen_language in language_choice.value:
        with mo.status.spinner(f"Asking the LLM in {chosen_language}..."):
            explanations[chosen_language] = explain_instance(
                ExplainInstanceRequest(
                    ebm,
                    instance,
                    global_explanation,
                    language=chosen_language,
                    prompt=PromptOptions(dataset_description=dataset_description),
                ),
            )
    mo.ui.tabs({
        chosen_language: mo.md(f"""
        **What would have to change**

        {explanation.counterfactual}

        **Which inputs settle it**

        {explanation.sufficient_reason}

        **How certain it gets as inputs become known**

        {explanation.completions}
        """)
        for chosen_language, explanation in explanations.items()
    })


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ### The same language, another audience

    The language and the audience are two settings, and each one has its own job. The audience decides which facts the
    LLM gets and how it is told to write for the reader. The language decides the language of the words. So a data
    scientist who reads Spanish gets the scores of the data scientist, in Spanish. Type any language name below. It does
    not need to be in the list.
    """)


@app.cell
def _(mo):
    audience_choice = mo.ui.dropdown(options=["layperson", "data_scientist"], value="data_scientist", label="Audience")
    language_text = mo.ui.text(value="Spanish", label="Language")
    audience_button = mo.ui.run_button(label="Explain with this audience and language")
    mo.hstack([audience_choice, language_text, audience_button], justify="start")
    return audience_button, audience_choice, language_text


@app.cell
def _(
    audience_button,
    audience_choice,
    dataset_description,
    ebm,
    explain_instance,
    ExplainInstanceRequest,
    global_explanation,
    instance,
    language_text,
    mo,
    PromptOptions,
):
    mo.stop(not audience_button.value, mo.md("Press the button to explain the passenger with this pair."))
    mo.stop(not language_text.value.strip(), mo.callout("Type a language name.", kind="warn"))
    with mo.status.spinner("Asking the LLM..."):
        audience_explanation = explain_instance(
            ExplainInstanceRequest(
                ebm,
                instance,
                global_explanation,
                audience=audience_choice.value,
                language=language_text.value.strip(),
                prompt=PromptOptions(dataset_description=dataset_description),
            ),
        )
    mo.md(f"""
    **What would have to change**

    {audience_explanation.counterfactual}

    **Which inputs settle it**

    {audience_explanation.sufficient_reason}

    **How certain it gets as inputs become known**

    {audience_explanation.completions}
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## When the setup is wrong

    EBaMl checks the provider before the first paid request, and it turns the failures that you can fix into errors with
    fix steps. The cell below sends no request. It asks for an explanation in French from a model whose key is not in
    the environment.
    """)


@app.cell
def _(ebm, explain_instance, ExplainInstanceRequest, instance, MissingAPIKeyError, mo, ModelOptions) -> None:
    # the variable name is made up, so the environment cannot have it
    unset = ModelOptions(provider="openai", model="gpt-4o-mini", api_key_env="EBAML_NOTEBOOK_UNSET_KEY")
    try:
        explain_instance(
            ExplainInstanceRequest(ebm, instance, "The global explanation.", language="French", model=unset),
        )
    except MissingAPIKeyError as error:
        missing_key_view = mo.callout(mo.md(f"`MissingAPIKeyError`: {error}"), kind="danger")
    missing_key_view


@app.cell
def _(mo) -> None:
    mo.md(r"""
    Pass `language=` to `explain_instance` in your own code in the same way:

    ```python
    explain_instance(ExplainInstanceRequest(ebm, instance, global_explanation, audience="layperson", language="French"))
    ```

    To change the role of the reader as well as the language, write an audience. The audiences notebook shows six.
    """)


if __name__ == "__main__":
    app.run()
