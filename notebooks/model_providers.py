"""A marimo notebook that sends the same request to different LLM providers with the ``model`` argument."""

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

    from ebaml import InvalidOptionError, LLMRequestError, MissingAPIKeyError, ModelOptions, describe_graph

    return (
        ExplainableBoostingClassifier,
        InvalidOptionError,
        LLMRequestError,
        MissingAPIKeyError,
        ModelOptions,
        describe_graph,
        fetch_openml,
        mo,
        os,
        pl,
    )


@app.cell
def _(mo) -> None:
    mo.md(r"""
    # Choose the model

    Every function of EBaMl that calls an LLM takes a `model` argument. This notebook sends the same request to
    several providers, so that you can compare the setup of each one and the text that each one writes.

    - Everything outside the buttons runs without a key and sends no request.
    - A button sends real requests (three turns for one graph) to the provider of its section. Press it only for a
      provider that you have set up. A section shows the key that it needs, and whether the environment has it.
    - The key never goes into `ModelOptions`. The options hold the *name* of the environment variable, and EBaMl reads
      the variable when it sends a request. To set a key, start the notebook from a shell that has the variable.

    The documentation page "Choosing the model" lists all providers and options.
    """)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## A small EBM to talk about

    The Titanic data and one feature, `sex`, whose graph has two bins. A short graph keeps each request cheap.
    """)


@app.cell
def _(ExplainableBoostingClassifier, fetch_openml, pl):
    titanic = pl.from_pandas(fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame)
    features = titanic.select("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked").with_columns(
        pl.col("sex", "embarked").cast(pl.String),
    )
    survived = titanic["survived"].cast(pl.String).cast(pl.Int8)
    # no pairwise interactions: the LLM reads one graph per feature
    ebm = ExplainableBoostingClassifier(feature_names=features.columns, interactions=0, random_state=42)
    ebm.fit(features, survived)
    sex_index = features.columns.index("sex")
    return ebm, sex_index


@app.cell
def _(mo, os):
    def key_status(variable: str | None) -> mo.Html:
        """Show whether the environment has the key of a provider, without showing the key.

        Returns
        -------
        mo.Html
            A callout with the status.

        """
        if variable is None:
            return mo.callout("This provider needs no key.", kind="neutral")
        if os.environ.get(variable):
            return mo.callout(f"`{variable}` is set.", kind="success")
        return mo.callout(f"`{variable}` is not set. Start the notebook from a shell that has it.", kind="warn")

    return (key_status,)


@app.cell
def _(LLMRequestError, describe_graph, ebm, mo, sex_index):
    def ask(model) -> mo.Html:
        """Describe the `sex` graph with the model, or show the fix steps of a failed request.

        Returns
        -------
        mo.Html
            The description, or a callout with the error.

        """
        try:
            with mo.status.spinner("Asking the LLM (three turns)..."):
                description = describe_graph(ebm, sex_index, num_sentences=3, model=model)
        except LLMRequestError as error:
            # the message names the cause and lists the steps that fix it
            return mo.callout(mo.md(f"**The request failed.**\n\n{error}"), kind="danger")
        return mo.md(description)

    return (ask,)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## The shorthand and `ModelOptions`

    The shorthand `"provider/model"` selects a provider and a model. It splits at the first `/` only, so
    `"openrouter/openai/gpt-4o-mini"` is the provider `openrouter` with the model `openai/gpt-4o-mini`. `ModelOptions`
    sets more: the temperature, the output limit, the address of the server and extra headers.
    """)


@app.cell
def _(ModelOptions, mo) -> None:
    shorthand = ModelOptions.parse("openrouter/openai/gpt-4o-mini")
    tuned = ModelOptions(provider="anthropic", model="claude-haiku-4-5", temperature=0.2, max_output_tokens=8000)
    mo.md(f"```\n{shorthand!r}\n\n{tuned!r}\n```")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## OpenRouter

    One key reaches many models. The free router `openrouter/free` is the default of EBaMl, and it picks a new model for
    each request. A pinned model gives repeatable results.
    """)


@app.cell
def _(key_status, mo):
    openrouter_button = mo.ui.run_button(label="Describe with OpenRouter")
    mo.vstack([key_status("OPENROUTER_API_KEY"), openrouter_button])
    return (openrouter_button,)


@app.cell
def _(ask, mo, openrouter_button) -> None:
    mo.stop(not openrouter_button.value, mo.md("Press the button to send the request."))
    ask("openrouter/openai/gpt-4o-mini")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## OpenAI

    The same call, with the provider `openai`.
    """)


@app.cell
def _(key_status, mo):
    openai_button = mo.ui.run_button(label="Describe with OpenAI")
    mo.vstack([key_status("OPENAI_API_KEY"), openai_button])
    return (openai_button,)


@app.cell
def _(ask, mo, openai_button) -> None:
    mo.stop(not openai_button.value, mo.md("Press the button to send the request."))
    ask("openai/gpt-4o-mini")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Anthropic

    This section uses `ModelOptions` to set the temperature and the output limit. A reasoning model counts its hidden
    tokens in `max_output_tokens`, so keep the limit large.
    """)


@app.cell
def _(key_status, mo):
    anthropic_button = mo.ui.run_button(label="Describe with Anthropic")
    mo.vstack([key_status("ANTHROPIC_API_KEY"), anthropic_button])
    return (anthropic_button,)


@app.cell
def _(ModelOptions, anthropic_button, ask, mo) -> None:
    mo.stop(not anthropic_button.value, mo.md("Press the button to send the request."))
    ask(ModelOptions(provider="anthropic", model="claude-haiku-4-5", temperature=0.2, max_output_tokens=8000))


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Ollama, a local model

    The model runs on your machine and needs no key. Install [Ollama](https://ollama.com), then run
    `ollama pull llama3.1` and keep the server running. `base_url` defaults to `http://localhost:11434/v1`. If the
    server is not running, the section shows the error with its fix steps.
    """)


@app.cell
def _(key_status, mo):
    ollama_button = mo.ui.run_button(label="Describe with Ollama")
    mo.vstack([key_status(None), ollama_button])
    return (ollama_button,)


@app.cell
def _(ask, mo, ollama_button) -> None:
    mo.stop(not ollama_button.value, mo.md("Press the button to send the request."))
    ask("ollama/llama3.1")


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## A gateway with an OpenAI-compatible API

    The provider `openai-generic` talks to any server that follows the OpenAI API: a company gateway, vLLM, LM Studio.
    It needs `base_url`. `api_key_env` names the variable that holds the key of the gateway, and `headers` adds headers
    to each request. Fill the fields, then press the button.
    """)


@app.cell
def _(mo):
    gateway_url = mo.ui.text(value="https://llm.example.com/v1", label="base_url", full_width=True)
    gateway_model = mo.ui.text(value="my-model", label="model")
    gateway_key_env = mo.ui.text(value="MY_GATEWAY_KEY", label="api_key_env")
    gateway_button = mo.ui.run_button(label="Describe with the gateway")
    mo.vstack([gateway_url, mo.hstack([gateway_model, gateway_key_env], justify="start"), gateway_button])
    return gateway_button, gateway_key_env, gateway_model, gateway_url


@app.cell
def _(ModelOptions, gateway_key_env, gateway_model, gateway_url, key_status, mo):
    gateway = ModelOptions(
        provider="openai-generic",
        model=gateway_model.value,
        base_url=gateway_url.value,
        api_key_env=gateway_key_env.value or None,
        headers={"x-team": "ebm"},
    )
    mo.vstack([key_status(gateway.api_key_env), mo.md(f"```\n{gateway!r}\n```")])
    return (gateway,)


@app.cell
def _(ask, gateway, gateway_button, mo) -> None:
    mo.stop(not gateway_button.value, mo.md("Press the button to send the request."))
    ask(gateway)


@app.cell
def _(mo) -> None:
    mo.md(r"""
    ## Other providers

    `google`, `azure`, `vertex` and `bedrock` work the same way. Their keys and settings differ: the table in the
    documentation page "Choosing the model" lists them.

    ## When the setup is wrong

    EBaMl checks the provider before the first paid request, and it turns the failures that you can fix into errors
    with fix steps. The two cells below send no request.
    """)


@app.cell
def _(InvalidOptionError, describe_graph, ebm, mo, sex_index) -> None:
    try:
        describe_graph(ebm, sex_index, model="not-a-provider/some-model")
    except InvalidOptionError as error:
        mo.callout(mo.md(f"`InvalidOptionError`: {error}"), kind="danger")


@app.cell
def _(MissingAPIKeyError, ModelOptions, describe_graph, ebm, mo, sex_index) -> None:
    # the variable name is made up, so the environment cannot have it
    unset = ModelOptions(provider="openai", model="gpt-4o-mini", api_key_env="EBAML_NOTEBOOK_UNSET_KEY")
    try:
        describe_graph(ebm, sex_index, model=unset)
    except MissingAPIKeyError as error:
        mo.callout(mo.md(f"`MissingAPIKeyError`: {error}"), kind="danger")


if __name__ == "__main__":
    app.run()
