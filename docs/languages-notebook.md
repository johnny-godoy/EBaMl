# Languages notebook

A [marimo](https://marimo.io) notebook that gets the local explanations of the Titanic EBM in several languages. The
argument `language=` of `explain_instance` asks the LLM to write the explanations in the language of the reader. The
notebook shows that the facts do not depend on the language, how to name a language, one passenger in several
languages side by side, and a language together with an audience.

- Everything up to "Put it in words" runs without an API key. This includes the error for a missing key.
- The last cells call the LLM (the global explanation, then the local ones) and only run when you press their button.
  The default model needs `OPENROUTER_API_KEY`. Pass `model=` to use another model, see
  [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/languages.py
```

The [guide](local-explanations.md#language) explains the ideas. The [custom audiences notebook](audiences-notebook.md)
shows how to change the role of the reader.

<p><a href="notebooks/languages.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/languages.html"
  title="The languages notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
