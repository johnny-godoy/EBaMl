# Custom audiences notebook

A [marimo](https://marimo.io) notebook that writes three audiences for the local explanations of the Titanic EBM. An
audience decides which facts the LLM gets and how the LLM is told to write. The notebook shows the short way (a
subclass of `Layperson`), a change of one rule (a subclass of `DataScientist`) and an audience with new rules (a
subclass of `Audience`). It prints the facts that each audience produces, side by side.

- Everything up to "Put it in words" runs without an API key. This includes the errors that a wrong audience raises: an
  unknown name, and a missing method.
- The last cells call the LLM (the global explanation, then the three local ones) and only run when you press their
  button. The default model needs `OPENROUTER_API_KEY`. Pass `model=` to use another model, see
  [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/audiences.py
```

The [guide](local-explanations.md#your-own-audience) explains the ideas. The
[local explanations notebook](local-notebook.md) shows the other parts of the feature.

<p><a href="notebooks/audiences.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/audiences.html"
  title="The custom audiences notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
