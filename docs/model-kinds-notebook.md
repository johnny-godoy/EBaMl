# Model kinds notebook

A [marimo](https://marimo.io) notebook that fits a binary classifier, a multiclass classifier and a regressor on the
same features of the iris data. It shows the kind that `model_kind` reads for each model, how the scores become the
output, the graph text of each kind, the fields of the local explanations, and the error for the kinds that the local
explanations do not accept.

- Everything up to "Put it in words" runs without an API key and sends no request. The iris data comes with
  scikit-learn, so the notebook downloads nothing.
- The last cells describe the petal length graph of each model with the LLM, and only run when you press their button.
  The default model needs `OPENROUTER_API_KEY`. Pass `model=` to use another model, see [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. The buttons wait for a
press, so they show their idle state here. To run them:

```bash
uv run marimo edit notebooks/model_kinds.py
```

The [usage guide](usage.md) explains the graph options, and the [local explanations guide](local-explanations.md)
explains the result types.

<p><a href="notebooks/model_kinds.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/model_kinds.html"
  title="The model kinds notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
