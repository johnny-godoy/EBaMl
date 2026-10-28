# Multiclass notebook

A [marimo](https://marimo.io) notebook that reads an EBM of the iris data, which has three classes. It shows the kind of
the model, the graph with one series per class, the centring that keeps the predictions, the `classes` and
`normalization` options, and the errors of wrong options.

- Everything up to "Put it in words" runs without an API key and sends no request. The iris data comes with
  scikit-learn, so the notebook downloads nothing.
- The last cells describe a graph and the whole model with the LLM, and only run when you press their button. The
  default model needs `OPENROUTER_API_KEY`. Pass `model=` to use another model, see [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. The buttons wait for a
press, so they show their idle state here. To run them:

```bash
uv run marimo edit notebooks/multiclass.py
```

The [usage guide](usage.md) explains the options.

<p><a href="notebooks/multiclass.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/multiclass.html"
  title="The multiclass notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
