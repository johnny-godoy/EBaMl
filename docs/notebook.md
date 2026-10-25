# Notebook

A [marimo](https://marimo.io) notebook that trains an EBM on the Titanic data, asks the LLM to describe a graph and the
whole model, and walks through the graph and text API. It follows TalkToEBM's *Spaceship Titanic* notebook.
To explain a single prediction, see the [local explanations notebook](local-notebook.md).

The page below is the notebook with the output of its cells, exported when the docs are built. The cells that call the
LLM wait for a button press, so they show their idle state here. To run them, set `OPENROUTER_API_KEY` and open the
notebook:

```bash
uv run marimo edit notebooks/titanic.py
```

<p><a href="../notebooks/titanic.html">Open the notebook in its own page</a></p>

<iframe
  src="../notebooks/titanic.html"
  title="The Titanic notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
