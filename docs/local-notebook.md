# Local explanations notebook

A [marimo](https://marimo.io) notebook that explains single predictions of the Titanic EBM. It is the best place to
learn the feature by using it: pick a passenger, move the confidence slider, see the three explanations change, check
the guarantees by brute force, and compare what the LLM reads for a layperson and a data scientist.

- Everything up to "Put it in words" runs without an API key: it computes and shows the facts.
- The last cells call the LLM (the global explanation, then the three local ones) and only run when you press their
  button. They need `OPENROUTER_API_KEY`.

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/local_explanations.py
```

New to the package? Start with the [Titanic notebook](notebook.md), which makes the global explanation this notebook
builds on. The [guide](local-explanations.md) explains the ideas.

<p><a href="../notebooks/local_explanations.html">Open the notebook in its own page</a></p>

<iframe
  src="../notebooks/local_explanations.html"
  title="The local explanations notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
