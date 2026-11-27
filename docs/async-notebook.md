---
title: Async notebook
description: A marimo notebook that describes an EBM and explains several predictions at the same time with the async API.
---

# Async notebook

A [marimo](https://marimo.io) notebook that uses the async form of the LLM calls. It describes a model with
`describe_ebm_async`, shows the features whose description failed, and explains several predictions at the same time in
one `asyncio.TaskGroup`. A second button runs the same calls one after the other, so you can compare the times.

- The model and the facts of each prediction need no API key. The data comes with scikit-learn, so the notebook needs
  no download.
- The cells that call the LLM run only when you press their button. The default model needs `OPENROUTER_API_KEY`. Pass
  `model=` in the requests to use another model, see [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/async_explanations.py
```

The [Async API](async.md) guide explains when to choose each form, the partial results, cancellation and timeouts.

<p><a href="notebooks/async_explanations.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/async_explanations.html"
  title="The async notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
