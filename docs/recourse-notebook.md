---
title: Recourse notebook
description: A marimo notebook that turns the counterfactual of a loan decision into plans that a person can carry out.
---

# Recourse notebook

A [marimo](https://marimo.io) notebook that turns the counterfactual of a loan decision into recourse plans. It uses
one applicant of a synthetic loan model, and it shows each option of `RecourseOptions` on that applicant: the features
that must not change, the allowed values with the rules `overlap` and `inside`, one section for each cost strategy,
several diverse plans, the cap on the number of changes, and the errors of options that do not fit the model.

- Everything up to "Difficulties that an LLM proposes" runs without an API key, because Python computes every plan.
- The last section sends one request to the LLM, and only when you press its button. The default model needs
  `OPENROUTER_API_KEY`. Pass `model=` to use another model, see [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/recourse.py
```

The [guide](local-explanations.md#recourse-limits) explains the ideas, and the [proof page](recourse-proof.md) says
what is proved about the plans.

<p><a href="notebooks/recourse.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/recourse.html"
  title="The recourse notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
