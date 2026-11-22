---
title: Model editing notebook
description: A marimo notebook that edits the graphs of an EBM with EditSession and checks business constraints.
---

# Model editing notebook

A [marimo](https://marimo.io) notebook that edits the graphs of an EBM with `EditSession`. It follows the examples of
the study of GAM Changer: smooth a jump, flatten a dip, make a range monotone, and neutralize a category. The report of
each edit shows before the notebook applies the edit. The notebook then checks business constraints and applies their
repairs, adds a bin above the oldest patient, measures the effect on the data and on a subgroup, shows which subgroups
an edit reaches, and saves the edits as a file that GAM Changer opens. At the end, it edits one class of a model with
three classes and shows the confusion matrix of all classes.

- The data are synthetic, so the notebook needs no download.
- No cell sends a request to an LLM, and no cell needs a key.

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/model_editing.py
```

The [guide](editing.md) explains the edits, the selections, the constraints, the effect on data and the history.

<p><a href="notebooks/model_editing.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/model_editing.html"
  title="The model editing notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
