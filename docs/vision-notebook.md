---
title: Vision models notebook
description: A marimo notebook that draws the graph images of each model kind and compares the text, vision and mixed modes.
---

# Vision models notebook

A [marimo](https://marimo.io) notebook that shows the graphs of an EBM as images. It draws the image of each model kind,
shows that the text and the image of mixed mode agree bin for bin, and sends the same graph in the text, vision and
mixed modes. It ends with the two behaviors of a call to a model that reads no images.

- The images need the extra `vision`: `pip install "ebaml[vision]"`.
- Everything up to "The three modes" runs without an API key, and so does the error of a model that reads no images.
- The buttons send real requests. The default model needs `OPENROUTER_API_KEY`, and it sends a request with an image to
  a free model that reads images. Pass `model=` to use another model, see [Choosing the model](providers.md).

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/vision.py
```

The [usage guide](usage.md#text-vision-and-mixed-modes) explains the modes and the fallback.

<p><a href="notebooks/vision.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/vision.html"
  title="The vision models notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
