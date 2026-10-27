# Model providers notebook

A [marimo](https://marimo.io) notebook that sends the same request to several LLM providers: OpenRouter, OpenAI,
Anthropic, Ollama and a gateway with an OpenAI-compatible API. Each section shows the setup of one provider and the
text that its model writes, so you can compare them.

- Everything outside the buttons runs without a key and sends no request.
- Each button sends real requests to one provider. Press it only for a provider that you have set up. A section shows
  the key that it needs and whether the environment has it.
- The last cells show the errors for an unknown provider and a missing key.

The page below is the notebook with the output of its cells, exported when the docs are built. The buttons wait for a
press, so they show their idle state here. To run them:

```bash
uv run marimo edit notebooks/model_providers.py
```

[Choosing the model](providers.md) lists all providers and options. To see the model argument at work in a longer
example, see the [Titanic notebook](notebook.md).

<p><a href="notebooks/model_providers.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/model_providers.html"
  title="The model providers notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
