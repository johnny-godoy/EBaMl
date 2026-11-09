# BAML agent notebook

A [marimo](https://marimo.io) notebook that runs the BAML agent. The agent is a language model that explains an EBM
through the tools of the `ebaml-mcp` server. The notebook runs the agent from the `baml` command, shows the tools
that the model called, and shows the message of each setup problem that a user can fix.

- Everything up to the last button runs without an API key. The notebook fits a small EBM on made-up data, prints the
  signature of `explain_with_mcp`, and runs the offline BAML test of the agent loop on a scripted model.
- The last button sends real requests to the LLM, one for each step of the agent. The default model needs
  `OPENROUTER_API_KEY`. Type `PROVIDER/MODEL` in the notebook to use another model, see
  [Choosing the model](providers.md).
- The notebook needs the BAML toolchain (`baml`, version 0.23 or later) and the `mcp` extra:
  `pip install "ebaml[mcp]"`.

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/baml_agent.py
```

The [guide](mcp.md#baml-agent) explains the agent and its functions.

<p><a href="notebooks/baml_agent.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/baml_agent.html"
  title="The BAML agent notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
