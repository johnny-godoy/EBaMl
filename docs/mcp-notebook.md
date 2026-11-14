# MCP server notebook

A [marimo](https://marimo.io) notebook that starts the real `ebaml-mcp` server and connects to it as a client does.
It is the best place to learn the server by using it: list the tools, read a graph with a token budget of your
choice, explain a row of the data, edit the model and save it after a confirmation, serve a model over HTTP, serve each
kind of model, and see the message of each problem that a user can fix.

- Everything up to the last button runs without an API key. The notebook fits a small EBM on made-up data, and the
  server calls no LLM.
- The last cell calls the LLM through `describe_feature` and only runs when you press its button. The default model
  needs `OPENROUTER_API_KEY`. Pass another `PROVIDER/MODEL` to `--llm` to use another model, see
  [Choosing the model](providers.md).
- The notebook needs the `mcp` extra: `pip install "ebaml[mcp]"`.

The page below is the notebook with the output of its cells, exported when the docs are built. To run it:

```bash
uv run marimo edit notebooks/mcp_server.py
```

The [guide](mcp.md) explains the tools and the options of the command.

<p><a href="notebooks/mcp_server.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/mcp_server.html"
  title="The MCP server notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
