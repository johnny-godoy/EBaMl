# Chat notebook

A [marimo](https://marimo.io) notebook that puts a chat box in front of the real `ebaml-chat` command. The agent is a
language model that reads an EBM through the tools of the `ebaml-mcp` server. Ask about the model or about one row of the
data, and ask follow-up questions. Each answer shows the tools that the agent called.

- The notebook fits a small EBM on made-up data and starts the chat with no key. The chat writes its first line without
  a request to an LLM.
- Each message in the chat box sends real requests to the LLM, one for each round of tool calls. The default model
  needs `OPENROUTER_API_KEY`. You can type a key in the notebook. Without a key, a message shows the error of a missing
  key, and the chat goes on.
- The notebook needs the BAML toolchain (`baml`, version 0.23 or later) and the `mcp` extra:
  `pip install "ebaml[mcp]"`.

The page below is the notebook with the output of its cells, exported when the docs are built. The exported page has no
live chat, because the chat box needs a running notebook. To run it:

```bash
uv run marimo edit notebooks/chat.py
```

The [guide](mcp.md#chat-with-the-agent) explains the command and its options.

<p><a href="notebooks/chat.html">Open the notebook in its own page</a></p>

<iframe
  src="notebooks/chat.html"
  title="The chat notebook, exported from marimo"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lighter);"
></iframe>
