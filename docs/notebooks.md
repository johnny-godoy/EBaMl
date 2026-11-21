# All notebooks

The notebooks in `notebooks/` are [marimo](https://marimo.io) notebooks. Each one shows one feature working, step by
step. Each page below shows a notebook with the output of its cells, exported when the docs are built.

The steps that need no API key run first. Each step that sends a request to an LLM waits for a button press, so a
notebook never sends a request by accident. The default model needs `OPENROUTER_API_KEY`: see
[Setup](setup.md#3-set-the-api-key-of-your-llm-provider) and [Choosing the model](providers.md).

| Notebook | What it shows |
| --- | --- |
| [Titanic](notebook.md) | An EBM of the Titanic data: the description of a graph and of the whole model, and the graph and text API (`from_ebm`, `simplify`, `to_text`, `from_text`). |
| [Local explanations](local-notebook.md) | One passenger at a time: the facts with no API key, the confidence slider, a check of the guarantees against random completions, and the text for each audience. |
| [Custom audiences](audiences-notebook.md) | Six audiences for the local explanations: subclasses of `Layperson`, and an `Audience` with its own rules. |
| [Languages](languages-notebook.md) | The local explanations in several languages with `language=`. The facts do not depend on the language. |
| [Model providers](providers-notebook.md) | The same request to OpenRouter, OpenAI, Anthropic, Ollama and a gateway with an OpenAI-compatible API, and the errors for an unknown provider and a missing key. |
| [Multiclass](multiclass-notebook.md) | An EBM of the iris data with three classes: one series for each class, the centring that keeps the predictions, and the `classes` and `normalization` options. |
| [Model kinds](model-kinds-notebook.md) | A binary classifier, a multiclass classifier and a regressor side by side: the graph text and the fields of the local explanations. |
| [Model editing](editing-notebook.md) | `EditSession`: smooth a jump, flatten a dip, make a range monotone and neutralize a category, with the report of each edit. |
| [MCP server](mcp-notebook.md) | The real `ebaml-mcp` server and a client: read a graph, explain a row, edit the model and save it after a confirmation. |
| [BAML agent](baml-agent-notebook.md) | The BAML agent of the `baml` command, which answers one question with the tools of the server. |
| [Chat](chat-notebook.md) | A chat box in front of the `ebaml-chat` command. |

To open a notebook and run its cells, use `marimo edit` from a source checkout:

```bash
uv run marimo edit notebooks/local_explanations.py   # interactive
uv run marimo run notebooks/titanic.py               # as an app
```
