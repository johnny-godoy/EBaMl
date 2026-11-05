# MCP server

`ebaml-mcp` gives an agent a fitted EBM through the [Model Context Protocol](https://modelcontextprotocol.io). The
agent can read the model, read each graph and explain one prediction. The read and explain tools call no LLM. They are
fast, free and exact, and they need no API key.

## Install

Install the `mcp` extra:

```bash
pip install "ebaml[mcp]"
```

## Start the server

!!! warning
    `--model` also loads a pickled or joblib file, and loading such a file runs code from the file. Load only a file
    that you trust, or use a skops file. The server loads only the path that you give at start. No tool can load or
    write a path.

Save a fitted EBM with `skops.io`, which runs no code when the server loads the file. The server trusts only the EBM
classes, so it rejects a skops file that holds any other type.

<!-- invisible-code-block: python
import os
import tempfile

from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
start_directory = os.getcwd()
os.chdir(tempfile.mkdtemp())
-->

```python
import skops.io

skops.io.dump(ebm, "model.skops")
```

<!-- invisible-code-block: python
assert os.path.exists("model.skops")
os.chdir(start_directory)
-->

Start the server. The same option loads a skops file or a joblib file, and the server reads the format from the file:

```bash
ebaml-mcp --model model.skops --data train.csv
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | The EBM in a skops file, or in a pickled or joblib file. This option is required. |
| `--data PATH` | A CSV file with the rows for the `row` argument of `explain_prediction`. |
| `--llm PROVIDER/MODEL` | Turn on the LLM tools. See [Choosing the model](providers.md). Without it, the server has no LLM tools. |
| `--transport` | `stdio` (the default) or `streamable-http`. |

## Connect a client

Claude Code:

```bash
claude mcp add ebaml -- ebaml-mcp --model /path/to/model.joblib --data /path/to/train.csv
```

Claude Desktop. Add this entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ebaml": {
      "command": "ebaml-mcp",
      "args": ["--model", "/path/to/model.joblib", "--data", "/path/to/train.csv"]
    }
  }
}
```

Any other client that starts a stdio server uses the same command and arguments. The MCP Inspector is one example:

```bash
npx @modelcontextprotocol/inspector ebaml-mcp --model /path/to/model.joblib
```

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `get_model_summary` | none | The kind, the outputs or classes, the intercept, the number of features, and the interaction terms that the server skips. |
| `list_features` | none | The name, the type and the importance of each feature, the most important first. |
| `get_graph` | `feature`, optional `normalization`, `x_axis_precision`, `y_axis_precision`, `max_tokens` | The graph text of `EBMGraph.to_text`. |
| `explain_prediction` | `instance` or `row`, optional `confidence_level` | The `LocalFacts` and their texts, as JSON. |
| `describe_feature` | `feature` | The words of `describe_graph`. It needs `--llm`. |
| `describe_model` | none | The words of `describe_ebm`. The server keeps the text. It needs `--llm`. |

The server also serves each graph as the resource `ebm://graphs/{feature}`.

`explain_prediction` works for binary classifiers without interactions. For other models, it returns an error with the
reason. A package error is a tool error with its message.

### Example session: read the model

An agent calls `get_model_summary`, then `list_features`, then `get_graph` with the name of the first feature:

```json
{"name": "get_graph", "arguments": {"feature": "age", "max_tokens": 1000}}
```

### Example session: explain a prediction

An agent gives the values of all features, or the number of a row of the data file:

```json
{"name": "explain_prediction", "arguments": {"row": 0, "confidence_level": 0.9}}
```

The result has `facts` (the prediction, the contributions, the counterfactual, the sufficient reason and the
completions) and `texts` (the same facts as the JSON texts of the prompts).

### Example session: describe with an LLM

Start the server with `--llm openrouter/openai/gpt-4o-mini`. The agent then calls `describe_feature` or
`describe_model`. These calls send requests to the provider and can cost money.
