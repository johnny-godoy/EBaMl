# MCP server

`ebaml-mcp` gives an agent a fitted EBM through the [Model Context Protocol](https://modelcontextprotocol.io). The
agent can read the model, read each graph and explain one prediction. The read and explain tools call no LLM. They are
fast, free and exact, and they need no API key.

A worked example, with its output, is in the [MCP server notebook](mcp-notebook.md).

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
| `--host ADDRESS` | The address of the `streamable-http` transport. The default is `127.0.0.1`, which accepts this computer only. |
| `--port NUMBER` | The port of the `streamable-http` transport. The default is `8000`. |

## Connect a client

Claude Code:

```bash
claude mcp add ebaml -- ebaml-mcp --model /path/to/model.skops --data /path/to/train.csv
```

Claude Desktop. Add this entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ebaml": {
      "command": "ebaml-mcp",
      "args": ["--model", "/path/to/model.skops", "--data", "/path/to/train.csv"]
    }
  }
}
```

Any other client that starts a stdio server uses the same command and arguments. The MCP Inspector is one example:

```bash
npx @modelcontextprotocol/inspector ebaml-mcp --model /path/to/model.skops
```

### Serve over HTTP

Use the `streamable-http` transport when the client runs on another computer or in a container:

```bash
ebaml-mcp --model model.skops --data train.csv --transport streamable-http --port 9000
```

The client connects to `http://127.0.0.1:9000/mcp`. The server has no login. Keep the default host `127.0.0.1`
unless a firewall or a proxy with authentication protects the port. With `--host 0.0.0.0`, every computer that
reaches the port can read the model and the rows of the data file.

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

The examples below come from a real session over stdio. The model is an EBM with `interactions=0` that was fitted on
500 synthetic rows with the features `age`, `income` and `city`. The server started with
`--model model.skops --data train.csv`. A result that is long is shortened, and `...` marks the cut.

### Example session: read the model

1. The agent calls `get_model_summary` to learn the kind of the model.

    ```json
    {"name": "get_model_summary", "arguments": {}}
    ```

    ```json
    {
      "kind": "BinaryClassifier",
      "outputs": ["1"],
      "classes": [0, 1],
      "score_meaning": "A score is the change of the log-odds of the class 1.",
      "intercept": [0.611152714154934],
      "feature_count": 3,
      "skipped_terms": []
    }
    ```

2. The agent calls `list_features` to choose a graph. The most important feature is first.

    ```json
    {"name": "list_features", "arguments": {}}
    ```

    ```json
    [
      {"name": "income", "type": "continuous", "importance": 1.1480326430918357},
      {"name": "age", "type": "continuous", "importance": 1.1147650229232495},
      {"name": "city", "type": "nominal", "importance": 0.840511802142521}
    ]
    ```

3. The agent calls `get_graph` for one feature. The result is text, not JSON. The server merges the bins of a
   continuous graph until the estimated size is near `max_tokens`. Here the budget is 1000 tokens, and the bins
   `(18.5, 19.5)`, `(19.5, 20.5)` and `(20.5, 21.5)` became the bin `(18.5, 21.5)`.

    ```json
    {"name": "get_graph", "arguments": {"feature": "age", "max_tokens": 1000}}
    ```

    ```text
    This graph represents a continuous-valued feature. The keys are intervals that represent ranges where the function predicts the same value.
    Feature Name: age
    Feature Type: continuous
    Means: {"(18.0, 18.5)": -2.497, "(18.5, 21.5)": -2.372, "(21.5, 22.5)": -2.283, "(22.5, 24.5)": -1.952, ...}
    Lower Bounds (95%-Confidence Interval): {"(18.0, 18.5)": -3.507, "(18.5, 21.5)": -3.254, "(21.5, 22.5)": -3.096, ...}
    Upper Bounds (95%-Confidence Interval): {"(18.0, 18.5)": -1.488, "(18.5, 21.5)": -1.491, "(21.5, 22.5)": -1.47, ...}
    ```

    If the budget is too small for the graph, the tool returns an error. With `"max_tokens": 100`, the error says:
    `The graph for feature age of type continuous is estimated at 652 tokens even at a simplification level of 10%.`

An agent that asks for a feature that the model lacks, or for an interaction term, gets a tool error that names the
features of the model.

## Chat with the agent

`ebaml-chat` opens a chat in the terminal. An LLM answers your questions with the tools of the server. Ask about the
model, or about one row of the data, and ask follow-up questions. The command starts the server for you, so you need no
other command.

```bash
ebaml-chat --model model.skops --data train.csv
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | The EBM in a skops file, or in a pickled or joblib file. This option is required. The same warning as for `ebaml-mcp` applies. |
| `--data PATH` | A CSV file with the rows that you can ask about, by their number. |
| `--llm PROVIDER/MODEL` | The LLM of the agent. See [Choosing the model](providers.md). The default model needs `OPENROUTER_API_KEY`. |
| `--max-steps NUMBER` | The most rounds of tool calls for one answer. The default is `12`. |

An empty line ends the chat. The server stays up for the whole chat, so only the first answer waits for it to start.
After each answer, the chat prints the tools that the agent called, so you can check that the answer rests on tool
results. This example is shortened:

```text
Ask about the model, or about one prediction. Press Enter on an empty line to leave.
you> Explain the prediction for row 3 of the data.

For row 3, the model predicts class 0 with a probability of 0.581. Age (34) pushes toward this prediction, and so
does income (45.1). The smallest change that flips it to class 1 is to move age from 34 into [77.5, 78.5). ...

(tools: explain_prediction)
you> What would flip it?

Only one change flips it: age from 34 into [77.5, 78.5). That change raises the score by 0.781 ...

you>
```

The agent remembers the earlier messages of the chat, but not the results of earlier tool calls. It calls a tool again
when it needs a number. A turn that fails, for example because of a rate limit, prints its error, and you can ask again.

## BAML agent

`ebaml-chat` runs `chat_with_mcp`, a BAML function. BAML can act as the MCP client for one question too. `explain_with_mcp` starts `ebaml-mcp`, offers its tools to a language model, and lets
the model call them. It returns the report and the names of the tools that the model called, in order. The agent needs
the BAML toolchain and `OPENROUTER_API_KEY`, or a model that you choose with `model=`.

```bash
baml run -e 'root.explain_with_mcp("Which features drive the model?", "ebaml-mcp", ["--model", "model.skops", "--data", "train.csv"])' --output-format json
```

The `tool_calls` field shows that the report rests on tool results. A report with an empty `tool_calls` has no
evidence from the model. `explain_with_tools` is the same loop on tools that you supply, for example the tools of
another MCP server.

The [BAML agent notebook](baml-agent-notebook.md) runs the agent step by step.

### Example session: explain a prediction

The agent gives the number of a row of the data file with `row`. It can also give the value of every feature with
`instance`. The tool returns the facts as numbers (`facts`) and the same facts as the texts of the prompts (`texts`).

```json
{"name": "explain_prediction", "arguments": {"row": 0}}
```

```json
{
  "facts": {
    "prediction": {"decision": 1, "output": 0.9564728682136135, "lower": 0.9259432129958867, "upper": 0.9747598512442751, "score": 3.089867959893999, "score_std": 0.28770285958904573, "margin": 3.089867959893999},
    "contributions": [
      {"feature": "age", "value": "70", "score": 1.3517285041424543, "std": 0.2544911539901107},
      {"feature": "city", "value": "north", "score": 0.8576651042270622, "std": 0.08806194055932723},
      {"feature": "income", "value": "53", "score": 0.2693216373695486, "std": 0.10125355589519544}
    ],
    "counterfactual": {
      "changes": [
        {"feature": "age", "from_value": "70", "to_value": "[18.5, 19.5)", "score_change": -3.7239215185041488},
        {"feature": "income", "from_value": "53", "to_value": "[25.5, 26.4)", "score_change": -2.5776554527787674}
      ],
      "new_decision": 0,
      "new_output": 0.9612725381888545
    },
    "sufficient_reason": {"features": ["income", "age"], "guaranteed_margin": 0.5450951842389209, "minimum": true},
    "completions": [
      {"fixed": [], "share": 0.6124},
      {"fixed": ["income"], "share": 0.7017},
      {"fixed": ["income", "age"], "share": 1.0},
      {"fixed": ["income", "age", "city"], "share": 1.0}
    ],
    "confidence_level": 0.95
  },
  "texts": {
    "prediction": "{\n  \"predicted_class\": 1,\n  \"contributions_largest_first\": [ ... ]",
    "counterfactual": "{\n  \"flip_possible\": true,\n  \"changes\": [ ... ]",
    "sufficient_reason": "...",
    "completions": "..."
  }
}
```

The facts say:

- The model predicts the class 1 with a probability of 0.956.
- Two changes flip the prediction: lower `age` to the interval [18.5, 19.5), and lower `income` to the interval
  [25.5, 26.4).
- The values of `income` and `age` fix the prediction. If the agent fixes them and leaves `city` free, the prediction
  stays the class 1 for all completions (`share` is 1.0).

### Example session: describe with an LLM

The LLM tools exist only when the server starts with `--llm`. These calls send requests to the provider and can cost
money. A client that starts the server must give it the API key, because most clients start the server with a small
environment. With Claude Code, use `--env`:

```bash
claude mcp add --env OPENROUTER_API_KEY=sk-or-... ebaml -- ebaml-mcp --model /path/to/model.skops \
    --llm openrouter/openrouter/free
```

The agent then calls `describe_feature` with the name of a feature. This result is the reply of a live call to
`openrouter/free`, shortened. The words of a model change from call to call.

```json
{"name": "describe_feature", "arguments": {"feature": "age"}}
```

```text
The predicted effect of age begins strongly negative for ages 18-22, with a mean around -2.5 and a wide 95 % confidence
interval that stays below zero. As age increases, the effect gradually rises, crossing from negative to near-zero
between ages 46-47. ...
```

The first call of `describe_model` describes the whole model and takes longer. The server keeps the text, so the next
calls return it at once and send no request.
