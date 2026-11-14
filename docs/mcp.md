# MCP server

`ebaml-mcp` gives an agent a fitted EBM through the [Model Context Protocol](https://modelcontextprotocol.io). The
agent can read the model, read each graph and explain one prediction. The agent can also edit the model so that it meets
a business requirement, and save the edited model after a person confirms the edits. The read, explain and edit tools
call no LLM. They are fast, free and exact, and they need no API key.

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
ebaml-mcp --model model.skops --data train.parquet
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | The EBM in a skops file, or in a pickled or joblib file. This option is required. |
| `--data PATH` | A Parquet or CSV file with the rows for the `row` argument of `explain_prediction`. |
| `--llm PROVIDER/MODEL` | Turn on the LLM tools. See [Choosing the model](providers.md). Without it, the server has no LLM tools. |
| `--output-dir DIR` | The directory where `save_model` writes the edited model and its history. The server creates it if it does not exist. Without this option, the server has no `save_model` tool. |
| `--edits PATH` | The history file of an earlier session (`*.history.json`). The server replays it at start, so the work goes on. The server refuses a history that started from another model. |
| `--transport` | `stdio` (the default) or `streamable-http`. |
| `--host ADDRESS` | The address of the `streamable-http` transport. The default is `127.0.0.1`, which accepts this computer only. |
| `--port NUMBER` | The port of the `streamable-http` transport. The default is `8000`. |

The server reads the first bytes of the data file, not its extension. A file that starts and ends with the Parquet mark `PAR1` is a Parquet file. Parquet keeps the type of each column. A file of UTF-8 text is a CSV file, and the server infers the types from the text. An empty file, a Parquet file that is cut off, and any other binary file raise a `DataFileError` that names the file and says how to save the rows.

## Connect a client

Claude Code:

```bash
claude mcp add ebaml -- ebaml-mcp --model /path/to/model.skops --data /path/to/train.parquet
```

Claude Desktop. Add this entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ebaml": {
      "command": "ebaml-mcp",
      "args": ["--model", "/path/to/model.skops", "--data", "/path/to/train.parquet"]
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
ebaml-mcp --model model.skops --data train.parquet --transport streamable-http --port 9000
```

The client connects to `http://127.0.0.1:9000/mcp`. The server has no login. Keep the default host `127.0.0.1`
unless a firewall or a proxy with authentication protects the port. With `--host 0.0.0.0`, every computer that
reaches the port can read the model and the rows of the data file.

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `get_model_summary` | optional `revision` | The kind, the outputs or classes, the intercept, the number of features, and the interaction terms that the server skips. |
| `list_features` | optional `revision` | The name, the type and the importance of each feature, the most important first. |
| `get_graph` | `feature`, optional `revision`, `normalization`, `x_axis_precision`, `y_axis_precision`, `max_tokens` | The graph text of `EBMGraph.to_text`. |
| `explain_prediction` | `instance` or `row`, optional `revision`, `confidence_level` | The `LocalFacts` and their texts, as JSON. |
| `list_bins` | `feature`, optional `revision` | The bins of a feature with the number, the edges, the training weight and the score of each bin, and the missing and the unseen bin. |
| `preview_edit` | `edit`, optional `revision` | The report of one edit and its effect on the rows of `--data`. It changes nothing. |
| `apply_edit` | `edit`, `reason`, optional `expected_revision` | Applies one edit to the session copy of the model, and returns the same result as `preview_edit`. |
| `undo_edit`, `redo_edit` | optional `expected_revision` | Undoes the last edit, or applies it again. |
| `checkout_edit` | `revision`, optional `expected_revision` | Sets the model to its state after one commit of the history. |
| `list_edits` | optional `include_replay_json` | The history as a table of commits, and on request the JSON that `--edits` reads. |
| `save_model` | `message`, optional `expected_revision` | Saves the edited model and its history after a person confirms. It needs `--output-dir`. |
| `describe_feature` | `feature` | The words of `describe_graph`. It needs `--llm`. |
| `describe_model` | none | The words of `describe_ebm` for the current model. The server keeps the text for the revision that it describes. It needs `--llm`. |

The server also serves each graph as the resource `ebm://graphs/{feature}`.

`explain_prediction` works for classifiers and regressors without interactions. For other models, it returns an error
with the reason. A package error is a tool error with its message.

Each JSON result has a `session` field. It names the revision that the call read, the head of the session, and the
number of edits that the head holds. `get_graph` writes the same facts in its first line.

The edit tools work for binary classifiers and regressors. For a multiclass classifier, they return an error, and the
read tools still work.

The examples below come from a real session over stdio. The model is an EBM with `interactions=0` that was fitted on
500 synthetic rows with the features `age`, `income` and `city`. The server started with
`--model model.skops --data train.parquet`. A result that is long is shortened, and `...` marks the cut.

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
      "skipped_terms": [],
      "session": {"revision": "original", "head": "original", "edit_count": 0}
    }
    ```

2. The agent calls `list_features` to choose a graph. The most important feature is first.

    ```json
    {"name": "list_features", "arguments": {}}
    ```

    ```json
    {
      "session": {"revision": "original", "head": "original", "edit_count": 0},
      "features": [
        {"name": "income", "type": "continuous", "importance": 1.1480326430918357},
        {"name": "age", "type": "continuous", "importance": 1.1147650229232495},
        {"name": "city", "type": "nominal", "importance": 0.840511802142521}
      ]
    }
    ```

3. The agent calls `get_graph` for one feature. The result is text, not JSON. The server merges the bins of a
   continuous graph until the estimated size is near `max_tokens`. Here the budget is 1000 tokens, and the bins
   `(18.5, 19.5)`, `(19.5, 20.5)` and `(20.5, 21.5)` became the bin `(18.5, 21.5)`.

    ```json
    {"name": "get_graph", "arguments": {"feature": "age", "max_tokens": 1000}}
    ```

    ```text
    Revision original. The head is original, and it holds 0 applied edits.
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

## Edit the model

An edit changes the scores of the bins of one graph. It changes real predictions. Apply only the edits that the person
asked for, and check the effect of an edit before you apply it. A written reason and a confirmation by a person protect
the model from a harmful edit, as in [GAM Changer](https://interpret.ml/gam-changer/) (Wang et al., KDD 2022). The edit
tools call the edit API of Python ([EditSession](editing.md)) and call no LLM. The agent is the LLM.

The server edits a copy of the model. It never changes the file of `--model`. A tool that reads the model, such as
`get_graph` or `explain_prediction`, reads the edited copy.

### The working order

1. Read the graph with `get_graph` and the bins with `list_bins`. A selection must end on the edges of the bins.
2. Call `preview_edit`. Read the changed bins, the share of the training weight that they hold, and the effect on the
   data.
3. Call `apply_edit` with a `reason`: the business requirement that the edit serves, in one or two sentences. The history
   keeps it, and a person reads it later.
4. Tell the person what changed. Call `save_model` with a `message` when the person wants the edited model.

If several edits meet the requirement, propose the one that changes the fewest scores.

### The edit argument

The `edit` argument is one JSON object. Its field `kind` names the edit, and the other fields are the fields of that
edit. The description of `preview_edit` and `apply_edit` lists every edit with its fields. A new edit of `ebaml.edit`
appears in the list with no new tool. A selection is a JSON object with a `kind` too.

| `kind` of the edit | Effect |
| --- | --- |
| `Shift` | Adds a value to the selected scores. |
| `SetScores` | Sets the selected scores to one value. |
| `Align` | Sets the selected scores to the score of the left bin, of the right bin, or to the weighted mean. |
| `Monotonize` | Makes the selected scores monotone, weighted by the training weight of the bins. |
| `Interpolate`, `FitLine` | Replaces the selected scores by a straight line, or by the weighted least-squares line. |
| `Scale` | Multiplies the graph by a factor. |
| `ZeroTerm`, `RemoveTerms`, `RemoveFeature` | Set a graph to 0, or remove terms. The model keeps its input columns. |
| `Recenter` | Moves the weighted mean of a graph into the intercept. No prediction changes. |

| `kind` of the selection | Bins |
| --- | --- |
| `Bins` | The bin numbers from `first` to `last`, both included. A bin number is the number that `list_bins` gives. |
| `Categories` | The bins of some categories of a nominal feature. |
| `ValueInterval` | Every bin that overlaps an interval of values. The report names the bins that it chose. |
| `MissingBin`, `UnseenBin`, `WholeGraph` | The missing bin, the unseen bin, or every real bin. |

An edit that sets scores does not re-centre the graph. The report shows the shift of the weighted mean of the graph
(`weighted_mean_shift`). Call `Recenter` when the person wants the shift in the intercept. A score that a person sets
has no fitted uncertainty, so its standard deviation is 0.

### The result of an edit

`preview_edit` and `apply_edit` return the report, its text, and the effect on the rows of `--data`. The report lists
the changed bins with their score before and after, the largest change, the share of the training weight that the
changed bins hold, the shift of the weighted mean, and the pair terms that still use the feature. The edit changes the
main effect only, so a pair term can still use the feature.

The effect on the data counts the rows whose prediction changes, and for a classifier the rows whose class changes. It
needs no labels. It does not say whether the edit is good. Without `--data`, the field `effect_on_data` is empty, and
`effect_on_data_note` says that nobody checked the effect. Tell the person when this happens.

### Revisions and the history

Each edit is a commit with an id. A read tool takes `revision`: `original`, `current` or the id of a commit. A read of a
revision never changes the model. `list_edits` shows the history as a table of commits: the id, the time, the feature,
the type, a description, the reason, and whether a person reviewed it. `undo_edit` and `redo_edit` move the head, and
`checkout_edit` sets it to any commit. A new edit after an undo drops the edits that the undo moved to the redo stack.

Several clients can edit at the same time. A tool that changes the model takes `expected_revision`, the head that the
agent read. If the head is another one, the tool fails and changes nothing. The error names the current head, so the
agent reads again and decides if its call still fits.

### Save the edited model

Start the server with `--output-dir DIR` to turn on `save_model`. The tool needs a `message`: the change that the saved
model holds. It asks the person to confirm. The question shows the message, each edit and its reason. The tool then writes
the model with `skops.io` and the history as JSON into the directory. Each save gets a new name, and a save never
overwrites a file. No tool argument names a path.

If the person does not confirm, the tool saves nothing. If the client cannot ask a person, the tool saves the model and
the history marks each edit as not reviewed by a person. The result names the case in the field `confirmation`:
`person_confirmed`, `no_edits_to_confirm` or `client_cannot_ask`. If the model changes while the person decides, the
tool asks again or fails, so a confirmation of an older model never saves a newer one.

Start the next session with `--edits PATH` and the first history file to continue the work. The server replays the edits
on the model of `--model`, and it refuses a history that started from another model.

### Example session: edit the model

The model is the same as above. The requirement is: "The city must not decide the approval." The result of each call is
shortened.

1. The agent reads the bins of `city`.

    ```json
    {"name": "list_bins", "arguments": {"feature": "city"}}
    ```

    ```json
    {
      "session": {"revision": "original", "head": "original", "edit_count": 0},
      "feature": "city",
      "feature_type": "nominal",
      "bins": [
        {"number": 0, "label": "north", "score": 0.8576651042270622, "weight": 245.0, "weight_share": 0.49},
        {"number": 1, "label": "south", "score": -0.824031178571099, "weight": 255.0, "weight_share": 0.51}
      ],
      "pair_terms_using_feature": []
    }
    ```

2. The agent previews the edit that sets the score of `north` to 0. The preview changes nothing.

    ```json
    {"name": "preview_edit", "arguments": {"edit": {"kind": "SetScores", "feature": "city", "selection": {"kind": "Categories", "names": ["north"]}, "value": 0.0}}}
    ```

    ```json
    {
      "applied": false,
      "report": {
        "description": "Set 1 bin of city (north) to 0",
        "changes": [{"position": 1, "label": "north", "before": 0.8576651042270622, "after": 0.0}],
        "weight_share": 0.49,
        "weighted_mean_shift": -0.4202559010712605,
        "intercept_change": 0.0
      },
      "effect_on_data": {"row_count": 500, "rows_with_changed_value": 245, "rows_with_changed_class": 31, "largest_value_change": 0.2111567810873875},
      "effect_on_data_note": null
    }
    ```

    The edit reaches 245 of 500 rows, and the class changes for 31 of them. The agent tells the person these numbers
    before it applies the edit.

3. The agent applies the edit with the reason. It passes the head that it read.

    ```json
    {"name": "apply_edit", "arguments": {"edit": {"kind": "SetScores", "feature": "city", "selection": {"kind": "Categories", "names": ["north"]}, "value": 0.0}, "reason": "The city must not decide the approval.", "expected_revision": "original"}}
    ```

    The result has the same fields as the preview. Its `session` field names the new head:

    ```json
    {"session": {"revision": "01a1d776-fd39-7094-b038-7b55279be217", "head": "01a1d776-fd39-7094-b038-7b55279be217", "edit_count": 1}, "applied": true}
    ```

4. The agent saves the model with a message. The client asks the person, and the person answers yes.

    ```json
    {"name": "save_model", "arguments": {"message": "Approval rules of the lending policy"}}
    ```

    ```json
    {
      "model_file": "edited-model-001.skops",
      "history_file": "edited-model-001.skops.history.json",
      "confirmation": "person_confirmed",
      "commit": {"kind": "save", "reason": "Approval rules of the lending policy", "reviewed": true}
    }
    ```

The [MCP server notebook](mcp-notebook.md) runs this session with a real server.

## Chat with the agent

`ebaml-chat` opens a chat in the terminal. An LLM answers your questions with the tools of the server. Ask about the
model, or about one row of the data, and ask follow-up questions. The command starts the server for you, so you need no
other command.

```bash
ebaml-chat --model model.skops --data train.parquet
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | The EBM in a skops file, or in a pickled or joblib file. This option is required. The same warning as for `ebaml-mcp` applies. |
| `--data PATH` | A Parquet or CSV file with the rows that you can ask about, by their number. |
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

The [chat notebook](chat-notebook.md) runs the same command behind a chat box.

The agent remembers the earlier messages of the chat, but not the results of earlier tool calls. It calls a tool again
when it needs a number. A turn that fails, for example because of a rate limit, prints its error, and you can ask again.

## BAML agent

`ebaml-chat` runs `chat_with_mcp`, a BAML function. BAML can act as the MCP client for one question too. `explain_with_mcp` starts `ebaml-mcp`, offers its tools to a language model, and lets
the model call them. It returns the report and the names of the tools that the model called, in order. The agent needs
the BAML toolchain and `OPENROUTER_API_KEY`, or a model that you choose with `model=`.

```bash
baml run -e 'root.explain_with_mcp("Which features drive the model?", "ebaml-mcp", ["--model", "model.skops", "--data", "train.parquet"])' --output-format json
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
    "sufficient_reason": {"features": ["income", "age"], "guaranteed_margin": 0.5450951842389209, "minimality": "sorting_proof"},
    "completions": [
      {"fixed": [], "share": 0.6124},
      {"fixed": ["income"], "share": 0.7017},
      {"fixed": ["income", "age"], "share": 1.0},
      {"fixed": ["income", "age", "city"], "share": 1.0}
    ],
    "confidence_level": 0.95
  },
  "session": {"revision": "original", "head": "original", "edit_count": 0},
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
