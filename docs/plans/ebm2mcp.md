# Plan: EBM2MCP (0.3.0, then 1.4.0)

You are a senior Python engineer who knows the Model Context Protocol (MCP) well. You will implement one milestone of
`ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's
hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
A user gives a fitted EBM to a command, and the command serves an MCP server. An agent that connects to it can:

1. **read** the EBM: its kind, its features, their importances and the text of each graph (0.3.0),
2. **explain** one prediction with the exact local facts (0.3.0),
3. **edit** the EBM so that it meets business requirements (1.4.0, as experimental tools).

The agent is the LLM. The read and explain tools return the output of the deterministic Python code (`EBMGraph.to_text`,
`compute_local_facts`, `facts_to_texts`) and call no LLM. This keeps them fast, free and exact.
</goal>

<context>
- 0.3.0 needs 0.1.0 (custom model providers) and 0.2.0 (model kinds). The tools use the kind abstraction from 0.2.0,
  so they work for every kind from the start. The optional LLM tools use `ModelOptions` from 0.1.0.
- The MCP tool names and their argument schemas become part of the stable API at 1.0.0
  (`.claude/rules/changelog-and-versioning.md`). Design them with care: after 1.0.0, a renamed tool or argument needs a
  new MAJOR version.
- The official Python SDK is the `mcp` package (`mcp.server.fastmcp.FastMCP`). It is in `uv.lock` only as a dependency
  of `semgrep` today. Make it a direct dependency of an optional extra, so a user who does not need the server does not
  install it.
</context>

<design>
<package>
- Add a new top layer `ebaml.mcp`. It may import `llm`, `local`, `graphs` and `core`. Add it to the import-linter
  layers contract in `pyproject.toml` and to the table in `.claude/rules/architecture.md`, as that file says. Only
  `ebaml.mcp` imports `mcp`. Add an import-linter contract for that.
- Give each module one job (SOLID S), with a name that no other module of `src/ebaml` has: for example
  `mcp/server.py` (create the server and register the tools), `mcp/read_tools.py`, `mcp/explain_tools.py`,
  `mcp/model_store.py` (load the EBM and hold the session state).
- Keep each tool a thin adapter: it checks its arguments, calls a function of a lower layer, and converts the result
  to JSON with `attrs.asdict` and `orjson`. Put no domain logic in a tool.
- Add the extra `mcp = ["mcp>=..."]` in `pyproject.toml`, and a console script `ebaml-mcp`.
</package>

<command>
```text
ebaml-mcp --model PATH [--data PATH] [--llm PROVIDER/MODEL] [--transport stdio|streamable-http]
```

- `--model` loads a pickled or joblib EBM. Loading a pickle runs code from the file. Load only the path that the user
  gives, never a path that a tool argument names, and put this warning in the docs before the step that loads a model.
- `--data` is an optional CSV file with the training or validation data. The explain tools use it for the instance
  values of a row, and the edit tools (1.4.0) use it to measure the effect of an edit.
- `--llm` enables the optional LLM tools with `ModelOptions.parse`. Without it, the server has no LLM tools.
</command>

<tools_0_3_0>
| Tool | Arguments | Result |
| --- | --- | --- |
| `get_model_summary` | none | kind, outputs or classes, intercept, number of features, interaction terms that the server skips |
| `list_features` | none | name, type and importance of each feature, sorted by importance |
| `get_graph` | `feature`, optional graph options (normalization, precision, `max_tokens`) | the graph text of `EBMGraph.to_text` |
| `explain_prediction` | `instance` (feature name to value) or `row` (index into `--data`), optional `confidence_level` | the `LocalFacts` as JSON and the fact texts |
| `describe_feature` (LLM) | `feature` | the output of `describe_graph` |
| `describe_model` (LLM) | none | the output of `describe_ebm`, computed once and kept |

- Also serve each graph as a resource (`ebm://graphs/{feature}`), for clients that read resources.
- Give each tool a description that tells the agent when to use it, written to the prompt rules of
  `.claude/rules/writing-style.md`: the tool descriptions are prompts.
- Return errors of the package (`EBaMlError`) as MCP tool errors with their message. Let other errors fail.
- Until 1.1.0 and 1.2.0, `explain_prediction` returns the `UnsupportedModelError` for a regressor or a multiclass
  classifier. Design its result on the general local types of 0.2.0, so those releases add support with no change to
  the schema.
</tools_0_3_0>

<tools_1_4_0>
The edit tools wrap the `ebaml.experimental` edit API of the `model-editing.md` plan (1.3.0). Their names start with
`experimental_`, so they are outside the stable API.

| Tool | Result |
| --- | --- |
| `experimental_preview_edit` | the graph before and after one edit, and its effect on `--data` if given, with no change to the model |
| `experimental_apply_edit` | applies one edit to the session copy of the model, and returns the same report |
| `experimental_undo_edit` | undoes the last edit |
| `experimental_list_edits` | the history of the session, as replayable JSON |
| `experimental_check_constraints` | the business constraints that the model breaks (from 1.5.0) |
| `experimental_save_model` | writes the edited model and the edit history |

- The server edits a copy. It never changes the file of `--model`.
- `experimental_save_model` writes only into a directory that the user gives at start (`--output-dir`). Without that
  option, the tool is not registered. Never write to a path from a tool argument outside that directory.
</tools_1_4_0>
</design>

<tests>
- Unit tests for each tool through an in-memory client session of the `mcp` SDK, so the tests check the real schemas
  and the JSON results. Fit small models of each kind in fixtures.
- A snapshot test of the tool names and their JSON schemas. From 1.0.0 on, a change of the snapshot is a breaking change
  of the public API, so the test makes it visible in review.
- Tests for the security limits: a tool cannot load or write a path outside the paths that the user gave.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- A new page `docs/mcp.md` (and an entry in `zensical.toml`): install the extra, start the server, and connect it from
  Claude Code (`claude mcp add`), Claude Desktop and one other client. Give one example session per tool group.
- A README section that links to the page, and a row "MCP server" in the README table "Supported models".
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Each PR with a user-visible change is a release (`.claude/rules/changelog-and-versioning.md`). The PR that completes
the read and explain tools takes version 0.3.0. The PR that completes the edit tools is the 1.4.0 milestone, which is a
MINOR release.
</release>

<done>
- An agent in Claude Code can connect to `ebaml-mcp`, list the features of a model of each kind, read a graph and
  explain a prediction of a binary classifier, with no LLM key set.
- The schema snapshot test exists and passes.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass.
</done>
