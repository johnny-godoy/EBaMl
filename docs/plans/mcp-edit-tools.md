# Plan: EBM2MCP edit tools (0.8.0)

You are a senior Python engineer who knows the Model Context Protocol (MCP) well. You will implement one milestone of
`ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's
hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
The `ebaml-mcp` server gets tools that let an agent **edit** the EBM so that it meets business requirements. The agent
is the LLM. The edit tools call the deterministic edit API of Python and call no LLM.
</goal>

<context>
- The server exists: `src/ebaml/mcp/` holds `server.py` (the command and the tool registration), `read_tools.py`,
  `explain_tools.py`, `llm_tools.py` and `model_store.py` (the loaded model, the data of `--data`, the session state).
  `docs/mcp.md` documents it. Read them first, and follow their patterns: thin tools, `EBaMlError` as tool errors, tool
  descriptions written as prompts.
- The edit tools wrap the `ebaml.edit` API of the `model-editing.md` plan (0.7.0). The tool names and their argument
  schemas become part of the stable API at 1.0.0. Until then, a change of them is allowed, marked **BREAKING**.
- `tests/unit/ebaml/mcp/test_server.py` has a snapshot test of the tool names and their JSON schemas.
</context>

<tools>
| Tool | Result |
| --- | --- |
| `preview_edit` | the graph before and after one edit, and its effect on `--data` if given, with no change to the model |
| `apply_edit` | applies one edit to the session copy of the model, and returns the same report |
| `undo_edit` | undoes the last edit |
| `list_edits` | the history of the session, as replayable JSON |
| `check_constraints` | the business constraints that the model breaks (from 0.9.0) |
| `save_model` | writes the edited model and the edit history |

- Put the edit tools in a module of their own (`mcp/edit_tools.py`), with a name that no other module of `src/ebaml`
  has.
- The server edits a copy. It never changes the file of `--model`.
- `save_model` writes only into a directory that the user gives at start (`--output-dir`). Without that option, the
  tool is not registered. Never write to a path from a tool argument outside that directory.
</tools>

<tests>
- Unit tests for each tool through an in-memory client session of the `mcp` SDK, as the existing tests do.
- Update the schema snapshot on purpose.
- Tests for the security limits: a tool cannot write a path outside the directory that the user gave.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- Add a section on the edit tools to `docs/mcp.md`, with one example session, and the `--output-dir` option.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Add a changelog fragment (`.claude/rules/changelog-and-versioning.md`). The PR that completes the milestone also sets
`version = "0.8.0"` in `pyproject.toml`, runs `uv lock`, and marks its row of `ROADMAP.md` as done.
</release>

<done>
- An agent in Claude Code can preview, apply and undo an edit, and save the edited model into the output directory.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass.
</done>
