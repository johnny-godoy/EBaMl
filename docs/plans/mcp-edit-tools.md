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
- The edit tools wrap the `ebaml.edit` API of the `model-editing.md` plan (0.7.0). Read its `<context>` first: it says
  what GAM Changer gives a person, and the tools must give an agent the same. The tool names and their argument schemas
  become part of the stable API at 1.0.0. Until then, a change of them is allowed, marked **BREAKING**.
- `tests/unit/ebaml/mcp/test_server.py` has a snapshot test of the tool names and their JSON schemas.
- GAM Changer protects a model from a harmful edit with four things: a preview, feedback on data, a history with a
  written reason for each edit, and a confirmation by a person before the save (Wang et al., KDD 2022, sections 3.2
  and 6). An agent edits faster than a person and sees no canvas, so each of the four must be in the tools.
</context>

<decisions>
The user answered these questions. Each answer binds you. If the code shows that an answer no longer fits, ask the user
again, and do not reinterpret it.

1. **Who confirms an edit.** `save_model` asks the person through MCP elicitation, with the list of the edits and their
   reasons. If the client does not support elicitation, it writes the model and marks each edit "not reviewed by a
   person" in the history file.
2. **Revisions.** The read tools (`get_graph` and the others) take `revision`: `original`, `current` or the id of a
   revision of the history (`model-editing.md`, `<session>`). The default is `current`, and each result says how many
   edits the session holds. A read of a revision never changes the model of the session.
3. **History.** A tool returns the history as a table of commits (id, time, feature, edit type, description, reason,
   reviewed). `save_model` needs a `message` (the change that the saved model holds), adds a `save` entry to the table,
   and writes the table next to the model (`model-editing.md`, `<session>`). The agent writes the message, and the
   person sees it in the elicitation of decision 1.
4. **Re-centring.** The descriptions of the edit tools tell the agent that a direct edit does not re-centre, that the
   report shows the shift of the weighted mean, and when to call `Recenter`.
</decisions>

<tools>
| Tool | Result |
| --- | --- |
| `list_bins` | the bin edges or the categories of a feature, with the training weight and the score of each bin, and the missing and unseen bins |
| `preview_edit` | the report of one edit (the graph before and after, the changed bins, the share of the training weight), and its effect on `--data` if given, with no change to the model |
| `apply_edit` | applies one edit to the session copy of the model, and returns the same report |
| `undo_edit`, `redo_edit` | undoes the last edit, or applies it again |
| `checkout_edit` | sets the model to its state after one entry of the history |
| `list_edits` | the history of the session as a table of commits, and as replayable JSON on request |
| `check_constraints` | the business constraints that the model breaks, each with the edit that repairs it (from 0.9.0) |
| `affected_rows` | the Feature Panel of GAM Changer: which other features differ most between the affected rows and all rows (from 0.9.0, needs `--data`) |
| `save_model` | needs a `message`, adds a `save` entry to the history, and writes the edited model and the history table |

- The edit is one argument with a `kind` field (`set_scores`, `monotonize`, and so on) and the fields of that edit. The
  JSON schema lists each kind, so the agent sees every edit without a tool for each one. A new edit of `ebaml.edit` is
  then a new entry of one table, and no new tool.
- `apply_edit` needs a `reason`: the business requirement that the edit serves, in one or two sentences. The history
  keeps it. The paper finds that auditors read the history as model documentation, and an edit with no reason cannot
  be reviewed.
- `list_bins` exists because a selection must end on bin edges. The error of a wrong selection also lists the nearest
  edges, so the agent can repair its call.
- Write each tool description as a prompt (`.claude/rules/writing-style.md`). Tell the agent the working order and the
  reason for it: read the graph, preview, check the effect on the data and on subgroups, apply with a reason, and tell
  the person what changed before the save. Tell it that an edit changes real predictions, so it applies only what the
  person asked for, and proposes the edit that changes the fewest scores when several meet the requirement.
- A report says only what the numbers show (`.claude/rules/semantic-contracts.md`, contract 3). It names the pair terms
  that still use the feature, so the agent does not tell the person that the whole model is monotone.
- Put the edit tools in a module of their own (`mcp/edit_tools.py`), with a name that no other module of `src/ebaml`
  has.
- The server edits a copy. It never changes the file of `--model`.
- `ModelStore.model_description` caches the output of `describe_ebm`. Clear it after each edit, undo, redo and
  checkout, or `describe_model` describes a model that no longer exists. Ship that with a test that fails without it.
- `--edits <file>` replays a saved history at start, so a later session continues the work. The replay refuses a
  history whose model fingerprint differs from the model of `--model`.
- `save_model` writes only into a directory that the user gives at start (`--output-dir`). Without that option, the
  tool is not registered. Never write to a path from a tool argument outside that directory. It writes the model with
  `skops.io` and the history as JSON, and it never overwrites a file: each save gets a new name.
- Without `--data`, the effect on data is absent, and the report says so in one field. The agent then knows that it
  did not check the effect, and can tell the person.
</tools>

<tests>
- Unit tests for each tool through an in-memory client session of the `mcp` SDK, as the existing tests do.
- Update the schema snapshot on purpose.
- Tests for the security limits: a tool cannot write a path outside the directory that the user gave, a save does not
  overwrite a file, and the file of `--model` has the same bytes after a session of edits.
- A test that each read tool and the explain tool use the edited copy after an edit, and that `describe_model` does
  not return the cached description of the model before the edit.
- A test of the confirmation of decision 1, for a client with elicitation and for one without.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- Add a section on the edit tools to `docs/mcp.md`, with one example session (a requirement, the preview, the effect,
  the apply with its reason, the save), and the `--output-dir` and `--edits` options.
- Extend `notebooks/mcp_server.py` with the edit tools, or add a notebook, as `.claude/rules/notebooks.md` says.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Add a changelog fragment (`.claude/rules/changelog-and-versioning.md`). The PR that completes the milestone also sets
`version = "0.8.0"` in `pyproject.toml`, runs `uv lock`, and marks its row of `ROADMAP.md` as done. The tools
`check_constraints` and `affected_rows` arrive with 0.9.0, in the PRs of `model-editing.md`.
</release>

<done>
- An agent in Claude Code can list the bins, preview, apply with a reason, undo and redo an edit, and save the edited
  model with its history into the output directory, after the confirmation of decision 1.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass.
</done>
