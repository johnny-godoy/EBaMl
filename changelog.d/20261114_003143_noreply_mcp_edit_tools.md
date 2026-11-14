### Added

- The MCP server has edit tools: `list_bins`, `preview_edit`, `apply_edit`, `undo_edit`, `redo_edit`, `checkout_edit` and `list_edits`. An agent edits a copy of the model with a written reason for each edit. The tools call no LLM.
- `preview_edit` and `apply_edit` count the rows of `--data` that an edit changes. They say in one field when the server has no data.
- `save_model` saves the edited model and its history after the person confirms the edits through MCP elicitation. If the client cannot ask, it saves and marks each edit as not reviewed by a person. `--output-dir` turns it on.
- `--edits` replays the history file of an earlier session, so the work goes on. The server refuses a history that started from another model.
- The read tools and `explain_prediction` take `revision` (`original`, `current` or the id of a commit) and report the revision that they read and the number of edits that the head holds.
- `EditSession.revision_info`, `EditSession.history_table(pending_only=True)` and the fields `base_revision` and `revision` of `EditReport`.

### Changed

- **BREAKING:** `list_features` returns an object with `session` and `features`, and each JSON result of the read and explain tools has a `session` field. `get_graph` starts with a line that names the revision.
- **BREAKING:** `ModelStore` keeps an `EditSession` in `session`, and the model of the loaded file is `original`. `explain_prediction` of `ebaml.mcp.explain_tools` takes a `PredictionRequest`.
- `describe_model` keeps its text for the revision that it describes, so an edit never leaves a description of the model before the edit.
- The `mcp` extra needs `pydantic`, which the confirmation of a save uses.
