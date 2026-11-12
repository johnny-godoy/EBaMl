# Architecture and guards

The layout of `src/ebaml/` makes the SOLID rules (`solid.md`) checkable. Tools enforce the limits below, so a change
that breaks one fails `prek run --all-files`. Put new code where the layer rules say.

## Layers

| Package | Job | May import |
| --- | --- | --- |
| `ebaml.core` | Shared kernel: errors and types. | No other `ebaml` package. |
| `ebaml.graphs` | Extract and simplify an `EBMGraph`, convert it to and from text. | `core` |
| `ebaml.local` | Compute the facts of one prediction, render them as text. | `edit`, `graphs`, `core` |
| `ebaml.edit` | Change the graphs of a fitted EBM: selections, edits, the history and the session. | `graphs`, `core` |
| `ebaml.llm` | The edge: the calls into `baml_sdk`, their options, the retry. | `local`, `edit`, `graphs`, `core` |
| `ebaml.mcp` | The MCP server: tools that read an EBM and explain a prediction. Only this layer imports `mcp`. | `llm`, `local`, `edit`, `graphs`, `core` |
| `ebaml.benchmark` | The TalkToEBM benchmark. It stands alone. | Nothing from `ebaml`. |

- Only `llm/descriptors.py` imports `baml_sdk`. The reason: domain code then runs in a plain unit test (SOLID: D).
- The import-linter contract covers every module of `ebaml.llm`, and its `ignore_imports` list names the allowed
  imports of `baml_sdk` and `baml_bridge`. Add an entry only with a reason in the `reason:` block above the list.
  `llm/sdk_setup.py` imports `baml_sdk` through `importlib`, which import-linter cannot see.
- A lower layer never imports a higher one. If `graphs` needs a service, take it as a `Callable` or `Protocol`
  parameter and let `llm` pass the real one.
- Search for an existing contract before you add a shared one (`reuse.md`), and register a new shared contract in
  `architecture/abstractions.yml`.
- Add a new feature as a new module in the layer that fits. Add a new layer only with a new entry in the import-linter
  layers contract and a new row in this table.
- Give each module a name that no other module in `src/ebaml` has. `scripts/mutation.py` finds the tests of a module by
  its name.
- Mirror the source tree in the test tree. Put the test of `src/ebaml/<pkg>/<name>.py` in
  `tests/unit/ebaml/<pkg>/test_<name>.py`. A test file or directory with no source at the mirrored path is an error.

## Guards

Each guard checks one kind of rule. Run all of them with `prek run --all-files`.

| Guard | Config | What it checks |
| --- | --- | --- |
| import-linter | `[tool.importlinter]` in `pyproject.toml` | Layer order and the limits on `baml_sdk`. Run it alone with `uv run lint-imports`. |
| Layout check | `scripts/check_layout.py` | Where files live: layers only, a docstring in each `__init__.py`, unique module names, a mirrored test tree, and the banned files. |
| semgrep | `semgrep/ebaml.yml` | Code patterns that mean a SOLID problem: `**kwargs` bags, `elif` chains on one value, `NotImplementedError` stubs, I/O in domain code, an `isinstance` check of a frame type, a bare `raise ValueError`, an `object` type hint, an `isinstance` check of a concrete container, branches on the type of one value, `range(len(x))` loops and unexplained loops over numpy arrays (`loops.md`), an SDK fake without `fake_sdk` in the tests, a `while True` loop with a sleep, a bare `asyncio.gather` or `create_task`, `skops.io.load` with `trusted=True`. |
| Abstraction registry | `architecture/abstractions.yml`, `scripts/check_abstractions.py` | One owner and one canonical symbol for each shared capability: unique IDs, symbols that exist, no second claim on a symbol or a responsibility. `reuse.md` has the process. |
| Contract overlap | `scripts/check_contract_overlap.py` | Public `Protocol` and `ABC` classes with identical signatures in one context. |
| Clones | `jscpd` hook in `prek.toml` | Any copy-paste clone in `src` (50 tokens, 5 lines). Needs Node.js. |
| Claude rules | `.claude/rules/` | The design judgment that no tool can check: one job per unit, small interfaces. |

- Fix the code when a guard fails. Do not add an exception to a guard to make it pass.
- Give each new semgrep rule a test in `semgrep/ebaml.py`, with one line that must match and one that must not. Run
  `uvx --python 3.13 --from semgrep==1.183.0 semgrep --test semgrep`.
- Add a new import limit as a contract in `pyproject.toml`, not as a semgrep rule, because import-linter sees the
  whole import graph.
