# Preferred libraries

Use the library that the section below names, and do not add its replacement back. The banned imports are the single
source in `[tool.ruff.lint.flake8-tidy-imports.banned-api]` of `pyproject.toml`: each entry names the module and its
replacement. Read that table when you need the list, and do not copy it here, because a copy goes stale. Ruff fails on
a banned import. An exception is an inline `# ruff: ignore[banned-api] -- <reason>`.

`scripts/check_layout.py` fails on the files of a replaced tool (`BANNED_FILES`: a `requirements.txt`, a
`.pre-commit-config.yaml`, a pickled model, a Jupyter notebook, a Sphinx or MkDocs config).

## Logging (`structlog`)

- Module-level logger: `logger = structlog.get_logger(__name__)`.
- Log an event name plus key-value pairs (`logger.info("usage", what=what, input_tokens=n)`); never %-format or f-string
  the message.
- Tests capture logs with `structlog.testing.capture_logs()` and assert on the event dicts, not on `caplog`.

## JSON (`orjson`)

- `orjson.loads` accepts `bytes` or `str`; prefer `path.read_bytes()` over `read_text()`.
- `orjson.dumps` returns `bytes`: write with `write_bytes`, and use `option=orjson.OPT_INDENT_2` instead of `indent=2`.
- Catch `orjson.JSONDecodeError`.

## Command line (`cyclopts`)

- Build a command from a function: the type hints are the interface, and the docstring (numpy style) is the help text.
  Create the app with `result_action="return_value"`, so that a successful run returns the value and does not call
  `sys.exit`.
- Keep `main(argv)` as a thin wrapper that calls `app(list(argv))`, so that a test passes the arguments.
- Raise `ValueError` in a converter. Cyclopts reports it as a usage error.
- Use `cyclopts.Group` with `cyclopts.validators.MutuallyExclusive` or `LimitedChoice` for flags that exclude each other.

## Tests of the code in the docs (`sybil`)

- `tests/docs/conftest.py` runs the `python` code blocks of `README.md` and `docs/*.md` and the docstring `Examples`
  with Sybil. One runner covers both, and it can give a file setup that the reader does not see.
- Do not run doctest itself (`--doctest-modules`, `--doctest-glob`, `doctest.testmod`), because a second runner skips
  that setup. Ruff bans the `doctest` import. Sybil reads the option flags of doctest, so its configuration keeps an
  inline ignore.

## Saved models (`skops.io`)

- Save and load a fitted model with `skops.io`. It loads only the types that the caller trusts, so a file runs no code.
  Pass the names of the trusted types to `skops.io.load(path, trusted=[...])`. Never pass `trusted=True`: the semgrep
  rule `ebaml-skops-load-trusted-true` fails on it.
- `pickle`, `shelve`, `dill`, `cloudpickle` and `joblib` run code when they load a file, so ruff bans them.
- The one exception is `ebaml.mcp.model_store`, which also loads the joblib files that users saved. The ignore on the
  import gives the reason. Tests that write such a file carry the same ignore.

## Data classes (`attrs`)

- Use `@attrs.define` (or `@attrs.frozen` for immutable values) rather than `@dataclass`; use `attrs.field` and
  `attrs.asdict`.

## Runtime type checks (`beartype`)

- `src/ebaml/__init__.py` calls `beartype_this_package()` before it imports any submodule, so every function in `ebaml`
  checks its hints. Keep that call first. A module imported earlier stays unchecked.
- Do not add `@beartype` to a function in `ebaml`. The hook already covers it.
- A hint that is wrong fails at run time. Fix the hint (see `types.md`), and do not remove the check.
- Pytest turns `BeartypeClawDecorWarning` into an error, because beartype skips a function that it cannot decorate.
- Checks of annotated variables are off (`claw_is_pep526=False`): structlog's lazy logger proxy fails its own hint.
- An import that only a hint uses is a run-time import, because beartype reads the hint. So ruff's rules that move
  such imports into a `TYPE_CHECKING` block are off.

## Data frames (`narwhals`, `polars`)

- A function that takes a frame from the caller wraps it with `narwhals.from_native` and calls the narwhals API. The
  caller keeps its own library, and ebaml imports none of them. Do not branch on `pd.DataFrame` or `pl.DataFrame`.
- Code that reads a file itself uses polars (`nw.read_csv(path, backend="polars")`).
- No module imports `pandas` without the inline ignore above.
  `PANDAS_NA_TYPE` in `ebaml.core.custom_types` is the one such import in `src/`: `pd.NA` has a type of its own.
  Tests import pandas with the same ignore, because they build the frames that `interpret` fits on.
- Notebooks use polars. `interpret` ignores the column names of a polars frame, so pass `feature_names=frame.columns`
  to the EBM.
- The semgrep rule `ebaml-no-frame-type-branches` fails on `isinstance` checks of a pandas or polars frame in `src/`,
  `scripts/` and `benchmark/`.

## Exception

- `.github/scripts/ci_report.py` runs under the runner's bare `python3` with no project environment, so it stays
  standard-library only (`json`).
