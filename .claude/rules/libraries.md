# Preferred libraries

Use these libraries instead of their standard-library or alternative counterparts. They are declared in `pyproject.toml`
dependencies; do not add the replaced modules back. Ruff fails on an import of `dataclasses`, `json`, `logging` and
`pandas` (`[tool.ruff.lint.flake8-tidy-imports.banned-api]` in `pyproject.toml`).

| Need | Use | Not |
| --- | --- | --- |
| Logging | `structlog` | `logging` (and `print` for diagnostics) |
| JSON | `orjson` | `json` |
| Data classes | `attrs` | `dataclasses` |
| Runtime type checks | `beartype` | `typeguard`, manual `isinstance` checks of hints |
| Data frames that a caller passes | `narwhals` | `pandas` imports, `isinstance` checks of a frame type |
| Data frames that ebaml reads itself | `polars` | `pandas` |

## Logging (`structlog`)

- Module-level logger: `logger = structlog.get_logger(__name__)`.
- Log an event name plus key-value pairs (`logger.info("usage", what=what, input_tokens=n)`); never %-format or f-string
  the message.
- Tests capture logs with `structlog.testing.capture_logs()` and assert on the event dicts, not on `caplog`.

## JSON (`orjson`)

- `orjson.loads` accepts `bytes` or `str`; prefer `path.read_bytes()` over `read_text()`.
- `orjson.dumps` returns `bytes`: write with `write_bytes`, and use `option=orjson.OPT_INDENT_2` instead of `indent=2`.
- Catch `orjson.JSONDecodeError`.

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

## Data frames (`narwhals`, `polars`)

- A function that takes a frame from the caller wraps it with `narwhals.from_native` and calls the narwhals API. The
  caller keeps its own library, and ebaml imports none of them. Do not branch on `pd.DataFrame` or `pl.DataFrame`.
- Code that reads a file itself uses polars (`nw.read_csv(path, backend="polars")`).
- Ruff bans `pandas` (`banned-api`), so no module imports it without an inline `# ruff: ignore[banned-api] -- <reason>`.
  `PANDAS_NA_TYPE` in `ebaml.core.custom_types` is the one such import in `src/`: `pd.NA` has a type of its own.
  Tests and two notebooks import pandas with the same ignore, because `interpret` fits on pandas frames.
- The semgrep rule `ebaml-no-frame-type-branches` fails on `isinstance` checks of a pandas or polars frame in `src/`,
  `scripts/` and `benchmark/`.

## Exception

- `.github/scripts/ci_report.py` runs under the runner's bare `python3` with no project environment, so it stays
  standard-library only (`json`).
