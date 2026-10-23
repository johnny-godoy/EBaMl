# Preferred libraries

Use these libraries instead of their standard-library or alternative counterparts. They are declared in `pyproject.toml`
dependencies; do not add the replaced modules back.

| Need | Use | Not |
| --- | --- | --- |
| Logging | `structlog` | `logging` (and `print` for diagnostics) |
| JSON | `orjson` | `json` |
| Data classes | `attrs` | `dataclasses` |

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

## Exception

- `.github/scripts/ci_report.py` runs under the runner's bare `python3` with no project environment, so it stays
  standard-library only (`json`).
