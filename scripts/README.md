# scripts

Repository maintenance scripts. Run them from the repo root with `uv run python scripts/<name>.py`.

| Script | What it does |
| --- | --- |
| `check_ignores.py` | Fails unless every lint/type/coverage/test workaround has a written reason next to it. |
| `compile_ignores.py` | Compiles every workaround and its reason into one Markdown report to audit. |
| `mutation.py` | Runs cosmic-ray mutation testing in parallel shards and fails if any mutant survives. |
| `docs_hooks.py` | MkDocs hook: runs the marimo notebook and exports its output into the built docs (`mkdocs.yml`). |
| `live_openrouter.py` | Optional live test: real OpenRouter calls through the BAML pipelines (needs `OPENROUTER_API_KEY`). |

## `check_ignores.py` and `compile_ignores.py`

A *workaround* silences a tool instead of fixing the code:

- inline comments: `noqa`, `type: ignore`, `pyrefly: ignore`, `fmt: off`, `pragma: no cover`, `pragma: no mutate`
- `pytest.mark.skip` / `skipif` / `xfail`, and `pytest.skip()` / `xfail()`
- ignore, exclude and omit settings in `pyproject.toml` and `cosmic-ray.toml`

Each workaround carries its reason right next to it, so the reason moves with the code and cannot go stale:

```python
run(argv)  # noqa: S603 -- fixed argv from sys.executable, never untrusted input
```

```toml
# reason: recurring pattern in tests/: bare assert and literal expected values are how pytest tests are written.
"tests/**/*.py" = ["assert", "magic-value-comparison"]
```

- Python comments end in ` -- <reason>`; pytest markers and calls may pass `reason="..."` instead.
- TOML settings get a comment block directly above the key, whose first line starts with `reason:`.
- The reason must say why the code cannot be fixed, not restate what the ignore does.

`check_ignores` fails on any workaround without a reason (or with a one-word one). `compile_ignores` gathers every
workaround and its reason into one Markdown report, grouped by tool, so they can be audited in one place. The report is
generated: do not commit or edit it.

```bash
uv run python -m scripts.check_ignores                          # fails if a reason is missing
uv run python -m scripts.compile_ignores                        # report to stdout
uv run python -m scripts.compile_ignores -o ignores-report.md   # report to a file
```

`check_ignores` also runs as the `check-ignores` hook in `prek.toml`. Do not edit the scripts or the hook to make a
violation pass. The finder shared by both is `ignore_sites.py`.

## `mutation.py`

cosmic-ray mutates the source tree in place, so each shard works on its own copy of `src`, `tests`, `pyproject.toml`
and `cosmic-ray.toml`. The script:

1. creates a session in `.mutation/session.sqlite` and drops mutants on `# pragma: no mutate` lines,
2. runs the test baseline and stops if it fails,
3. splits the work items over the shards and runs them concurrently,
4. merges the results back and prints the report and survival rate.

```bash
uv run python scripts/mutation.py      # one shard per CPU
uv run python scripts/mutation.py 4    # four shards
```

Exits non-zero if any mutant survives. CI requires a 100% mutation score. Settings (module path, timeout, test command)
are in `cosmic-ray.toml`.
