# prek and ignores

## Run prek before finishing

- After changing any file, run `prek run --all-files` and fix every failure before reporting the task done or committing.
- Files not yet tracked by git are skipped by `--all-files`; run `prek run --files <paths>` for those.
- Hooks may rewrite files (ruff fixes, formatters). Re-run until it passes cleanly, and re-read files you plan to edit again.
- `.claude/settings.json` runs `prek run --files <file>` after each `Edit`, `Write` or `NotebookEdit`. If it reports a failure, fix it before the next edit. It checks one file, so the full `prek run --all-files` is still needed before you finish.
- Never bypass hooks (`--no-verify`, `SKIP=...`, or removing/editing a hook to make it pass) unless the user explicitly asks.
- Do not report "done" with prek failing. If a failure is unrelated or cannot be fixed, say so with the output.

## Cooldown exemptions expire

`exclude-newer-package` in `pyproject.toml` skips the 7-day uv cutoff for one package, for example to take a security
fix. Use it only for a security fix, and write the `reason:` block above it. The `check-exclusions` hook fails when the
locked version is older than the cutoff. Then delete the entry and run `uv lock`. Do not edit the hook to keep an entry.

## Workarounds need a written reason next to them

A workaround silences a tool instead of fixing the code: `# noqa`, `# type: ignore`, `# pyrefly: ignore`, `# fmt: off`,
`# pragma: no cover`, `# pragma: no mutate`, `pytest.mark.skip/skipif/xfail`, `pytest.skip()`, and ignore/exclude/omit
settings in `pyproject.toml` or `cosmic-ray.toml`. Fix the root cause first; a workaround is a last resort.

### Count before you add

A copied ignore hides its cause, and the copies grow with each feature. An audit found about 60 sites that were copies
of four causes. So search first, and fix a cause that repeats once.

- Before you add a workaround, run `rg` for its rule, or `uv run python -m scripts.compile_ignores`, and look for the
  same rule with the same reason.
- If three or more sites share one cause, fix the cause: a shared helper, a shared import, a fixture, or one config
  entry with a `reason:` block that names the pattern. Do not add a third copy.
- A workaround for one tool must not appease another tool with a trick, such as a call that exists only so that vulture
  sees a caller. Use the tool's own mechanism (`ignore_names` for a hook that Python calls by name) or a better design.
  A chain of workarounds hides the first cause.
- The description of a pull request that adds an ignore states the new total from `compile_ignores`, and why no shared
  fix applies. A reviewer can then see the growth.

### Fix the cause by pattern

| Repeated ignore | Shared fix |
| --- | --- |
| `suspicious-subprocess-import`, `subprocess-without-shell-equals-true` in `scripts/` or tests | `run_fixed` in `scripts/fixed_command.py`. A test that builds a git repository takes the `git` and `make_repo` fixtures of `tests/unit/scripts/conftest.py`. |
| `implicit-reexport` of an `interpret.glassbox` class | Import it from `ebaml.core.ebm_classes`. |
| `banned-api` for pandas or joblib in a test | The aliases and helpers of `tests/unit/ebaml/conftest.py`: `FittedClassifier`, `frame_of_columns`, `frame_of_rows`, `categorical`, `save_model`. A new test imports neither library. |
| A vulture report of a hook that Python or pytest calls by name | `ignore_names` of `[tool.vulture]` in `pyproject.toml`. |

`src/ebaml/llm/sdk_setup.py` keeps its own `subprocess` ignore, because `src/ebaml` must not import from `scripts/`. A
need that the table does not cover is a new shared fix: build it once, then add a row.

### Scope: inline for a one-off, shared fix for a repeat

- A one-off gets an **inline** ignore (`# noqa: RULE`, `# pyrefly: ignore[...]`, ...) on the exact line, so each one is
  vetted at the place it applies and carries its own reason.
- The same ignore in three places is a recurring pattern. Apply the shared fix above, and do not add a third copy.
- Config-level ignores (`per-file-ignores`, `ignore`, `exclude`, `omit`, `ignore_names`, ... in `pyproject.toml` or
  `cosmic-ray.toml`) are only for a **recurring pattern** that would otherwise need the same inline ignore over and over
  (e.g. `assert` in `tests/**`). Never use them to silence a rule for one function or a few lines in a source file.
- Before you add a config-level ignore, check how many places it covers. If it covers only a few, go inline. If an
  existing config-level ignore covers only a few places, convert it to inline ignores and remove the config entry.
- The reason of a config-level ignore must name the recurring pattern it covers.

### Where the reason goes

The reason lives with the workaround, so it moves with the code and cannot go stale. There is no separate registry.

- **Python comments** end in ` -- <reason>`: `run(argv)  # noqa: S603 -- fixed argv, never untrusted input`.
- **pytest markers and calls** take `reason="..."`, or the same ` -- <reason>` comment on the line.
- **TOML settings** get a comment block directly above the key; its first line starts with `reason:`:

  ```toml
  # reason: recurring pattern in tests/: bare assert and literal expected values are how pytest tests are written.
  "tests/**/*.py" = ["assert", "magic-value-comparison"]
  ```

The reason must say why it cannot be fixed, not restate what the ignore does (at least 4 words and 20 characters). Never
invent a reason; ask the user if unsure. Remove a workaround whenever the tool passes without it.

Verify with `uv run python -m scripts.check_ignores` (also runs as the `check-ignores` hook). To audit all of them in one
place, run `uv run python -m scripts.compile_ignores` (Markdown to stdout, or `-o FILE`).

Do not edit `scripts/ignore_sites.py`, `scripts/check_ignores.py` or the `check-ignores` hook to make a violation pass.

Exempt: generated code in `src/baml_sdk/`, `.pyi` files and `baml_src/`.
