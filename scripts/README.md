# scripts

Repository maintenance scripts. Run them from the repo root with `uv run python scripts/<name>.py`.

| Script | What it does |
| --- | --- |
| `check_ignores.py` | Fails unless every lint/type/coverage/test workaround has a written reason next to it. |
| `compile_ignores.py` | Compiles every workaround and its reason into one Markdown report to audit. |
| `mutation.py` | Runs cosmic-ray mutation testing in parallel shards, for all or only changed modules, and fails if any mutant survives. |
| `build_docs.py` | Exports the marimo notebooks, builds or serves the docs with Zensical (`zensical.toml`), and fails on a docstring warning. |
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

## `check_exclusions.py`

`exclude-newer-package` in `pyproject.toml` lets one package skip the 7-day cutoff of uv, for example to take a security
fix at once. Such an entry is temporary. `check_exclusions` reads `uv.lock` and fails when the locked version of an
exempt package is older than the cutoff, or when the package is not locked. If it fails, delete the entry and run
`uv lock`.

```bash
uv run python -m scripts.check_exclusions   # fails if an exemption is no longer needed
```

It also runs as the `check-exclusions` hook in `prek.toml`.

## `build_docs.py`

`build` is the default command. It exports the notebooks into `docs/notebooks/` and runs `zensical build --strict`.
It fails if the output has a `griffe:` warning. CI runs the same command. `serve` exports the notebooks and starts the
preview server. The module docstring explains why the export goes into `docs/`.

```bash
uv run --group docs python -m scripts.build_docs [build|serve]
```

## `mutation.py`

cosmic-ray mutates the source tree in place, so each shard works on its own copy of `src`, `tests`, `pyproject.toml`
and `cosmic-ray.toml`. The script:

1. creates a session in `.mutation/session.sqlite` and drops mutants on `# pragma: no mutate` lines,
2. if you give `--base` or `--module`, drops the mutants of all other modules,
3. runs the test baseline and stops if it fails,
4. splits the work items of each module over the shards and runs them concurrently,
5. prints a progress line every minute, with the finished, surviving and remaining mutants,
6. merges the results back and prints the survival rate.

Two measures keep each mutant fast. Each shard runs the tests of the mutated module first, because they kill most of
its mutants (`tests/unit/conftest.py` reads `MUTATION_FIRST`). All shards share one cache of fitted EBMs, which the
baseline fills (`tests/unit/ebaml/conftest.py` reads `EBAML_FIT_CACHE`). Without these variables, the tests run in
the normal order and fit every model.

```bash
uv run python scripts/mutation.py      # one shard per CPU
uv run python scripts/mutation.py 4    # four shards
uv run python scripts/mutation.py --base origin/main   # only modules changed since main
uv run python scripts/mutation.py --module src/ebaml/local/facts.py   # only one module
```

If no module changed, the script exits with code 0 and runs nothing. The PR run in CI uses `--base`. The workflow
`mutation-weekly.yml` runs all modules each Monday, and only if code changed in the last 7 days. It starts one job for
each module with `--module`, and opens one issue that lists the surviving mutants of all jobs.

Exits non-zero if any mutant survives. CI requires a 100% mutation score. Settings (module path, timeout, test command)
are in `cosmic-ray.toml`.
