# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<project>
EBaMl is BAML for Explainable Boosting Machines. It is a port and extension of TalkToEBM. Python extracts graphs from an
EBM (`interpret`). BAML holds the prompts, the LLM clients and the chain-of-thought pipelines. The default LLM is
OpenRouter's `openrouter/free`, which needs `OPENROUTER_API_KEY`.
</project>

<commands>
Requirements: Python 3.13 or later, `uv`, and the BAML toolchain 0.22 or later. Put `~/.baml/bin` first on `PATH`,
because older `baml` versions lack `describe`.

```bash
uv sync
baml generate                      # run after every .baml change; writes src/baml_sdk/ (gitignored, generated)
baml fmt && baml check             # format and compile-check .baml files
OPENROUTER_API_KEY=unused baml test -x root::Live   # offline BAML tests on a scripted client; the variable must exist
baml test -i root::Live            # live prompt tests (baml_src/prompt_tests.baml); send real requests, need a real key
uv run pytest                      # unit tests in tests/unit; fails below 100% coverage
uv run pytest tests/unit/ebaml/test_x.py::test_name --no-cov   # one test, without the coverage gate
prek run --all-files               # all hooks: ruff, vulture, pyrefly, baml fmt, check-ignores, uv lock --check
uv run python scripts/mutation.py [--base origin/main | --module src/ebaml/x.py]  # cosmic-ray; mutates only the changed or named modules; CI requires zero survivors
uv run python scripts/live_openrouter.py   # optional live test; sends real requests
uv run python benchmark/run.py --task monotonicity --limit 10 --output results.json   # sends real requests
uv run --group docs mkdocs serve   # preview the docs; the first build runs the notebook and is slow
```

`ModuleNotFoundError: baml_sdk` means `baml generate` has not run. CI runs prek and the secret scans first, then
`baml` and pytest, then mutation testing of the changed modules. A weekly workflow runs all modules, one job each, when code changed that week.
</commands>

<architecture>
The boundary between the two languages is the generated `baml_sdk` package.

- `baml_src/talk_to_ebm.baml` holds the prompts, the LLM clients, the three-turn graph description chain, the summary,
  `describe_ebm` and the token accounting. The clients are `RouterBase`, `RouterLongBase` and `RouterBenchmarkBase`.
  `ai.clients.Retry` wraps each one (`Router`, `RouterLong`, `RouterBenchmark`). To change the provider, edit only the
  `*Base` lines and run `baml generate`. The Python code stays the same. Every BAML entry point accepts an optional
  `client: ai.Client`, which the offline tests use. The Python wrappers do not expose it.
- `baml_src/local_explanations.baml` and `src/ebaml/local.py` produce local (per-prediction) explanations. Python
  computes the facts exactly from EBM scores and confidence intervals (counterfactual, sufficient reason, completions;
  arXiv:2010.12265). The LLM only puts them in words. Only binary classifiers without interactions work.
- `src/ebaml/graphs.py` and `graph_text.py` extract and simplify an `EBMGraph` and convert it to and from text. The
  token budget is an estimate of about 3 characters per token. `descriptors.py` connects this code to `baml_sdk`
  (`describe_graph`, `describe_ebm` with a thread pool, `explain_instance`).
- `src/ebaml/benchmark.py` and `benchmark/` are the port of the TalkToEBM benchmark. `is_correct` parses and compares
  values.
- `baml_src/talk_to_ebm_test.baml` holds the offline BAML tests. `notebooks/` uses marimo. `docs/` uses MkDocs and
  includes README sections through `--8<--` markers, so keep the markers intact.
</architecture>

<conventions>
Rules in `.claude/rules/` load automatically. Follow them, and keep these points in mind:

- Run `prek run --all-files` before you report a task as done. The hooks enforce the project's quality bar.
- Use `structlog`, `orjson` and `attrs`. The project replaces `logging`, `json` and `dataclasses` with them.
- `beartype` checks the type hints of `ebaml` at run time, so write exact hints (`.claude/rules/types.md`).
- Add a `CHANGELOG.md` entry under `[Unreleased]` for each user-visible change.
- Write a reason next to each `noqa`, `type: ignore`, pragma, skipped test and config ignore (` -- reason`, or a
  `# reason:` block in TOML). Check with `uv run python -m scripts.check_ignores`. Fix the code or ask the user when a
  hook fails. Leave `scripts/check_ignores.py`, `scripts/ignore_sites.py` and the hook unchanged.
- Write documentation in ASD-STE100 style, and write LLM prompts to Anthropic's prompting guide. See
  `.claude/rules/writing-style.md`.
- Ruff selects all rules with line length 120. Tests live in `tests/unit/ebaml/`.
- Leave `src/baml_sdk/` and `.pyi` files unedited. Regenerate them with `baml generate`.
</conventions>
