# Development

This page is for contributors. [CONTRIBUTING.md](https://github.com/johnny-godoy/EBaMl/blob/main/CONTRIBUTING.md) has the steps
of a change, and [Releasing](releasing.md) explains the release workflow.

## How it works

| Where | What |
| --- | --- |
| `src/ebaml/core/` | Shared kernel: types and errors. It imports no other part of `ebaml`. |
| `src/ebaml/graphs/` | Graph extraction and simplification (numpy, `interpret`), and the text the LLM reads. |
| `src/ebaml/llm/` | The edge: the calls into BAML (`descriptors.py`), their options and the retry after a cut-off reply. |
| `baml_src/talk_to_ebm.baml` | Prompts, the default LLM clients, the three-turn description chain, the summary, `describe_ebm`, token accounting. |
| `baml_src/model_spec.baml` | The model of a call: `ModelSpec` and one client builder for each provider. |
| `src/ebaml/local/`, `baml_src/local_explanations.baml` | Local explanations: the facts of one prediction computed deterministically, their rendering as JSON texts, and the prompts that put them in words. |
| `baml_src/tests/` | The BAML tests: offline tests on a scripted client, one `<source>_test.baml` for each source file, and the live prompt tests in `live/`. |
| `src/ebaml/edit/` | Model editing: the selections of bins, the edits, the report, and the session with its history. It calls no LLM. |
| `src/ebaml/mcp/`, `baml_src/mcp_agent.baml` | The MCP server (`ebaml-mcp`), the agent that reads it through its tools, and the terminal chat (`ebaml-chat`). |
| `baml.toml` | Generator config for the Python SDK. |
| `src/ebaml/benchmark.py`, `benchmark/` | The benchmark: loading, answer parsing and scoring, and the data and runner (see [Benchmark](benchmark.md)). |
| `notebooks/` | Example notebooks, written with marimo. |
| `docs/`, `zensical.toml` | The documentation, built with Zensical (see [the commands](#commands) below). |
| `tests/unit/` | Python unit tests (100% coverage required). `scripts/llm/openrouter.py` is the optional live test. |
| `scripts/` | Maintenance scripts, see [scripts/README.md](https://github.com/johnny-godoy/EBaMl/blob/main/scripts/README.md). |
| `.github/` | CI workflow and the scripts that turn failures into reports. |
| `src/baml_sdk/` | Generated. Do not edit. |

## Commands

```bash
baml fmt          # format .baml files
baml check        # compile check
baml test -x root::Live   # run the offline BAML tests
baml generate     # regenerate src/baml_sdk/
uv run pytest     # Python unit tests, fails under 100% coverage
uv run pytest README.md docs src --no-cov   # the code examples of the docs and the docstrings
prek run --all-files                       # every hook: formatters, ruff, vulture, pyrefly, baml fmt, ...
uv run python scripts/mutation/run.py          # mutation testing of all modules, fails if any mutant survives
uv run python scripts/mutation/run.py --base origin/main   # the same, but only for the lines changed since main
uv run python scripts/mutation/run.py --module src/ebaml/local/facts.py   # the same, but only for one module
uv run python -m scripts.ignores.check     # every lint/type/test workaround needs a reason written next to it
uv run python -m scripts.ignores.report   # all workarounds and their reasons as one Markdown report to audit
uv run --group docs python -m scripts.docs.build         # build the docs; fails on a broken link or a docstring warning
uv run --group docs python -m scripts.docs.build serve   # preview the docs (exports the notebooks, so the first run is slow)
```

CI (`.github/workflows/ci.yml`) runs prek, secret and security scans (trufflehog, semgrep, trivy), `baml check` and
`baml test`, pytest with 100% coverage, the code examples of the documentation, and, in pull requests, mutation testing of the changed lines with a 100% score. A weekly run tests all modules, in jobs of at most 500 lines each. Lint and type workarounds
(`# noqa`, `# type: ignore`, skipped tests, ...) must carry a written reason next to them.

The `python` code blocks of `README.md` and `docs/*.md` run in order, and the blocks of one file share their variables.
The `Examples` sections of the docstrings run as doctests. `tests/docs/conftest.py` gives each Markdown file a small
data set as `X` and `y`, and it replaces the LLM calls with fakes, so the examples send no request. Put setup that the
reader does not need in an `<!-- invisible-code-block: python -->` comment. To skip a block, put
`<!-- skip: next -->` above it, and write the reason in the text.
