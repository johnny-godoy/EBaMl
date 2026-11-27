# Benchmarks

`benchmark/` holds every measurement of EBaMl. Each kind of benchmark is one subpackage and one subcommand of
`benchmark/run.py`:

| Kind | Command | It measures | Real requests |
| --- | --- | --- | --- |
| Accuracy | `uv run python benchmark/run.py accuracy` | How well an LLM reads the graphs of an EBM, with the tasks of TalkToEBM. | Yes |
| Performance | `uv run python benchmark/run.py performance` | The time of the work of EBaMl around the LLM requests. | No |

Run `uv run python benchmark/run.py KIND --help` to see the flags of a kind. The page of each kind gives the details:
`benchmark/accuracy/README.md` and `benchmark/performance/README.md`.

## Add a kind

1. Make a subpackage `benchmark/<kind>/` with an `__init__.py` that has a docstring.
2. Put a cyclopts app in `benchmark/<kind>/command.py`. Create it with `result_action="return_value"`, so that a run
   returns its exit code.
3. Add a row to `KINDS` in `benchmark/run.py`, with the import path of the app (`benchmark.<kind>.command:app`).
   Cyclopts imports a kind only when the command runs it.
4. Add the flags of the kind to `BENCHMARK_FLAGS` in `tests/api/test_command_lines.py`.
5. Write `benchmark/<kind>/README.md`, add a page `docs/benchmark-<kind>.md` that includes it, and add the page to the
   nav in `zensical.toml`.

A kind does not edit the modules of another kind.
