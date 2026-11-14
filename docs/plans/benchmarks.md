# Plan: one benchmark directory for accuracy and performance (0.14.0)

You are a senior Python performance engineer with experience in numerical code, profilers and benchmark design. You
will implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in
`.claude/rules/` first, with special care for `.claude/rules/semantic-contracts.md`, `.claude/rules/solid.md` and
`.claude/rules/proofs.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

<goal>
`benchmark/` becomes the one home of every measurement of EBaMl. It measures the accuracy of the LLM answers (the
TalkToEBM benchmark that is there today) and the performance of the system (time, memory and import cost), and it
accepts a new kind of benchmark later (for example token cost or the quality of local explanations) by adding one
module, with no edit of the others. With the performance benchmark in place, find where EBaMl spends its time and
memory, and remove the hotspots that cost a user real time. A fix must make no explanation less correct: speed that
changes a fact is a bug.
</goal>

<context>
- `benchmark/` holds `run.py` (a cyclopts command with `--task`, `--limit`, `--model`, `--output` and `--progress`), the
  task files in `data/` and a `README.md`, which `docs/benchmark.md` includes whole. `src/ebaml/benchmark.py` holds the
  task logic (`Task`, `load_questions`, `run_task`, `is_correct`, `format_results`). It stands alone and imports nothing
  from `ebaml` (`.claude/rules/architecture.md`).
- `CLAUDE.md`, `benchmark/README.md`, `docs/benchmark.md`, `tests/api/test_command_lines.py` and the CI workflows name
  the current command `uv run python benchmark/run.py --task ...`. A change of the command is a change of a CLI, which is
  public API (`changelog-and-versioning.md`).
- The LLM request dominates the wall time of a call, and EBaMl does not control it. The performance benchmark measures
  the work around it: the graph extraction and simplification (`ebaml.graphs`), the graph text and the token budget, the
  facts of a local explanation (counterfactual, sufficient reason through OR-Tools, sampled completions in
  `ebaml.local`), the edits and their reports (`ebaml.edit`), the MCP tools, and the overhead of the BAML runtime with a
  local fake LLM server (the `fake_llm` fixture of `tests/conftest.py`).
- Two costs apply to every call and are easy to miss: the import time of `ebaml` and `baml_sdk`, and the run-time checks
  of beartype on every function of `ebaml` (`ebaml.core.beartype_hook`).
- The semantic contracts and the proofs cover the local explanation algorithms. A change of such an algorithm needs the
  oracle test of contract 1 and an update of the proof correspondence table.
- CI runners are noisy: the same job varies by 10 % or more. An absolute time threshold on a pull request fails at
  random.
</context>

<steps>
1. **Restructure `benchmark/`.** Give each kind of benchmark its own subdirectory, and one command with one subcommand
   for each kind. Propose the layout to the user before you move files, for example:
   - `benchmark/run.py`: the cyclopts app. It finds the kinds in one table (`solid.md`, O), so a new kind adds a row and
     a module.
   - `benchmark/accuracy/`: the TalkToEBM tasks and their `data/`. `uv run python benchmark/run.py accuracy --task jumps`
     replaces the current command.
   - `benchmark/performance/`: the workloads and the timing code.
   - `benchmark/README.md`: one section for each kind, and the steps to add a kind.
   Keep `src/ebaml/benchmark.py` for the accuracy logic that users import. The performance code measures `ebaml`, so it
   imports `ebaml` and lives in `benchmark/performance/`, not in `src/ebaml/benchmark.py`, which must stay alone. Update
   every place that names the command, and the row in `tests/api/test_command_lines.py`.
2. **Define the workloads.** Write a small set of fixed workloads that cover the user paths, each with a seed:
   - The Titanic classifier of the notebooks, a regressor and a multiclass classifier with five classes.
   - Synthetic EBMs that stress size: 10, 100 and 500 features, 32 and 1024 bins, and many categorical levels.
   - The calls: `describe_ebm` and `explain_instance` with the fake LLM, one edit with its report, and one call of each
     MCP tool.
   Build the fitted models with a script and save them with `skops`, so each run measures the same model.
3. **Profile before you measure.** Run each workload under a sampling profiler (py-spy or pyinstrument) and under a
   memory profiler (memray). Measure the import time with `python -X importtime -c "import ebaml"`. Measure the cost of
   beartype by a run with the claw hook off, in a scratch copy only. Save the flame graphs in the scratchpad and put the
   ten largest hotspots in a table: the function, its share of the time, and the workload.
4. **Write the performance benchmark.** Use pytest-benchmark, or asv if you need a history across commits, and say
   why. `benchmark/run.py performance` runs it and writes a JSON file with the same shape for every run, so two runs
   compare. It never sends a real request. If its tests live under `tests/`, register the suite in `testing.md` and
   `scripts/check_test_suites.py`, and keep it out of the default `testpaths`.
5. **Run it where it is reliable.** Run the performance benchmark on a weekly schedule and on demand. On a pull request,
   compare the base and the head in the same job on the same runner, and report the change in the CI comment. Gate only
   on a large change (for example 25 % slower on a median of several rounds), and calibrate the limit from the first ten
   runs. Do not gate on an absolute time.
6. **Choose the fixes.** For each hotspot, estimate the gain for a user and the risk to correctness. Fix only the ones
   where the gain is real for a normal workload. Show the list and your choice to the user before you change a
   proof-covered algorithm.
7. **Fix one hotspot per commit.** Write each fix as a `perf` commit with the numbers before and after in the body.
   Prefer, in order: work that is not needed (a repeated extraction, a cache that the caller can hold), a vectorized
   numpy form (`loops.md`), a better data structure, a lazy import, and last a different algorithm.
8. **Prove that the output is the same.** For each fix, add a differential test that runs the old and the new code on
   many inputs (hypothesis, or the workloads) and asserts equal results, or equal guarantees where a tie may change the
   set (contract 2). For a proof-covered algorithm, also run the oracle test of contract 1 and update the paper.
9. **Keep beartype.** If beartype is a hotspot, do not turn it off for the package. Find the hint that is slow to check
   (a deep `Sequence` or `Mapping` of a large array) and give it a cheaper exact hint, or ask the user.
10. **Write the results.** Make `docs/benchmark.md` an overview with one page for each kind: the accuracy page that
    exists, and a performance page with the workloads, the hotspot table, the fixes with their gains, and how to run
    them. Update the commands in `CLAUDE.md`.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.14.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment marks the new benchmark command as **BREAKING** under `Changed`, and lists each performance fix that a user
feels, with its gain on a named workload. Mark the 0.14.0 row of `ROADMAP.md` as done. The restructure and the fixes
can come in earlier pull requests, each with its own fragment.
</release>

<done>
- `benchmark/` holds the accuracy and the performance benchmarks behind one command, and the README says how to add a
  kind.
- The workloads, the profiles and the hotspot table are in the docs.
- The performance benchmark runs weekly and compares base and head on pull requests without random failures.
- Each fix has a differential test, and every semantic contract test and every proof stays true.
- `prek run --all-files`, `uv run pytest`, mutation testing of the changed lines and the strict docs build pass.
</done>
