# Plan: performance benchmarks and hotspots (0.14.0)

You are a senior Python performance engineer with experience in numerical code, profilers and benchmark design. You
will implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in
`.claude/rules/` first, with special care for `.claude/rules/semantic-contracts.md` and `.claude/rules/proofs.md`,
because the project's hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
Know where EBaMl spends its time and memory, keep that knowledge current with repeatable benchmarks, and remove the
hotspots that cost a user real time. A fix must make no explanation less correct: speed that changes a fact is a bug.
</goal>

<context>
- `benchmark/` is the TalkToEBM benchmark of LLM answer accuracy. It is not a performance benchmark. Give the new code a
  name that a reader cannot confuse with it, for example `performance/` and "performance benchmark", and use that name
  everywhere.
- The LLM request dominates the wall time of a call, and EBaMl does not control it. Measure the work around it: the
  graph extraction and simplification (`ebaml.graphs`), the graph text and the token budget, the facts of a local
  explanation (counterfactual, sufficient reason through OR-Tools, sampled completions in `ebaml.local`), the edits and
  their reports (`ebaml.edit`), the MCP tools, and the overhead of the BAML runtime with a local fake LLM server (the
  `fake_llm` fixture of `tests/conftest.py`).
- Two costs apply to every call and are easy to miss: the import time of `ebaml` and `baml_sdk`, and the run-time
  checks of beartype on every function of `ebaml` (`ebaml.core.beartype_hook`).
- The semantic contracts and the proofs cover the local explanation algorithms. A change of such an algorithm needs the
  oracle test of contract 1 and an update of the proof correspondence table.
- CI runners are noisy: the same job varies by 10 % or more. An absolute time threshold on a pull request fails at
  random.
</context>

<steps>
1. **Define the workloads.** Write a small set of fixed workloads that cover the user paths, each with a seed:
   - The Titanic classifier of the notebooks, a regressor and a multiclass classifier with five classes.
   - Synthetic EBMs that stress size: 10, 100 and 500 features, 32 and 1024 bins, and many categorical levels.
   - The calls: `describe_ebm` and `explain_instance` with the fake LLM, one edit with its report, and one call of each
     MCP tool.
   Keep the fitted models as `skops` files built by a script, so each run measures the same model.
2. **Profile before you measure.** Run each workload under a sampling profiler (py-spy or pyinstrument) and under a
   memory profiler (memray). Measure the import time with `python -X importtime -c "import ebaml"`. Measure the cost of
   beartype by a run with the claw hook off, in a scratch copy only. Save the flame graphs in the scratchpad and put the
   ten largest hotspots in a table: the function, its share of the time, and the workload.
3. **Add the benchmarks.** Add a performance benchmark with pytest-benchmark (or asv if you need a history across
   commits, and say why). Put it in its own directory with its own marker, outside the default `testpaths`, and
   register it in `testing.md` and `scripts/check_test_suites.py`. It never sends a real request.
4. **Run it where it is reliable.** Run the benchmarks on a weekly schedule and on demand. On a pull request, compare the
   base and the head in the same job on the same runner, and report the change in the CI comment. Gate only on a large
   change (for example 25 % slower on a median of several rounds), and calibrate the limit from the first ten runs. Do
   not gate on an absolute time.
5. **Choose the fixes.** For each hotspot, estimate the gain for a user and the risk to correctness. Fix only the ones
   where the gain is real for a normal workload. Show the list and your choice to the user before you change a
   proof-covered algorithm.
6. **Fix one hotspot per commit.** Write each fix as a `perf` commit with the numbers before and after in the body.
   Prefer, in order: work that is not needed (a repeated extraction, a cache that the caller can hold), a vectorized
   numpy form (`loops.md`), a better data structure, a lazy import, and last a different algorithm.
7. **Prove that the output is the same.** For each fix, add a differential test that runs the old and the new code on
   many inputs (hypothesis, or the workloads) and asserts equal results, or equal guarantees where a tie may change the
   set (contract 2). For a proof-covered algorithm, also run the oracle test of contract 1 and update the paper.
8. **Keep beartype.** If beartype is a hotspot, do not turn it off for the package. Find the hint that is slow to check
   (a deep `Sequence` or `Mapping` of a large array) and give it a cheaper exact hint, or ask the user.
9. **Write the results.** Add a page `docs/performance.md` with the workloads, the hotspot table, the fixes with their
   gains, and how to run the benchmarks. Add the run command to `CLAUDE.md` in one line.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.14.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment lists under `Changed` each fix that a user feels, with its gain on a named workload. Mark the 0.14.0 row of
`ROADMAP.md` as done. The fixes can come in earlier pull requests, each with its own fragment.
</release>

<done>
- The workloads, the profiles and the hotspot table are in `docs/performance.md`.
- The performance benchmark runs weekly and compares base and head on pull requests without random failures.
- Each fix has a differential test, and every semantic contract test and every proof stays true.
- `prek run --all-files`, `uv run pytest`, mutation testing of the changed lines and the strict docs build pass.
</done>
