# Plan: verified kernels in Lean (0.18.0)

You are a senior engineer who writes Lean 4 proofs and ships native code in Python packages. You will implement one
milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, with
special care for `.claude/rules/proofs.md`, `.claude/rules/semantic-contracts.md` and
`.claude/rules/changelog-and-versioning.md`, because the project's hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
Run an algorithm of EBaMl as compiled Lean code, so that the code that runs is the code that the proof covers. Today
each Lean proof describes a model of the Python code (contract 1 of `semantic-contracts.md`), and a unit test guards
the link between the two. A kernel removes that gap for the part it covers: Python calls a function that Lean compiled,
and the theorems are about that same function. The first kernel is `EBMGraph.simplify`, because it is the largest
hotspot of the graph text that remains after the fixes of 0.16.0, and its claim ("no merged bin moves more than the
threshold") is exact. The milestone also builds the parts that every later kernel reuses: the Lean library, the C
binding, the wheels and the Python fallback.
</goal>

<context>
These facts were true when the plan was written. Check each one again before you act on it. The facts that nobody has
tested yet are marked "check in phase 0".

- The performance benchmark of 0.16.0 (`benchmark/performance/README.md`) measures `graphs.render.*`. After the fixes
  of the follow-up PR of 0.16.0, `EBMGraph.simplify` takes about half of the rendering time of a graph with 1024 bins.
  `to_text` tries up to 101 levels (`_SIMPLIFICATION_STEP = 0.001`, `_MAX_SIMPLIFICATION_STEPS = 100`), and each level
  runs `simplify` again on the full graph, in a Python loop over each bin.
- `simplify` is a greedy merge. The first bin opens a group. Each next bin joins the open group if no score of the bin
  differs from the score of the first bin of the group by more than `threshold`; else it opens a new group. The
  threshold is `(max(scores) - min(scores)) * min_variation_per_cent`. A merged bin keeps the score of its first bin,
  the interval from the low end of its first bin to the high end of its last bin, and the largest standard deviation of
  its bins. A multiclass graph has one score for each class, and a bin joins only if every class stays within the
  threshold.
- The impact and effort report of 0.16.0 (an or-solver run, not in the repository) rated a compiled `simplify` at
  about 95 % of the time of `simplify`, for 24 hours of work plus 30 hours for the build of the Lean library. With
  40 hours of budget it did not fit. With 80 hours it was the best choice for any value of a proof above zero. The
  user chose to plan it as a milestone of its own.
- The proofs live in `proofs/`: one Lake project, Lean and Mathlib pinned to `v4.35.0`, one topic for each claim
  (`LocalExplanations`, `MulticlassScores`, `Recourse`). The proofs never import the Python code, and CI runs
  `lake build` only when `proofs/` changes (`.github/workflows/proofs.yml`).
- Lean 4 compiles each definition to C. `@[export some_name]` gives a definition a fixed C symbol. Lake can build a
  library as a static or a shared object. A C caller must start the Lean runtime once
  (`lean_initialize_runtime_module`, then the `initialize_<Module>` function of the library, then
  `lean_io_mark_end_initialization`). Check in phase 0: the exact names and calls for `v4.35.0`.
- The core `Float` of Lean is an IEEE 754 double, as the `float` of Python. Its operations are opaque to the kernel of
  Lean (`@[extern]` functions), so a theorem cannot reason about the result of `a - b` on floats. Mathlib has no
  lemmas about `Float` either.
- The package is pure Python today: `uv_build` builds one `py3-none-any` wheel, with the generated `baml_sdk`. A native
  library needs one wheel for each platform.
- A ctypes call does not use the C API of CPython. So one compiled library serves every Python version, and the wheel
  can have the tag `py3-none-<platform>`. This matters for 0.17.0, which adds Python 3.12 to the matrix.
</context>

<decisions>
The plan fixes these points. Ask the user before you change one.

- The first kernel is `simplify`. A second kernel is out of scope for this milestone. Write it down as a candidate in
  the docs page instead (the sorting rule of `LocalExplanations/Greedy.lean` is the next candidate).
- The Python code of each kernel stays, as the fallback and as the reference of the differential tests. A user without
  the compiled library gets the same result, more slowly.

Ask the user about these points before phase 2, with the measurements of phase 0:

1. **Where the library ships.** The recommendation is a second package, `ebaml-kernels`, with one wheel for each
   platform, and an extra `ebaml[kernels]` that installs it. `ebaml` stays one pure wheel that installs everywhere.
   The other options: platform wheels of `ebaml` itself (every install needs a wheel of its platform), or a source
   build at install time (every install needs a C compiler and Lean).
2. **Which platforms.** The recommendation is Linux x86-64 and arm64 (manylinux), macOS arm64 and x86-64, and Windows
   x86-64. Each one is a CI job of `cibuildwheel`.
3. **Whether the kernel is on by default.** The recommendation is on when the library loads, with the environment
   variable `EBAML_KERNELS=0` to turn it off, and one structlog event at the first call that says which
   implementation runs.
</decisions>

<steps>
0. **Check that the route works (a spike, about 8 hours).** On a branch, write a Lean file with one exported function
   over a `FloatArray`, build it as a shared library with the runtime linked in statically, and call it from Python
   with ctypes on Linux, macOS and Windows (CI runners). Measure and write down:
   - the size of the library, with and without the static runtime;
   - the time of the start of the runtime, and of one call on a graph of 1024 bins and 5 classes;
   - whether two Lean libraries in one process conflict (a later kernel adds a second module);
   - whether the library loads in a thread other than the main thread, because `describe_ebm` renders graphs in a pool
     of threads, and whether each thread must call `lean_initialize_thread`.
   If a platform fails, stop and report to the user with the measurements. Do not work around a failure with a second
   build system.
1. **Write the kernel and its spec in Lean, apart from the proofs.** Add a Lake library `EBaMlKernels` in
   `proofs/` that imports only the core of Lean, never Mathlib, so that the compiled library stays small and its
   build needs no Mathlib cache.
   - Write the merge as a generic function over the test of closeness:
     `groupStarts (close : α → α → Bool) (rows : Array α) : Array Nat`, which returns the index of the first bin of
     each group. The proofs then hold for every `close`, and so for the one over floats, without any lemma about
     `Float`.
   - Write the float instance in one small definition: a row is a slice of a `FloatArray`, and `close first row` is
     true if for every class `Float.abs (first - row) ≤ threshold`. Compare in the order of the Python code, so that
     NaN gives the same answer. Check the rule for NaN against Python in a test, and write the rule in the paper.
   - Export one C function, for example
     `ebaml_group_starts(scores, bin_count, class_count, threshold, out_starts) -> group_count`, through a C shim that
     copies the input into a `FloatArray` and the result into a buffer of the caller. The Python side never touches a
     Lean object.
   - Export a second function that returns the group starts of all 101 levels in one call, if phase 0 shows that the
     cost of one call is large next to one level. `to_text` keeps its loop over the levels and its checks of the
     budget, so the chosen level stays the same.
2. **Prove the claims (topic `GraphSimplification`).** Add `proofs/EBaMlProofs/GraphSimplification/` that imports
   `EBaMlKernels` and Mathlib. Prove, for every `close` and every array of rows:
   - **Partition.** The starts begin with 0, increase strictly, and stay below the number of rows. So the groups are
     runs of neighbouring bins that cover every bin once.
   - **Closeness.** Every bin of a group is close to the first bin of its group.
   - **Greedy rule.** The first bin of each group except the first is not close to the first bin of the group before
     it. So a group never ends early.
   - **Uniqueness.** The three properties above fix the starts. So any other code with the same three properties gives
     the same groups, which is the claim that the differential test of step 4 checks.
   - **Monotone count (only if true).** Try to prove that a larger threshold never gives more groups when `close` comes
     from a threshold. This is not obvious for a greedy merge with an anchor: write a counterexample if it is false. If
     it is true, it allows a binary search over the levels in a later milestone. Do not change the search of `to_text`
     in this milestone, because a different level changes the text that the LLM reads.
   Use no `sorry`, `admit` or new `axiom`, and add `Axioms.lean` as the other topics do. Write the paper
   `proofs/papers/graph-simplification/paper.typ` from the template of `proofs.md`, with the table from each symbol to
   the name in the Lean kernel and in `ebm_graph.py`. Its limits name what is not proved: the arithmetic of floats, the
   C shim, ctypes, and the steps of `simplify` outside the groups (the merged intervals, the scores of the first bins
   and the largest standard deviations stay in Python).
3. **Bind it in Python.** Add a module in `ebaml.graphs` (for example `graphs/group_kernel.py`) that loads the library
   once, lazily, and gives one function `group_starts(scores, threshold) -> list[int]`. `simplify` calls it, and keeps
   its Python loop as the fallback when the library is missing, does not load, or `EBAML_KERNELS=0`. Keep `simplify`
   itself in Python, so that the public API and the beartype checks stay the same. The layer `graphs` may hold the
   binding, because ctypes is no I/O in the sense of `architecture.md`. Check the semgrep rule of I/O in domain code,
   and ask the user if it fails. Register the binding in `architecture/abstractions.yml` if a second kernel will reuse
   its loader.
4. **Prove that the output is the same.** Add a differential test with hypothesis: random scores (one class and
   several classes, ties, equal rows, large and tiny values, NaN only if the rule of step 1 covers it) and random
   thresholds, and assert that the kernel and the Python loop give the same starts. Run the graph tests of
   `tests/unit/ebaml/graphs/` once with the kernel and once with the fallback. Add a contract test that the exported
   symbol and the C signature in the binding agree.
5. **Build and ship the library.** Follow the decision of the user on the package. For the recommended second package:
   - Add `kernels/` with its `pyproject.toml` (a build backend that runs `lake build EBaMlKernels:shared` and the C
     shim, for example scikit-build-core with a small CMake file, or hatchling with a build hook), and a
     `cibuildwheel` workflow for the platforms of the decision.
   - Pin the version of `ebaml-kernels` to the version of `ebaml` in the extra, and check it in
     `scripts/release/`, as `toolchain_pin.py` checks the BAML toolchain.
   - Run `scripts/release/smoke_install.py` with `--extras kernels` on each platform, and check that the kernel runs and
     not the fallback.
   - Update the release workflow so that one tag publishes both packages.
6. **Measure.** Run `uv run python benchmark/run.py performance --workload "graphs.render.*" --workload "llm.*"` with
   the kernel and with `EBAML_KERNELS=0`, and compare with `benchmark/run.py performance compare`. Put the table in the
   commit of the kernel and in `benchmark/performance/README.md`. If the gain on `graphs.render.features-100-bins-1024`
   is below 30 %, report it to the user before you ship.
7. **Write the docs.** Add `docs/graph-simplification-proof.md` with the parts that `proofs.md` lists and the iframe of
   the paper, and a page or a section that says how to install the kernels, how to turn them off, and which platforms
   have them. Point the docstring of `simplify` to the paper in one sentence. Add the topic to `proofs/README.md`.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.18.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment says under `Added` that the extra `kernels` runs `simplify` as compiled code that a Lean proof covers, with the
gain on a named workload. Mark the 0.18.0 row of `ROADMAP.md` as done. Phase 0 and the Lean kernel with its proofs can
come in earlier pull requests, each without a version change.
</release>

<done>
- `simplify` runs the compiled Lean kernel when the library loads, and the Python loop otherwise, with the same groups,
  and a differential test shows it.
- The topic `GraphSimplification` proves the partition, the closeness, the greedy rule and the uniqueness of the groups
  of the kernel, with only the three standard axioms. The paper and its docs page exist, and the paper names the
  arithmetic of floats as outside the proof.
- The wheels of the kernels install and run on every platform of the decision, and `ebaml` alone still installs and
  passes its tests without them.
- `tests/api/public_api.json` changes only by the extra and any new public name that the user approved.
- `prek run --all-files`, `uv run pytest`, mutation testing of the changed lines, `cd proofs && lake build` and the
  strict docs build pass.
</done>
