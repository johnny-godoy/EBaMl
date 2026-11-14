# Plan: adversarial mutants (0.15.0)

You are a senior test engineer who thinks like a careless but competent contributor. Your job is to break EBaMl in
the ways a real person would, find each break that no test notices, and add the guard that would notice it. You will
implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/`
first, with special care for `.claude/rules/semantic-contracts.md`, `.claude/rules/bug-fixes.md` and
`.claude/rules/testing.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass
before you finish.

<goal>
Find the test gaps that matter before 1.0.0 freezes the API. cosmic-ray already requires zero survivors, but it makes
small syntactic changes one at a time: it flips an operator or a constant. A person makes other bugs: they swap two
arguments of the same type, use the class index of the wrong class, forget to pass an option through a wrapper, or
delete a sentence from a prompt. Those bugs pass every hook, and some pass every test. This milestone writes such
bugs on purpose, keeps each one as a permanent check, and closes each gap that a real contributor could open.
</goal>

<context>
- `cosmic-ray.toml` mutates `src/ebaml` only, against `tests/unit`. It does not mutate `baml_src/`, `scripts/`,
  `benchmark/` or the BAML prompts, and it never runs the integration, API or contract suites.
- `scripts/mutation.py` copies the project for each shard, so a mutant never touches the working tree. Reuse its copy
  logic.
- A recent commit (`test(integration): guard options that mutants of describe dropped`) shows the kind of gap that
  this plan looks for: a wrapper that dropped an option, which only an integration test could see.
- The semantic contracts name the properties that matter most: a counterfactual must flip the decision, a sufficient
  reason must keep it, the text must claim no more than the facts, and no model part may be dropped in silence.
- Oracles and contract tests need the approval of the user before they change (`semantic-contracts.md`, "Evidence").
</context>

<steps>
1. **Build the corpus.** Add a top-level `mutants/` directory. Each mutant is one small patch file with a header: the
   target file, the kind of bug, the plausibility (low, medium, high), the impact (wrong explanation, wrong number,
   crash, cosmetic), and the expected result (killed, or the guard that kills it). Add `scripts/check_mutants.py`, which
   applies each patch to a copy of the project, runs the suites that the header names, and fails if a mutant marked
   killed survives or a patch no longer applies. Run it weekly in CI and on demand.
2. **Set the bar for a mutant.** A mutant counts only if a real contributor could commit it: it compiles, passes ruff,
   pyrefly, beartype and every prek hook, and reads like code that someone meant to write. A mutant that a hook rejects
   already has a guard. Throw away equivalent mutants, and write down why each one is equivalent.
3. **Attack by layer, with one subagent for each.** Give each subagent one area, this plan and the list of bug kinds
   below, and ask it for at least 15 mutants of medium or high plausibility. Ask each to read the code and the tests of
   its area first, and to aim at the places where the tests look thin.
   - `ebaml.graphs`: bin edges off by one, a swapped pair of bounds, the term of the wrong feature, a lost
     categorical level, a token budget that counts the wrong text.
   - `ebaml.local`: the logit of the wrong class, a sign error in a margin, `<` for `<=` at a decision boundary, a
     completion sampled with the wrong bin weights, a fallback that reports a stronger guarantee than its search.
   - `ebaml.edit`: an edit that changes a copy and not the model, a history that loses an undo, a report computed on
     the model before the edit, a monotone constraint in the wrong direction.
   - `ebaml.llm`: an option that a wrapper drops (`model=`, `audience=`, `progress_bar=`), the precedence of `client`
     and `model` reversed, results of the thread pool in the wrong order, a provider error mapped to the wrong class, a
     retry that never retries.
   - `ebaml.mcp`: a tool that ignores an argument, a save without a confirmation, a schema that accepts a value the tool
     rejects.
   - `baml_src/`: a prompt that loses one fact, an instruction that claims a property (certainty, optimality) the facts
     do not show, a swapped turn in the three-turn chain, a client builder with the wrong provider row.
   - `scripts/` and the hooks: a guard that passes everything (for example a regex that matches nothing). A broken guard
     hides every later bug of its kind.
4. **Run and sort.** Run each mutant against every suite, not only the unit suite. Put each in one class:
   - Killed: keep it in the corpus as killed.
   - Survived, plausible and harmful: add a guard (step 5).
   - Survived, but cosmetic or very unlikely: keep it in the corpus as accepted, with the reason, and list it for the
     user.
5. **Add the guards.** For each harmful survivor, choose the guard by `bug-fixes.md`: a unit test for a wrong value, a
   semgrep rule for a pattern that can return anywhere, an integration test for a wrong crossing of the boundary, an
   offline BAML test for a prompt. Name the bug in the test name. Watch the guard fail on the mutant and pass on the
   real code, then mark the mutant killed by that guard. Ask the user before you change an oracle or a contract test.
6. **Widen the regular mutation testing.** If the round shows that a whole area escapes cosmic-ray, extend it: add
   `scripts/` or `benchmark/` to the mutated modules, or run the integration suite for the mutants of `ebaml.llm`.
   Measure the cost in CI time first, and put the change on the weekly workflow if it is too slow for a pull request.
7. **Fix real bugs.** A survivor can show that the code is already wrong. Fix each such bug in its own `fix` commit with
   its guard and a `Fixed` changelog entry.
8. **Report.** Write `mutants/README.md` with how to add and run a mutant, and a table of the round: the area, the
   mutants written, killed at once, killed by a new guard, and accepted. List the gaps that the round could not close and
   why.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.15.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment lists under `Fixed` each real bug that the round found. The corpus and the guards are contributor tooling and
need no entry. If the round finds no user-visible bug, ask the user whether the milestone still takes a version. Mark
the 0.15.0 row of `ROADMAP.md` as done.
</release>

<done>
- `mutants/` holds at least 100 plausible mutants over every area of step 3, and `scripts/check_mutants.py` runs them
  weekly in CI.
- Every harmful survivor has a guard that fails on it. Each accepted survivor has a reason that the user saw.
- Each real bug that the round found is fixed with a guard.
- `prek run --all-files`, `uv run pytest` and mutation testing pass.
</done>
