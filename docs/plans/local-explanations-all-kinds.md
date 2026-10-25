# Plan: local explanations of every model kind (1.1.0 and 1.2.0)

You are a senior engineer in Python and in formal proofs with Lean. You will implement one milestone of `ROADMAP.md`
in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, with special care for
`.claude/rules/proofs.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

<goal>
`explain_instance` gives the three local explanations (counterfactual, sufficient reason, count completions) for
regressors (1.1.0) and multiclass classifiers (1.2.0), with the same guarantees as for binary classifiers: Python
computes the facts, Lean proves each exact claim, and the LLM only puts the facts in words.
</goal>

<context>
- 1.1.0 and 1.2.0 come after the API freeze at 1.0.0. They must not break the public API. The `model-kinds.md` plan
  (0.2.0) made the result types general and wrote the definitions for each kind, which the user agreed to. Read those
  definitions first, and build on them. If they turn out wrong, stop and ask the user: a change of the types is now a
  breaking change.
- `src/ebaml/local/facts.py` holds the binary algorithms. Its module docstring explains them. The proofs live in
  `proofs/EBaMlProofs/LocalExplanations/` and in the paper `proofs/papers/local-explanations/paper.typ`.
- The model kind abstraction of 0.2.0 tells each kind how scores become a decision.
</context>

<design>
- Put the code of each kind behind one interface, with one class per kind (SOLID O). The binary class wraps the code
  that exists, with no change to its results. `compute_local_facts` selects the class by the model kind and never
  tests the kind in an `if` chain.
- Regressor (1.1.0): the decision is "the prediction is above the threshold" (or inside the interval) of the 0.2.0
  definition. With the identity link, the margin is a sum of one score per feature, so reuse the binary sorting
  algorithms on the shifted margin. Prove in Lean that the reduction is exact: state it as a theorem that maps the
  regression problem onto the binary model of the existing topic, so the existing proofs do the work.
- Multiclass (1.2.0): implement the definitions of 0.2.0.
  - Counterfactual "no longer class `c`": the minimum over the other classes of the binary problem on the margin
    `s_c - s_j`. Prove that the minimum of the per-class answers is the answer.
  - Sufficient reason: one linear condition for each other class. If 0.2.0 chose an exact search, prove it exact and
    state its cost. If it chose the weaker "minimal" claim, prove that, and use only the word that the proof supports.
  - Count completions: sample as for binary. The share is an estimate, so it makes no exact claim.
- Confidence: keep the rule of the module docstring. Each score moves by `z` standard deviations in the direction that
  makes the claim harder. For multiclass, define that direction for each margin, and write it in the paper.
- Text and prompts: extend `ebaml.local.fact_text` with the text for each kind, for both audiences. Change
  `baml_src/local_explanations.baml` only as much as the new texts need, and follow the prompt rules of
  `.claude/rules/writing-style.md`.
</design>

<tests>
- Unit tests for each kind against a brute-force search on small models: the counterfactual and the sufficient reason
  are the true minimum (or minimal, as claimed) for every instance of a small grid. Use Hypothesis for the property
  tests.
- The binary tests pass unchanged.
- Offline BAML tests for the new texts, and a live prompt test in `baml_src/prompt_tests.baml` for each new kind.
- `lake build` in `proofs/` with no `sorry`, and `Axioms.lean` shows only the expected axioms.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- Update `docs/local-explanations.md` with an example for each new kind, and the paper with the new theorems.
- Update the README table "Supported models": the local explanations row gets ✅ for the new kind.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Each milestone is a MINOR release (`.claude/rules/changelog-and-versioning.md`): regressors in one, multiclass
classifiers in the next. Ask the user before you start a proof that looks large, as the proofs rule says.
</release>

<done>
- `explain_instance` explains a prediction of a regressor (1.1.0) or of a multiclass classifier (1.2.0), and every
  exact claim in the docstrings and the docs has a Lean proof.
- `prek run --all-files`, `uv run pytest`, the offline BAML tests, mutation testing, the strict docs build and the proof
  build pass.
</done>
