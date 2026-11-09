# Plan: an independent oracle of the proved model

You are a senior engineer who writes property-based tests for numerical algorithms. You will add an oracle that
computes the local explanations straight from the definitions of the proof paper, and you will compare it with
`compute_local_facts` on random small models. Read `CLAUDE.md` and every file in `.claude/rules/` first, especially
`semantic-contracts.md` (contract 1) and `proofs.md`, because the hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
A test fails when `src/ebaml/local/facts.py` stops computing what `proofs/papers/local-explanations/paper.typ`
proves, for every value of `z` and of the plausible bins that the paper covers.
</goal>

<context>
Found on 2026-11-09:

- Brute-force tests exist in `tests/unit/ebaml/local/test_facts.py`: `test_matches_brute_force` (a binary
  categorical model, 27 rows), `test_multiclass_facts_are_the_smallest_by_brute_force`,
  `test_a_target_counterfactual_is_the_smallest_over_every_choice_of_bins`, `test_regressor_threshold_matches_brute_force`
  and `test_regressor_interval_matches_brute_force`. They enumerate with `ebm.decision_function`, so they do not
  depend on the code under test.
- `test_matches_brute_force` runs with `confidence_level=0` and `min_support=0`. Check the options of the others. With
  `z = 0` and no trimmed bins, the tests do not see the pessimistic score `p_i`, the guaranteed score `g_i` or the
  plausible set `P_i` of the paper. The paper proves the algorithms for every `z >= 0` and every `P_i`.
- The tests use a few fitted models. Hypothesis strategies for covering problems exist in
  `tests/unit/ebaml/local/conftest.py`, and `fake_model` is in `tests/unit/ebaml/conftest.py`.
- CI runs mutation testing on the changed lines and requires zero survivors (`scripts/mutation.py`). A change to a
  covered line that no test notices fails there. So the review's "trigger the tests when the algorithm changes" is
  already in place. Do not add a second trigger.
- The repository has no `.github/CODEOWNERS`, and `.github/rulesets/main.json` has `require_code_owner_review: false`.
</context>

<steps>
1. Write the oracle as plain functions that read the inputs of the paper: the signed scores `t_i`, the deviations
   `sigma_i`, the plausible sets `P_i`, the intercept and `z`. Implement the definitions of the paper by enumeration:
   the robust margin `R(S, w)` of a change, the smallest flipping change, and the smallest set whose worst completion
   over `P_i` with `g_i` keeps the margin above zero. Use the names of the correspondence table of the paper. Import
   nothing from `ebaml.local.facts`.
2. Find where a test helper may live. `scripts/check_layout.py` rejects a test file with no mirrored source. A
   `conftest.py` is allowed. Read the check before you choose.
3. Write a Hypothesis strategy of small models: 2 to 4 features, 2 to 4 bins, deviations at or above 0, positive
   weights, and `confidence_level` and `min_support` drawn from a few values that include 0. Build the EBM from it, by
   fitting once and setting `term_scores_`, `standard_deviations_` and `bin_weights_`, or by `fake_model`. Check that
   `ebm.decision_function` agrees with the scores that you set.
4. Compare, for each model kind (binary, multiclass with 3 classes, regressor with `Threshold` and with `Interval`):
   the existence and the size of the counterfactual and of the sufficient reason, the guaranteed margin, and
   `minimum`. A tie can give two valid sets, so compare the sizes and check that the returned set is valid in the
   oracle. Do not compare the sets.
5. Keep the run time low: a few hundred examples, `deadline=None`, and the Hypothesis profile that the repository uses
   in CI.
6. Ask the user whether to protect the oracle and the proofs with a `CODEOWNERS` entry and
   `require_code_owner_review`. That is a change of the repository policy, and `scripts/check_ruleset.py` checks the
   ruleset. Do it only if the user agrees.
</steps>

<constraints>
- The oracle follows the paper, not the code. If they differ, report the difference to the user as an open question
  (`proofs.md`, step 1). Do not change the oracle to match the code.
- A failing example is a finding. Keep it as an `@example` in the test after the fix.
</constraints>

<release>
Tests only: no changelog fragment. If a bug is found and fixed, the fix gets a `Fixed` fragment.
</release>

<done>
- The oracle covers `z > 0` and `min_support > 0` for every model kind that the paper covers.
- Replacing `towards_outcome=False` with `True` in `_counterfactual`, or skipping `plausible` in `_robust_scores`,
  makes the test fail. Check both by hand, then revert.
- `prek run --all-files` and `uv run pytest` pass.
</done>
