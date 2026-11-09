# Plan: the facts agree with the fitted model

You are a senior engineer who writes property-based tests. You will check that the local facts agree with
`ebm.predict` and `ebm.decision_function` on random small models of every kind. Read `CLAUDE.md` and every file in
`.claude/rules/` first, especially `semantic-contracts.md` (contract 5), because the hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
A test on random models fails when a reported class, score, bin or feature value of `compute_local_facts` disagrees
with the fitted EBM.
</goal>

<context>
`proof-oracles.md` checks the code against the model of the paper. This plan checks it against the real EBM through
the API of `interpret`, so it also catches an error of reading the model (a wrong term, bin or category). Found on
2026-11-09, in `tests/unit/ebaml/local/test_facts.py`:

- `test_prediction_matches_model` checks the score, the interval and the class of one row of a binary model.
- `test_counterfactual_really_flips` and `test_the_changes_of_a_counterfactual_flip_the_prediction` apply the changes
  and call `ebm.predict`. `test_sufficient_reason_decides_outcome` mixes 50 random training rows into the free
  features, at `confidence_level=0`.
- Multiclass and regressor tests check the class and the predicted value.
- `test_counterfactual_is_none_for_a_tie` covers one tie.

Gaps:

- The models are a few fixed fits. No test draws random models.
- No test checks that the intercept plus the contributions equals `decision_function` for every class.
- The tie rule: a binary classifier predicts the second class iff the score is above 0 (`local_kinds.py`). The
  counterfactual flips at a robust margin below 0, and the sufficient reason needs a guaranteed margin above 0
  (`facts.py`). At an exact tie, both are conservative: they never claim a flip or a guarantee that the model does not
  give. No test states this.
- `Change.to_value` is a bin label, such as `[1.8e+04, 2.4e+04)`. To apply a change of a continuous feature, a test
  must pick a value inside the bin.
</context>

<steps>
1. Reuse the strategy of small models from `proof-oracles.md`, or write it there first.
2. For each model kind, and for random instances, assert:
   - `prediction.decision` equals `ebm.predict`, and `prediction.score` equals `decision_function` for the column of
     the decision;
   - the intercept plus the contributions equals the score, within a floating-point tolerance;
   - after the changes of the counterfactual, with a value inside each new bin, `ebm.predict` gives
     `new_decision`, and the probability or value equals `new_output`;
   - with the features of the sufficient reason fixed, each completion over the plausible bins keeps the decision
     (enumerate when the space is small, else sample);
   - two calls with the same model, instance and options give equal facts, with the default `MipCover`.
3. Add tie tests: build a model where a change reaches a score of exactly 0, and assert that no fact claims more than
   the tie rule gives.
4. Add alignment tests for a categorical feature whose categories are not in sorted order, and for a feature with a
   missing value in training. Check that each name, bin and category of the facts matches the instance.
</steps>

<constraints>
- A failing example is a finding. Fix the code with a guard (`bug-fixes.md`), or report it to the user if the fix
  changes a claim. Keep the example as an `@example`.
</constraints>

<release>
Tests only: no changelog fragment, unless a bug fix follows.
</release>

<done>
- The properties above run for binary, multiclass and regressor models.
- `prek run --all-files` and `uv run pytest` pass.
</done>
