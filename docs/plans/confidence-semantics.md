# Plan: one stated uncertainty model for each claim

You are a senior engineer with a background in statistics. You will make each confidence claim of the local
explanations name its uncertainty model, and add tests that fail when a change makes a claim less conservative. Read
`CLAUDE.md` and every file in `.claude/rules/` first, especially `semantic-contracts.md` (contract 7) and `proofs.md`,
because the hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
The code, the docstrings, the docs and the paper describe the same uncertainty model for each claim. A test fails
when more uncertainty turns a failed claim into a valid one.
</goal>

<context>
The local explanations use three uncertainty models. Each one is correct for its claim, but the text does not always
name the right one. Found on 2026-11-09:

| Claim | What moves by `z` deviations | Where |
| --- | --- | --- |
| Counterfactual | Only the score of the new bin of each changed feature. The other features keep their nominal score. | `_counterfactual`, `_alternatives` in `src/ebaml/local/facts.py` |
| Sufficient reason | Each score, against the decision: the fixed features at their bin, the free features at their worst plausible bin. | `_sufficient_reason`, `_gain_table` |
| Probability interval | The total score, with the variance of a sum of independent errors. | `score_stds` in `compute_local_facts` |
| Multiclass margin | The difference of two class scores, with the sum of the two deviations, because EBMs store no covariance. | `_Feature.stds` |

- The module docstring of `facts.py` says: "every score is moved by `z` standard deviations in the direction that
  makes the claim harder to prove". That is false for the counterfactual. The paper says so in its section "change
  and robust margin" and discusses it in its limits section, but the docstring is unchanged.
- `docs/local-explanations.md`, section "Confidence", is correct for the counterfactual ("the effects of the changed
  features").
- The prompt sentence about the flip names "the model's effects". The plan `faithful-prompts.md` fixes the prompts.
  Do that plan first, or leave the prompts to it.
- `test_confidence_makes_claims_harder` in `tests/unit/ebaml/local/test_facts.py` checks one row of one binary model
  at two levels, and only the sizes.
</context>

<steps>
1. Fix the module docstring of `facts.py`: say what moves for each claim, as in the table above. Then update the
   correspondence table and the limits section of the paper if they quote the docstring, and the commit in its
   appendix (`proofs.md`, "Keep the proof true").
2. Add a Hypothesis test of monotonicity in `z`, on random small models of every kind (binary, multiclass, regressor
   with `Threshold` and `Interval`). For two levels `low < high`:
   - if a sufficient reason exists at `high`, one exists at `low`, and its size is not larger;
   - the guaranteed margin of the sorting result does not grow with `z`;
   - if a counterfactual exists at `high`, one exists at `low`, and its size is not larger;
   - the probability interval at `high` contains the interval at `low`.
   Restrict the size checks to results with `minimum` true. A heuristic solver (`GreedyCover`, `LocalSearchCover`)
   has no such property, so it is out of scope. Reuse the strategy of small models from `proof-oracles.md` if it
   exists.
3. Add boundary tests: a margin that is exactly zero, a deviation of zero, and `confidence_level` close to 1. Assert
   that a claim never holds on the side that the tie rule of `local_kinds.py` gives to the other decision.
4. Check each statement of the paper that depends on `z >= 0`. If a step needs a new Lean lemma, ask the user before
   you write it.
</steps>

<constraints>
- Do not change the uncertainty model itself. This plan makes the text true. A new model, such as one with a
  covariance bound, is a separate change with its own proof.
- A property that fails on a real input is a finding. Report it to the user. Do not weaken the test to make it pass.
</constraints>

<release>
A docstring and docs fix needs a `Fixed` fragment, because the docstrings are in the API reference.
</release>

<done>
- The module docstring, the docs, the paper and the prompts name the same uncertainty model for each claim.
- The monotonicity test runs for every model kind and fails if `_robust_scores` moves a score the wrong way.
- `prek run --all-files`, `uv run pytest`, `lake build` (if the Lean files changed) and the docs build pass.
</done>
