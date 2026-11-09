# Plan: every result names the guarantee that it has

You are a senior Python engineer. You will make the docstrings of the local explanations state the guarantee that each
result has, and add tests that tie each solver path to the guarantee that it reports. Read `CLAUDE.md` and every file
in `.claude/rules/` first, especially `semantic-contracts.md` (contract 2), because the hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
No docstring, doc or prompt calls a result the smallest unless `minimum` is true. A test covers each path that sets
`minimum`, from the solver to the `is_smallest_set` key that the prompt reads.
</goal>

<context>
The review asked for a typed guarantee. Most of it exists. Found on 2026-11-09:

- `Counterfactual`, `SufficientReason` and `Cover` carry `minimum: bool` and `lower_bound: int`. `make_cover` in
  `src/ebaml/local/covering_problem.py` sets `minimum` to "proved, or the size reaches the bound". Sorting sets it to
  true (`_counterfactual`, `_sufficient_reason` in `facts.py`).
- `facts_to_texts` writes `is_smallest_set`, and a data scientist also gets the lower bound.
  `test_a_counterfactual_that_is_not_proved_is_not_called_the_smallest` and
  `test_a_minimal_set_is_not_called_the_smallest` exist in `tests/unit/ebaml/local/test_fact_text.py`.
- The prompts handle `is_smallest_set: false`, with offline and live tests.

These texts are wrong:

- The docstring of `Counterfactual` starts "The smallest set of changes after which the decision does not hold". With
  `target_class` and `GreedyCover`, `LocalSearchCover`, or a `MipCover` that hits its time limit, the set is not proved
  the smallest.
- The docstring of `compute_local_facts` says "one prediction of a binary classifier or a regressor", and that it
  raises for "an EBM that is not a binary classifier or a regressor". Multiclass classifiers work
  (`MulticlassLocal` in `local_kinds.py`). The `Raises` section of `_local_kind` says the same.
</context>

<steps>
1. Fix the three docstrings. Search `src/ebaml/local/`, `docs/` and `README.md` for "smallest", "fewest",
   "minimum", "minimal", "optimal" and "binary", and read each match against the code.
2. Write one parametrized test, end to end through `compute_local_facts` and `facts_to_texts`, with one case for each
   way that `minimum` is set:
   - sorting (one margin): true;
   - `MipCover` with an optimal result: true;
   - `MipCover` at its time limit, with a fake backend that returns a feasible result that is not optimal: false,
     unless the size reaches the bound;
   - `GreedyCover` and `LocalSearchCover`, with a set larger than the bound: false;
   - `_no_set_found` of a monotone problem: all groups, false unless they reach the bound.
   Assert `minimum`, `lower_bound` and the `is_smallest_set` key. Read `tests/unit/ebaml/local/test_covering_solvers.py`
   first, and do not repeat a case that it covers at the level of the solver.
3. Decide with the user whether to replace `minimum: bool` with an enum that says why (a proof by sorting, a proof by
   the solver, the size at the bound, no proof). The prompts need only the boolean, so recommend the change only if a
   consumer needs the reason. A change of the field is **BREAKING**.
</steps>

<release>
Add a `Fixed` fragment for the docstrings, because the API reference shows them.
</release>

<done>
- No text calls a set the smallest when `minimum` can be false.
- The parametrized test covers each path above.
- `prek run --all-files`, `uv run pytest` and the docs build pass.
</done>
