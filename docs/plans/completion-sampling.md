# Plan: completion shares are tested estimates

You are a senior engineer with a background in statistics. You will test the count-completions estimate against an
exact count, and remove a test that asserts a property that the estimate does not have. Read `CLAUDE.md` and every file
in `.claude/rules/` first, especially `semantic-contracts.md` (contract 6), because the hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
A test fails when the sampling of `_completions` stops estimating the share that its docs describe. No test asserts
that the share grows.
</goal>

<context>
Found on 2026-11-09, in `_completions` of `src/ebaml/local/facts.py`:

- One `np.random.default_rng(options.seed)` draws `n_samples` bins for each feature, with the probabilities
  `weights / weights.sum()`. Each step uses the same draws, so the steps differ only by the known features.
- The draws cover every bin. The sufficient reason uses only the plausible bins (`min_support`). Both are correct,
  because the share describes the training distribution, but the docs must say that the two domains differ.
- The share uses the nominal scores and ignores `z`.
- A completion keeps the decision when `(intercept + total > 0) == (sign == 1)` for each margin, which matches the tie
  rule of `local_kinds.py`.

In `tests/unit/ebaml/local/test_facts.py`, `test_completions_are_reproducible_and_grow` asserts that the last share is
at least the first. That holds for its seed, but the estimate has no such property: a share can fall when an input
becomes known. Pull request johnny-godoy/EBaMl#124 removed the same claim from the prompt.
</context>

<steps>
1. Remove the growth assertion, and rename the test to say what it checks (reproducible shares in `[0, 1]`).
2. Write a test of the exact share on tiny models (2 or 3 features, 2 or 3 bins): sum the product of the bin
   probabilities over each completion that keeps the decision. Assert that the estimate is within four binomial
   standard errors, `4 * sqrt(p * (1 - p) / n_samples)`, plus a small constant for `p` near 0 or 1. Fix the seeds, so
   that the test is deterministic.
3. Add a test with unequal bin weights, where uniform sampling gives a share outside the bound of step 2. The test then
   fails if the code samples uniformly.
4. Add a test of a model where the exact share falls when the first input becomes known. It shows that a fall is a
   correct result.
5. Check that the same seed gives equal shares, that two seeds give different draws, and that `n_samples` changes only
   the precision. Do not assert equal shares for two sample counts.
6. Read the completions section of `docs/local-explanations.md`. Make it name the assumptions: independent inputs,
   the training bin weights, every bin (not only the plausible bins), the nominal scores and a fixed seed.
</steps>

<release>
Tests only, unless the docs change: a docs change in `docs/` for users gets a `Fixed` fragment.
</release>

<done>
- The exact-count test covers binary, multiclass and regressor decisions.
- The test of step 3 fails when you replace the weights with uniform probabilities. Check it by hand, then revert.
- `prek run --all-files` and `uv run pytest` pass.
</done>
