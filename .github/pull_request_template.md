<!--
Title: a conventional commit, for example "feat(llm): choose the model with model=". Use a lower-case description with
no period, at most 72 characters. A squash merge makes the title the commit on main.
.claude/rules/commits-and-prs.md has the rules.
-->

## Summary

<!-- What changes for users or contributors, and why. Two to five sentences. -->

## Changes

<!-- One line for each commit or group of commits. Each commit has one purpose. -->

-

## Changelog

<!-- Choose one. -->

- [ ] This PR adds a fragment in `changelog.d/` (`uv run scriv create`).
- [ ] Users see no change. The PR has the label `no-changelog`, or it changes no file in `src/ebaml/`, `baml_src/` or
      `benchmark/`.

## Breaking changes

<!-- "None", or what breaks and what users must change. Mark the fragment entry with **BREAKING:**. -->

None.

## Roadmap

<!-- The milestone and the plan in docs/plans/ that this PR works on, or "None". A PR that completes a milestone sets
its version in pyproject.toml and its status in ROADMAP.md. -->

None.

## Local verification

<!-- A runbook for the reviewer, so that they can see the change work. Write "None" if there is nothing to run, for
example for a CI or docs change. Do not write "run pytest": the Checks section covers that. Give exact commands that
the reviewer can paste, and say what they must see. Use the sample EBM, a script in the scratchpad or a marimo
notebook for the input. Mark each step that sends real requests and needs OPENROUTER_API_KEY. -->

1. Setup: the branch, the commands to install and generate (`uv sync`, `baml generate`), and the key if needed.
2. Run: the command or snippet that uses the changed behavior.
3. Expect: the output, file or log event that shows the change works.
4. Compare: the same command on `main`, and how its output differs. Use this for a bug fix.

## Checks

<!-- Tick what you ran. Say why for each check that you could not run. -->

- [ ] `prek run --all-files`
- [ ] `uv run pytest` (100% coverage)
- [ ] `uv run pytest README.md docs src --no-cov` (the documentation examples)
- [ ] `OPENROUTER_API_KEY=unused baml test -x root::Live` (if `baml_src/` changed)
- [ ] `uv run python scripts/mutation.py --base origin/main` (if `src/ebaml/` changed)
- [ ] `uv run --group docs python -m scripts.build_docs` (if the docs changed)
- [ ] `cd proofs && lake build` (if `proofs/` or code that a proof covers changed)
