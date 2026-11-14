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

## Tests and guards

<!-- Tick each kind of test or guard that this PR adds or changes, and name the tests after the colon. Write "None" for
a change with no code. .claude/rules/testing.md tells which suite answers which question, and
.claude/rules/bug-fixes.md tells which guard a bug fix needs. -->

- [ ] Unit tests in the mirrored path of `tests/unit/ebaml/`:
- [ ] Property-based tests (Hypothesis):
- [ ] Differential test against a reference oracle. An oracle changes only with the approval of the user:
- [ ] Integration tests in `tests/integration/` (the real `baml_sdk` and a local LLM server):
- [ ] API tests and snapshots (`public_api.json`, `tool_schemas.json`, `test_mcp_tools.py`, `test_command_lines.py`,
      `test_exceptions.py`):
- [ ] Contract tests in `tests/contract/`. A new call into `baml_sdk` needs a row in `CALLED`:
- [ ] Examples in the docs (`README.md`, `docs/*.md`, the docstring `Examples`):
- [ ] Offline BAML tests in `baml_src/tests/`, or live prompt tests in `baml_src/tests/live/`:
- [ ] Tests of a progress bar (one line for each step, no output by default, no bar for an empty loop):
- [ ] A guard of a bug fix that fails on the old code (a unit test, or a semgrep rule with `ruleid` and `ok` lines in
      `semgrep/ebaml.py`):
- [ ] A new semgrep rule or import-linter contract:
- [ ] A semantic contract of `.claude/rules/semantic-contracts.md` that the change touches, and the test that guards it:
- [ ] A Lean proof or a paper in `proofs/`:
- Guarantees that no test or proof shows:

## Abstraction reuse

<!-- Answer for a change that adds a shared Protocol, ABC, service, adapter or cross-module helper. Write "None" for any
other change. A reviewer judges the reasons. CI checks only the registry. -->

- New shared abstraction? [ ] No [ ] Yes
- Existing candidates inspected (`uv run python -m scripts.architecture.abstractions --search <words>`):
- Capability ID and owner in `architecture/abstractions.yml` (if new):
- Why the existing abstractions cannot be reused:

## Workarounds

<!-- Answer for a change that adds an ignore (`noqa`, `type: ignore`, `pyrefly: ignore`, a pragma, a skipped test or a
config ignore). Write "None" for any other change. .claude/rules/prek-and-ignores.md has the rules. -->

- Total from `uv run python -m scripts.ignores.report` (before and after):
- Why no shared fix applies (helper, import, fixture or one config entry):

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
its version in pyproject.toml and its status in ROADMAP.md, and deletes its plan unless a later milestone uses it. -->

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
- [ ] `uv run python scripts/mutation/run.py --base origin/main` (if `src/ebaml/` changed)
- [ ] `uv run --group docs python -m scripts.docs.build` (if the docs changed)
- [ ] `cd proofs && lake build` (if `proofs/` or code that a proof covers changed)
