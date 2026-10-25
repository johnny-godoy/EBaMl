# Changelog and versioning

## Release each user-visible change in its own PR

Every merge to `main` reaches the users of the repository, so a PR with a user-visible change is a release. Agents left
changes under `[Unreleased]` with the old version before, and the version then did not say what users had. So:

- In the PR that makes a user-visible change, raise `version` in `pyproject.toml`, run `uv lock`, and write the
  entries under a new heading `## [X.Y.Z] - YYYY-MM-DD` with today's date. Do not keep an `[Unreleased]` section.
- Raise the version from the one on the base branch. If another release merges first, the merge conflict in
  `pyproject.toml`, `uv.lock` or `CHANGELOG.md` shows it: take the base version, raise it again, and keep your
  entries under your own new heading, never under a heading that is already on `main`.
- One PR is one release, even with several changes. Take the largest bump that any of its changes needs.
- A PR with no user-visible change (tests, CI, hooks, contributor docs, plans) gets no entry and no new version.
- The `check-changelog` hook (`scripts/check_changelog.py`) fails on an `[Unreleased]` heading, on a version that does
  not match the newest heading, and on headings out of order.
- After the merge, the maintainer pushes the tag `vX.Y.Z`, and `.github/workflows/release.yml` publishes it. Do not push
  tags.

## Semantic versioning

The project follows [Semantic Versioning 2.0.0](https://semver.org/) strictly. The single source of truth is `version`
in `pyproject.toml`. `ROADMAP.md` lists the planned versions.

| Version range | Breaking change | New feature | Fix, docs, internal change |
| --- | --- | --- | --- |
| Before 1.0.0 | PATCH, marked **BREAKING** | PATCH | PATCH |
| From 1.0.0 | MAJOR, after a deprecation in a MINOR | MINOR | PATCH |

- Before 1.0.0, the PR that completes a milestone of `ROADMAP.md` takes the MINOR version of that milestone (for
  example 0.1.0) instead of a PATCH. The roadmap then decides the MINOR numbers, and the PRs between milestones do not
  use them up.
- The public API is the public Python names, the BAML entry points that `baml_sdk` exposes, the MCP tool names and
  argument schemas, and the CLI flags. Prompt wording, LLM output text and log events are not part of it.
- `ebaml.experimental` and the MCP tools whose names start with `experimental_` are outside the public API. They can
  change in a MINOR or PATCH at any time. Say so in their docstrings.
- From 1.0.0, deprecate before you remove: keep the old name for at least one MINOR with a `DeprecationWarning`, and add
  a `Deprecated` entry. Remove it only in the next MAJOR.
- Never invent a version. Derive it from the table above and from `ROADMAP.md`.

## Changelog

- Keep `CHANGELOG.md` at the repo root in [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format, with
  no `[Unreleased]` section, because each change is released at once.
- Group entries under: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`. Omit empty groups.
- Write entries for users, not for the diff: say what changed and why it matters, one line each. Skip pure internal churn (formatting, test-only edits, CI tweaks).
- The changelog is for people who use the package: public API changes, BAML function signatures, prompt behavior, bug fixes and dependency changes that affect them. Leave out repository metadata and developer tooling (`CITATION.cff`, `dependabot.yml`, issue and PR templates, CI workflows, test suites, prek hooks, contributor docs, `ROADMAP.md`, `docs/plans/`), even when the change is new.
- Do not add compare links at the bottom of the file while the versions have no tags.
