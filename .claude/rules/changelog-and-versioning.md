# Changelog and versioning

## Semantic versioning

- This project follows [Semantic Versioning 2.0.0](https://semver.org/). The single source of truth is `version` in `pyproject.toml`.
- Bump rules:
  - MAJOR: incompatible change to the public API (including the generated `baml_sdk` surface or BAML function signatures consumers rely on).
  - MINOR: backwards-compatible new functionality.
  - PATCH: backwards-compatible bug fixes, docs, internal refactors.
- While the major version is `0`, breaking changes may land in a MINOR bump, but still list them under `Changed`/`Removed` and mark them **BREAKING**.
- After changing `version` in `pyproject.toml`, run `uv lock` so `uv.lock` stays in sync.
- Do not bump the version unless the user asks for a release, or the change set clearly warrants one and the user agrees. Never invent a version number; derive it from the rules above.

## Changelog

- Keep `CHANGELOG.md` at the repo root in [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format.
- Every user-visible change (feature, fix, behavior change, deprecation, removal, security fix, dependency change that affects users) needs an entry in the same change, under `## [Unreleased]`.
- Group entries under: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`. Omit empty groups.
- Write entries for users, not for the diff: say what changed and why it matters, one line each. Skip pure internal churn (formatting, test-only edits, CI tweaks).
- On release: rename `[Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and update the compare links at the bottom of the file.
- Keep the changelog and `pyproject.toml` version consistent: the newest released heading must match `version`.
