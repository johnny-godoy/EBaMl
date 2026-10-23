# Contributing

1. Install [uv](https://docs.astral.sh/uv/) and the BAML toolchain (see the README), then `uv sync` and `baml generate`.
2. Make your change with tests. Run `prek run --all-files` and fix every failure; do not bypass the hooks.
3. A workaround (`# noqa`, `# type: ignore`, `pytest.mark.skip`, ...) needs a written reason next to it. Check with
   `uv run python -m scripts.check_ignores`.
4. User-visible changes get an entry under `## [Unreleased]` in `CHANGELOG.md`.
5. Open a pull request. CI must pass.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
