# Contributing

1. Install [uv](https://docs.astral.sh/uv/) and the BAML toolchain (see the README), then `uv sync` and `baml generate`.
2. Make your change with tests. Run `prek run --all-files` and fix every failure; do not bypass the hooks.
3. If a function has an invariant, write a Hypothesis property test for it. CI loads the `ci` profile (see
   `tests/unit/conftest.py`), so CI draws the same examples every time. If a property finds a bug, keep the failing
   example as a separate regression test.
4. A workaround (`# noqa`, `# type: ignore`, `pytest.mark.skip`, ...) needs a written reason next to it. Check with
   `uv run python -m scripts.check_ignores`.
5. User-visible changes get an entry under `## [Unreleased]` in `CHANGELOG.md`.
6. Open a pull request. CI must pass.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
