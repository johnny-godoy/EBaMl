# Contributing

1. Install [uv](https://docs.astral.sh/uv/) and the BAML toolchain (see the README), then `uv sync` and `baml generate`.
2. Make your change with tests. Run `prek run --all-files` and fix every failure; do not bypass the hooks.
3. If a function has an invariant, write a Hypothesis property test for it. CI loads the `ci` profile (see
   `tests/unit/conftest.py`), so CI draws the same examples every time. If a property finds a bug, keep the failing
   example as a separate regression test.
4. To fake a `baml_sdk` function in a test, wrap the fake in `fake_sdk` from `tests/unit/ebaml/conftest.py`. The
   wrapper checks each call against the stub that `baml generate` writes, so a changed signature fails the test.
5. A workaround (`# noqa`, `# type: ignore`, `pytest.mark.skip`, ...) needs a written reason next to it. Check with
   `uv run python -m scripts.check_ignores`.
6. For a user-visible change, run `uv run scriv create` and write the entries in the new file in `changelog.d/` (see
   `changelog.d/README.md`). Do not edit `CHANGELOG.md` or the version: the release workflow sets them after the merge.
   If the change is internal but touches `src/ebaml/`, `baml_src/` or `benchmark/`, add the label `no-changelog`.
7. Open a pull request. CI must pass.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
