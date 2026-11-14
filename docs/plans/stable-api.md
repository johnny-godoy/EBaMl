# Plan: stable API (1.0.0)

You are a senior Python engineer who maintains libraries with a stable public API. You will implement one milestone of
`ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's
hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
Release 1.0.0, the last milestone of `ROADMAP.md`: declare the public API, and add the checks that stop an accidental
breaking change. After this release, a breaking change of the public API needs a new MAJOR version
(`.claude/rules/changelog-and-versioning.md`). This milestone adds no feature. A release that only stabilizes is easy
to review.
</goal>

<context>
- 1.0.0 needs 0.10.0. By then every other milestone of `ROADMAP.md` is on `main`: the model providers, the model
  kinds, the custom audiences, the MCP server with its read, explain and edit tools, the local explanations of every
  model kind, and model editing with GAM Changer parity.
- The public API is: the public Python names, the BAML entry points that `baml_sdk` exposes, the MCP tool names and
  argument schemas, and the CLI flags (`benchmark/run.py`, `ebaml-mcp`). Prompt wording, LLM output text and log events
  are not part of it.
- `.claude/rules/writing-style.md` has a section "Current state, not history" that says the package makes no promise of
  backward compatibility. That stops being true at 1.0.0.
</context>

<steps>
1. **Audit the public names.** List every name that a user can import without a leading underscore. For each one,
   decide: public (keep, document), or internal (rename with an underscore, or move into a private module). Give each
   public module an `__all__`. Ask the user about each name that you are not sure of, before you change it.
2. **Write the contract.** Add a page `docs/stability.md` (and a `zensical.toml` entry) that lists what the public API
   covers, what it does not cover, and the deprecation policy. Link it from the README.
3. **Check the Python API.** `scripts/release/api_breaks.py` compares `ebaml.__all__` with the base branch through griffe, and
   it passes a break only if a fragment has a **BREAKING** entry. Extend it for 1.0.0: compare with the last released tag
   when a pull request targets a release, and fail a break that has no MAJOR version. Until `v1.0.0` exists, the base
   branch is the reference.
4. **Check the BAML entry points.** Add a snapshot test of the signatures of the public `baml_sdk` functions, read from
   the generated stub as `tests/unit/ebaml/conftest.py` does, so a changed BAML signature shows in review.
5. **Check the MCP schemas.** The snapshot test in `tests/unit/ebaml/mcp/test_server.py` exists. Make sure CI runs it.
6. **Add deprecation support.** Add one small helper that emits a `DeprecationWarning` with the release that will
   remove the name, and a semgrep rule or a test that every deprecated name has a `Deprecated` changelog entry.
7. **Update the rules.** Rewrite "Current state, not history" in `.claude/rules/writing-style.md`: the docs describe
   the current state, and deprecated items stay in the docs, marked as deprecated, until their removal. Update
   `CLAUDE.md` if it mentions the pre-1.0 policy.
8. **Test every claimed platform and Python version.** Until 1.0.0, CI runs Linux with Python 3.13 only, so a pull request
   does not pay for a platform that no user needs. Before 1.0.0, add the matrix and decide what the package claims:
   - Add a matrix of operating systems (Linux, macOS, Windows) and Python versions (3.13 and each later release that
     exists) to the pytest gate, or to a gate that runs the unit tests and `scripts/release/smoke_install.py`. A full matrix on
     each pull request costs runner time. Prefer a schedule and the release workflow, and keep Linux with one Python
     version on each pull request.
   - Run `scripts/release/smoke_install.py` on each matrix entry in the release workflow, so a release installs on every claimed
     platform. The script uses `bin/python`, so make it find `Scripts/python.exe` on Windows.
   - Make the claim match the result. Set the classifiers `Operating System ::` and `Programming Language :: Python ::`
     of `pyproject.toml` to what the matrix passes, and lower or raise `requires-python` the same way. Update the
     "Requirements" list of the README and the table "Compatibility" of `docs/releasing.md`.
9. **Version the documentation.** The site shows `main`. Publish a copy for each released version, with a selector, or
   write on the site which version it shows. Check that Zensical supports versions before you choose.
10. **Write the migration guide.** Add `docs/migration.md` with one section for each MAJOR version, linked from the
    changelog entries that carry **BREAKING**. Add a second reviewer to the ruleset of `main` if the project has a second
    maintainer by then (`docs/releasing.md`, "Review rules").
11. **Update the user docs.** Make the README table "Supported models" true, and remove "Full support is on the roadmap"
   where it does not apply.
</steps>

<release>
This PR completes the milestone: it sets `version = "1.0.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment lists the public API, links `docs/stability.md`, and lists the names that the audit made private as
**BREAKING**. Mark the 1.0.0 row of `ROADMAP.md` as done. The release workflow tags and publishes 1.0.0 after the
merge. From then on, `scripts/release/version.py` raises the MAJOR, MINOR or PATCH number from the fragments.
</release>

<done>
- Every public name is documented in `docs/api.md`, and every other name is private.
- A change of a public name, a BAML entry point or an MCP schema fails a check in CI. `scripts/release/api_breaks.py`
  already fails a break of `ebaml.__all__` without a **BREAKING** fragment. Extend it to compare with the last tag, and to
  fail a break without a MAJOR version.
- CI and the release workflow cover every platform and Python version that the package claims.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass.
</done>
