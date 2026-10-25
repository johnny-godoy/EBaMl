# Plan: stable API (1.0.0)

You are a senior Python engineer who maintains libraries with a stable public API. You will implement one milestone of
`ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's
hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
Release 1.0.0: declare the public API, and add the checks that stop an accidental breaking change. After this release,
a breaking change of the public API needs a new MAJOR version (`.claude/rules/changelog-and-versioning.md`). This
milestone adds no feature. A release that only stabilizes is easy to review.
</goal>

<context>
- 1.0.0 needs 0.3.0. By then the model providers (0.1.0), the model kinds and the general local types (0.2.0) and the
  read and explain tools of the MCP server (0.3.0) are on `main`.
- The public API is: the public Python names, the BAML entry points that `baml_sdk` exposes, the MCP tool names and
  argument schemas, and the CLI flags (`benchmark/run.py`, `ebaml-mcp`). Prompt wording, LLM output text and log events
  are not part of it. `ebaml.experimental` and the MCP tools whose names start with `experimental_` are not part of it.
- `.claude/rules/writing-style.md` has a section "Current state, not history" that says the package makes no promise of
  backward compatibility. That stops being true at 1.0.0.
</context>

<steps>
1. **Audit the public names.** List every name that a user can import without a leading underscore. For each one,
   decide: public (keep, document), or internal (rename with an underscore, or move into a private module). Give each
   public module an `__all__`. Ask the user about each name that you are not sure of, before you change it.
2. **Write the contract.** Add a page `docs/stability.md` (and a `zensical.toml` entry) that lists what the public API
   covers, what it does not cover, and the deprecation policy. Link it from the README.
3. **Check the Python API.** Add a CI step that runs `griffe check ebaml --against v1.0.0` (griffe is in the docs
   group) once the tag exists, so a breaking change of a public name fails CI. Until the tag exists, check against the
   base branch.
4. **Check the BAML entry points.** Add a snapshot test of the signatures of the public `baml_sdk` functions, read from
   the generated stub as `tests/unit/ebaml/conftest.py` does, so a changed BAML signature shows in review.
5. **Check the MCP schemas.** The snapshot test of the `ebm2mcp.md` plan exists. Make sure CI runs it.
6. **Add deprecation support.** Add one small helper that emits a `DeprecationWarning` with the release that will
   remove the name, and a semgrep rule or a test that every deprecated name has a `Deprecated` changelog entry.
7. **Update the rules.** Rewrite "Current state, not history" in `.claude/rules/writing-style.md`: the docs describe
   the current state, and deprecated items stay in the docs, marked as deprecated, until their removal. Update
   `CLAUDE.md` if it mentions the pre-1.0 policy.
8. **Update the user docs.** Make the README table "Supported models" true, and remove "Full support is on the roadmap"
   where it no longer applies.
</steps>

<release>
This PR completes the milestone and takes version 1.0.0. Its `CHANGELOG.md` entry lists the public API, links
`docs/stability.md`, and lists the names that the audit made private as **BREAKING**. Mark the 1.0.0 row of
`ROADMAP.md` as done. After the merge, the maintainer pushes the tag `v1.0.0`.
</release>

<done>
- Every public name is documented in `docs/api.md`, and every other name is private.
- A change of a public name, a BAML entry point or an MCP schema fails a check in CI.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass.
</done>
