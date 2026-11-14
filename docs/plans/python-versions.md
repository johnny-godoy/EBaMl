# Plan: Python 3.11 and later (0.11.0)

You are a senior Python engineer who maintains libraries that support several Python versions. You will implement one
milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because
the project's hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
EBaMl installs and passes its tests on every supported CPython release from 3.11 to the newest stable release, and CI
proves it with a matrix. Many teams that fit EBMs run on Python 3.11 or 3.12 (managed notebooks, company base images),
and today `requires-python = ">=3.13"` turns them away. The package must claim exactly the versions that the matrix
passes: no more, and no fewer.
</goal>

<context>
These facts were true when the plan was written. Check each one again before you act on it.

- `pyproject.toml` sets `requires-python = ">=3.13"`, the classifier `Programming Language :: Python :: 3.13` and the
  ruff `target-version = "py313"`. The README "Requirements" says that CI tests Linux with Python 3.13 only.
- The source uses syntax and names that Python 3.11 lacks:
  - PEP 695 `type` aliases in `src/ebaml/core/custom_types.py` (`Frame`, `ClassLabel`, `FeatureValue`, the recursive
    `JSONValue`, `JSONMapping`, `JSONObject`) and in `src/ebaml/local/ortools_backend.py` (`MathOptSolve`).
  - PEP 695 generic classes and functions: `ShrinkingGraphCall[R, E: Exception]` and `call_with_shrinking_graph` in
    `src/ebaml/llm/retry.py`, and `coerce_enum[E: enum.Enum]` in `custom_types.py`.
  - `typing.override` (3.12) in `src/ebaml/edit/edits.py`.
  - About 20 more sites in `scripts/`, `benchmark/` and `tests/`. Find them all with ruff (step 2), not with a grep.
  - PEP 701 f-strings (a quote of the same kind inside the braces) are valid only from 3.12. Ruff reports them as syntax
    errors once the target is `py311`.
- The lower bounds of two direct dependencies need Python 3.12: `numpy>=2.5.4` and `scipy>=1.18.1`. `pandas`,
  `scikit-learn`, `polars` and `cyclopts` need 3.11. `baml-bridge==0.23.0`, `mcp`, `beartype`, `narwhals`, `orjson`,
  `structlog` and `attrs` accept 3.10 or older. Check `interpret` and `ortools` yourself.
- The wheel ships the generated `baml_sdk`. `baml-bridge` accepts 3.10, but the generated code may still use newer
  syntax. Import it on 3.11 before you decide anything.
- beartype checks every hint of `ebaml` at run time (`.claude/rules/libraries.md`). A change of an alias form changes
  what beartype checks, so the unit tests are the guard, not only the type checker.
- `uv` resolves one lockfile for every version in `requires-python`. With environment markers it can pick different
  versions of a package for different Python versions (a forked resolution).
</context>

<steps>
1. **Find the real floor.** In a scratch environment, run `uv sync --python 3.11` and `uv sync --python 3.12`, then
   `python -c "import ebaml, baml_sdk"` on each. Write down each blocker: a dependency that has no release for the
   version, generated code that does not import, or a feature that a dependency drops. If 3.11 needs a dependency floor
   that removes a feature or a security fix, stop and ask the user whether the floor is 3.11 or 3.12. Show the blocker
   and the cost of each choice.
2. **Lower the target of the tools.** Set `requires-python`, the ruff `target-version` and the Python version of
   pyrefly to the floor. Run `uv run ruff check` and fix each finding in the code. Do not add ignores for the `UP` rules
   or for version syntax errors: the point of the change is code that the oldest version runs.
3. **Rewrite the new syntax.** Use one form for each construct, and apply it everywhere:
   - Aliases: `typing_extensions.TypeAliasType` for the recursive `JSONValue` and for any alias that beartype must keep
     lazy, and a plain annotated alias (`Frame: TypeAlias = ...`) for the others. Check that beartype accepts each form
     on every version of the matrix.
   - Generics: `TypeVar` with a bound, declared once next to its users.
   - `override`: import it from `typing_extensions`, and add `typing_extensions` to the dependencies only if no
     dependency already brings it. deptry checks this.
   Keep the public names and their meaning. `tests/api/public_api.json` must not change, unless the user agrees.
4. **Fork the dependency floors only where needed.** If numpy or scipy cannot go below their 3.12 floor for all
   versions, give them two entries with environment markers (`numpy>=X; python_version < "3.12"`). Pick the newest
   release that supports 3.11, not an arbitrary one. Write a `# reason:` block above each forked entry. Run `uv lock`,
   and check that the lockfile resolves each version.
5. **Keep the developer tools on one version.** prek, the docs build, mutation testing, marimo and the Lean proofs need
   one Python only. Leave them on the newest version that the project uses. If a dependency group cannot install on the
   floor, give the group its own `requires-python` in uv instead of raising the floor of the package.
6. **Add the matrix to CI.** In `.github/workflows/ci.yml`, run the unit suite with the coverage gate on the newest
   version, and the unit, integration, API and contract suites without coverage on every other version. Before 1.0.0, a
   pull request pays for runner time on every push, so:
   - On each pull request, run the floor and the newest version.
   - On a push to `main`, on the weekly schedule and in the release workflow, run every version.
   - Run the next CPython pre-release on the schedule only, as a job that reports and does not gate.
   Matrix jobs get names from their version, so update `.github/rulesets/main.json`, the `GATES` of
   `.github/scripts/ci_report.py`, and keep `check-ruleset` and `check-report-gates` green. A required check must exist on
   every pull request.
7. **Test the install, not only the source.** Run `scripts/smoke_install.py` on the built wheel on every matrix version
   in `install-weekly.yml` and in `release.yml`. A source checkout can hide a missing file or a dependency marker that
   only the wheel shows.
8. **Keep the toolchain pin true.** `scripts/check_toolchain_pin.py` and `scripts/baml_toolchain.py` must give the same
   `baml generate` on every version. If the generated `baml_sdk` needs a newer Python than the floor, stop and ask the
   user: the floor of the wheel is the floor of `baml_sdk`.
9. **Make the claim match the result.** Set one `Programming Language :: Python :: 3.X` classifier for each version that
   the matrix passes. Update the README "Requirements", `docs/setup.md` and the table "Compatibility" of
   `docs/releasing.md`. Update the sentence about Python versions in `ROADMAP.md` and step 8 of
   `docs/plans/stable-api.md`, which then needs to add the operating systems only.
10. **Guard the claim.** Add a contract test in `tests/contract/` that reads `requires-python`, the classifiers and the
    versions of the CI matrix, and fails when they disagree. A claim that drifts from the matrix is the bug this
    milestone prevents.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.11.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment says under `Added` which Python versions the package supports, and under `Changed` each dependency floor that
the PR lowered or forked. Mark the 0.11.0 row of `ROADMAP.md` as done.
</release>

<done>
- `uv run pytest` passes on every version from the floor to the newest stable CPython, in CI.
- The wheel installs and imports on every claimed version (`scripts/smoke_install.py`).
- `requires-python`, the classifiers, the README and the CI matrix name the same versions, and a contract test checks it.
- `tests/api/public_api.json` is unchanged, or the user approved each change.
- `prek run --all-files`, mutation testing of the changed lines and the strict docs build pass.
</done>
