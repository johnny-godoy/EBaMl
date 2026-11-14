# Plan: Python 3.12 and later (0.11.0)

You are a senior Python engineer who maintains libraries that support several Python versions. You will implement one
milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because
the project's hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
EBaMl installs and passes its tests on every CPython release from 3.12 to the newest stable release, and CI proves it
with a matrix. Many teams that fit EBMs run on Python 3.12 (managed notebooks, company base images), and today
`requires-python = ">=3.13"` turns them away. The package must claim exactly the versions that the matrix passes: no
more, and no fewer.
</goal>

<context>
These facts were true when the plan was written. Check each one again before you act on it.

- `pyproject.toml` sets `requires-python = ">=3.13"`, the classifier `Programming Language :: Python :: 3.13` and the
  ruff `target-version = "py313"`. The README "Requirements" says that CI tests Linux with Python 3.13 only.
- The user chose 3.12 as the floor, not 3.11. The reason: the lower bounds of `numpy` (2.5.4) and `scipy` (1.18.1)
  need Python 3.12, and 3.11 would need a forked resolution of both. Do not lower the floor below 3.12.
- The source uses PEP 695 `type` aliases and generics (`src/ebaml/core/custom_types.py`, `src/ebaml/llm/retry.py`,
  `src/ebaml/local/ortools_backend.py`) and `typing.override` (`src/ebaml/edit/edits.py`). Python 3.12 has all of
  them, so they stay.
- A search found no name that only 3.13 has (`typing.TypeIs`, `typing.ReadOnly`, `copy.replace`,
  `warnings.deprecated`, type parameter defaults). Ruff with `target-version = "py312"` is the real check: it reports
  each syntax error and each new name of the standard library that 3.12 lacks.
- Every direct dependency accepts 3.12 at its current lower bound. Check `interpret` and `ortools` again, and check
  the packages of the dependency groups (docs, dev, dist).
- The wheel ships the generated `baml_sdk`. `baml-bridge==0.23.0` accepts 3.10, but the generated code may still use
  newer syntax. Import it on 3.12 before you decide anything.
- beartype checks every hint of `ebaml` at run time (`.claude/rules/libraries.md`), so the unit suite on 3.12 is the
  guard for the hints, not only the type checker.
</context>

<steps>
1. **Check the floor.** In a scratch environment, run `uv sync --python 3.12`, then `python -c "import ebaml, baml_sdk"`
   and `uv run pytest --no-cov`. Write down each failure. If a dependency or the generated `baml_sdk` needs 3.13, stop
   and ask the user. Show the blocker and the cost.
2. **Lower the target of the tools.** Set `requires-python = ">=3.12"`, the ruff `target-version = "py312"` and the
   Python version of pyrefly. Run `uv lock`, then `uv run ruff check`, and fix each finding in the code. Do not add an
   ignore for a version syntax error.
3. **Keep the developer tools on one version.** prek, the docs build, mutation testing, marimo and the Lean proofs need
   one Python only. Leave them on the newest version that the project uses. If a dependency group cannot install on
   3.12, give the group its own `requires-python` in uv instead of raising the floor of the package.
4. **Add the matrix to CI.** In `.github/workflows/ci.yml`, run the unit suite with the coverage gate on the newest
   version, and the unit, integration, API and contract suites without coverage on every other version. Before 1.0.0, a
   pull request pays for runner time on every push, so:
   - On each pull request, run 3.12 and the newest version.
   - On a push to `main`, on the weekly schedule and in the release workflow, run every version.
   - Run the next CPython pre-release on the schedule only, as a job that reports and does not gate.
   Matrix jobs get names from their version, so update `.github/rulesets/main.json` and the `GATES` of
   `.github/scripts/ci_report.py`, and keep `check-ruleset` and `check-report-gates` green. A required check must exist on
   every pull request.
5. **Test the install, not only the source.** Run `scripts/smoke_install.py` on the built wheel on every matrix version
   in `install-weekly.yml` and in `release.yml`. A source checkout can hide a missing file that only the wheel shows.
6. **Keep the toolchain pin true.** `scripts/check_toolchain_pin.py` and `scripts/baml_toolchain.py` must give the same
   `baml generate` on every version.
7. **Make the claim match the result.** Set one `Programming Language :: Python :: 3.X` classifier for each version that
   the matrix passes. Update the README "Requirements", `docs/setup.md`, the "Requirements" of `CLAUDE.md` and the table
   "Compatibility" of `docs/releasing.md`.
8. **Guard the claim.** Add a contract test in `tests/contract/` that reads `requires-python`, the classifiers and the
   versions of the CI matrix, and fails when they disagree. A claim that drifts from the matrix is the bug that this
   milestone prevents.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.11.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment says under `Added` which Python versions the package supports. Mark the 0.11.0 row of `ROADMAP.md` as done.
</release>

<done>
- `uv run pytest` passes on every version from 3.12 to the newest stable CPython, in CI.
- The wheel installs and imports on every claimed version (`scripts/smoke_install.py`).
- `requires-python`, the classifiers, the README and the CI matrix name the same versions, and a contract test checks
  it.
- `tests/api/public_api.json` is unchanged.
- `prek run --all-files`, mutation testing of the changed lines and the strict docs build pass.
</done>
