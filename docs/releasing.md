# Releasing

A merge to `main` releases the pending changelog fragments. This page lists what the release workflow checks, what a
user can check, and what to do when a release fails. The workflow is `.github/workflows/release.yml`.

## What a release contains

| File | Purpose |
| --- | --- |
| `ebaml-<version>-py3-none-any.whl` | The wheel. It holds the generated `baml_sdk`, so it needs no BAML toolchain. |
| `ebaml-<version>.tar.gz` | The source distribution. |
| `SHA256SUMS` | The SHA-256 checksum of each distribution file. |
| `ebaml-sbom.cdx.json` | A CycloneDX 1.5 SBOM of the packages in `uv.lock`, the set that the release was tested with. |

The GitHub release has all four files. PyPI has the two distribution files and an attestation for each.

## The order of the gates

1. **CI.** The workflow starts when the CI workflow of `main` ends. It stops unless CI passed on the head of `main`. A
   manual run checks the same. If `main` moved on while CI ran, the CI run of the newer commit releases it.
2. **Version.** `scripts/release.py` chooses the version from the fragments. The workflow pushes the release commit and
   the tag. The release commit changes only the version and `CHANGELOG.md`, so it has no CI run before the build.
3. **Build.** The job builds from the tag and checks, in this order:
   - the tag matches the version,
   - the wheel holds `baml_sdk`,
   - `twine check --strict` accepts the metadata of both files,
   - `pydistcheck` finds no unexpected file,
   - each file installs in a clean environment with no `baml` command, and `scripts/smoke_install.py` passes. The test
     imports the package and the SDK, builds the graph text of a small EBM, and runs `ebaml-mcp --help`.
4. **Upload.** The publish job checks `SHA256SUMS` against the downloaded files. It then runs `uv publish`, which uploads
   a PEP 740 attestation for each file. It runs only if the repository variable `PUBLISH_TO_PYPI` is `true`.
5. **Verify.** The last job installs the published version from PyPI in a clean environment and runs the smoke test
   again. PyPI can need some minutes to list a new version, so the job tries again for several minutes.

A weekly workflow (`install-weekly.yml`) installs the wheel of `main` with the newest dependencies and with the lowest
versions that the requirements allow. It opens an issue if the smoke test fails. The smoke test found two faults that
the test suite cannot see, because the lockfile hides them: a `baml-bridge` that does not match the generated SDK, and
`mcp` 2.x.

## Check a release

Compare the checksum of a downloaded file with `SHA256SUMS`:

```bash
sha256sum --check --ignore-missing SHA256SUMS
```

Check the attestation of a file on PyPI. It names the repository and the workflow that built the file:

```bash
uvx pypi-attestations verify pypi --repository https://github.com/johnny-godoy/EBaMl \
  pypi:ebaml-<version>-py3-none-any.whl
```

The SBOM lists the locked packages. A user who installs from PyPI gets the newest releases that the requirements allow,
so the installed set can differ from the SBOM.

## Recover from a failed release

PyPI never accepts the same file name twice, so a release is fixed with a new version.

| Failure | What to do |
| --- | --- |
| The build job fails. | Nothing is public. Fix the cause in a pull request with a fragment. The release commit and the tag already exist, so the next merge releases a new PATCH version. |
| The upload fails before PyPI accepts a file. | Use "Re-run failed jobs" on the same workflow run. The jobs reuse the built files, so the upload sends the files that the build job tested. |
| The verify job fails. | Read the log. If the version does not install, or the smoke test fails, yank the version on PyPI (a yanked version stays installable only by an exact pin) and release a fix. |
| A published version has a defect. | Yank it on PyPI, fix the defect in a pull request with a `Fixed` fragment, and merge. The fix is a new PATCH version. |

To yank a version, open the page of the project on PyPI, choose "Manage", select the release, and choose "Yank". Give
the reason, because PyPI shows it.

## Hotfix

There is one branch of releases, `main`, so a hotfix is a normal pull request with a `Fixed` fragment. The release
workflow gives it the next PATCH version after the merge. Mark a fix that must wait for no other change by keeping the
pull request small. If `main` holds a change that is not ready, revert it in its own pull request first.

## Review rules

`.github/rulesets/main.json` requires the CI gates, the docs build, and the `Changelog` and `Commits` checks. It does
not require a human review: the repository has one maintainer, so the gates carry the review. An admin can push without
a pull request, and the release workflow uses that right (`RELEASE_TOKEN`) to push the release commit and the tag. The
ruleset makes no other use of the bypass. Add a required review when a second maintainer joins.

## Compatibility

| What | When it runs | State |
| --- | --- | --- |
| Tests, coverage, mutation testing | Each pull request. | Linux, Python 3.13. |
| Clean install of the wheel and the sdist | Each release. | Linux, Python 3.13. |
| Install with the newest and the lowest dependencies | Each week. | Linux, Python 3.13. |
| Other operating systems and Python versions | Planned before 1.0.0. | The plan `docs/plans/stable-api.md` lists them. |

`requires-python` is `>=3.13`. The tests cover Python 3.13 only, so a later version is not tested yet.

The documentation site shows the state of `main`. Each version has its entry in the [changelog](changelog.md).

## Versions and deprecation before 1.0.0

Before 1.0.0, a release can break the public API. The changelog marks such a change with **BREAKING**, and a pull
request that removes or changes an exported name fails a CI step unless a fragment has that mark
(`scripts/check_api_breaks.py`). The changelog entry names the replacement. Deprecation warnings and a migration guide
start with 1.0.0. See `docs/plans/stable-api.md`.

## Saved files

The learned token weights are in `$XDG_CACHE_HOME/ebaml/token_weights.json`. A file that the package cannot read is
ignored with a `token_weights_unreadable` warning, and the package learns the weights again. A change of the file
format therefore needs no migration.
