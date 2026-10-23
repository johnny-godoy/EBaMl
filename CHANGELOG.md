# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Live prompt tests in `baml_src/prompt_tests.baml`. Run them with `baml test -i root::Live` and a real key.
- `CITATION.cff`, so GitHub shows a "Cite this repository" button.

### Changed

- A separate `Prompt tests` workflow runs the live prompt tests with the `OPENROUTER_API_KEY` repository secret. It is slow, so the CI workflow does not wait for it.
- The offline BAML tests now run with `baml test -x root::Live`.

## [0.1.0] - 2024-06-19

### Added

- Project base.
