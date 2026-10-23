# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Live prompt tests in `baml_src/prompt_tests.baml`. Run them with `baml test -i root::Live` and a real key.
- `EBMGraph.to_body_text` returns the graph text without the preamble.
- Runtime type checks with `beartype`. Every function in the `ebaml` package now checks its argument and return types. A call with a wrong type raises a `BeartypeCallHintViolation`.

### Changed

- `InvalidOptionError` takes any `Collection[str]` of valid options, not only a `list[str]`.
- Fixed type hints that did not match the code: `EBMGraph.x_vals` can hold categories, `EBMGraph.scores` can be a NumPy array, and `EBMGraph.from_ebm` and `EBMGraph.to_text` accept plain strings for their enum options.
- **BREAKING:** `ebaml.graph_text.graph_to_text` is now the method `EBMGraph.to_text`. The module `ebaml.graph_text` keeps `xy_to_json`, `parse_str_tuple_to_float_tuple` and `parse_graph_text`.
- Function parameters take abstract types (`Sequence`, `Mapping`, `Iterable`, `str | os.PathLike[str]`), and return values stay concrete.
- The warning for a large graph points at the caller, and not at the `beartype` wrapper.
- A separate `Prompt tests` workflow runs the live prompt tests with the `OPENROUTER_API_KEY` repository secret. It is slow, so the CI workflow does not wait for it.
- The offline BAML tests now run with `baml test -x root::Live`.

## [0.1.0] - 2024-06-19

### Added

- Project base.
