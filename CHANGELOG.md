# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING:** `ebaml.graphs` no longer imports `baml_sdk`. `EBMGraph.to_text` takes a new keyword argument,
  `check_budget`, a function that compares the estimated tokens of the text with `max_tokens` and
  `overbudget_percent`. `baml_sdk.check_token_budget` is such a function. If you call `EBMGraph.to_text` yourself and
  pass no `check_budget`, the function checks nothing: it writes the text once, and does not warn or simplify. Pass
  `check_budget=baml_sdk.check_token_budget` to get the old behavior. `describe_graph` and `describe_ebm` pass it for you.
- **BREAKING:** `facts_to_texts` moved from `ebaml.local` to the new module `ebaml.local_text`. `ebaml.local` now computes
  the facts only, and `ebaml.local_text` renders them. Change `from ebaml.local import facts_to_texts` to
  `from ebaml.local_text import facts_to_texts`.

## [0.0.2] - 2026-10-23

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

## [0.0.1] - 2026-10-23

### Added

- Project base.
