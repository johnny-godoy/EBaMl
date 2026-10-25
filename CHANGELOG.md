# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `EBMGraph.to_text` writes the confidence level of the bounds with its decimals, for example `87.5%`. It rounded the
  level to a whole percent before, so `EBMGraph.from_text` computed the standard deviations at the wrong level.
- `EBMGraph.from_ebm` and `EBMGraph.simplify` return an instance of the subclass when you call them on a subclass of
  `EBMGraph`. They returned a plain `EBMGraph` before, so the runtime type check raised an error.
- `parse_str_tuple_to_float_tuple` raises a `ValueError` for a text with more or fewer than two values. It returned a
  tuple of any length before, for example three ends for the interval `"(1, 2, 3)"`.
- `TaskResult.accuracy` is `0.0` for a task with no outcomes. It raised a `ZeroDivisionError` before, and so did
  `format_results`.

### Changed

- **BREAKING:** The modules of `ebaml` now sit in packages. Change your imports as the table shows. The classes and
  functions are the same.

  | Before | After |
  | --- | --- |
  | `ebaml.custom_types`, `ebaml.exceptions` | `ebaml.core.custom_types`, `ebaml.core.exceptions` |
  | `ebaml.graphs`, `ebaml.graph_text` | `ebaml.graphs.ebm_graph`, `ebaml.graphs.graph_text` |
  | `ebaml.local`, `ebaml.local_text` | `ebaml.local.facts`, `ebaml.local.fact_text` |
  | `ebaml.descriptors`, `ebaml.options`, `ebaml.retry` | `ebaml.llm.descriptors`, `ebaml.llm.options`, `ebaml.llm.retry` |

  The names `ebaml.EBMGraph`, `ebaml.Audience` and the other names in `ebaml.__all__` do not change. The `structlog`
  logger `ebaml.descriptors` is now `ebaml.llm.descriptors`.
- All prompts in `baml_src/` follow the prompting guide. Inputs are in XML tags, long data comes first and the task
  comes last, rules give their reason, and multi-turn requests are in `<instructions>` tags. The default
  `task_description` is now "Describe the general pattern of the graph." Compare results before and after on the
  benchmark.
- **BREAKING:** `ebaml.benchmark` no longer imports `baml_sdk`. `run_task(task, data_dir, limit)` is now
  `run_task(task, questions, ask)`. Load the questions with `load_questions` and cut them with a slice for a limit. Pass
  `baml_sdk.answer_benchmark_question` as `ask`, or any function with the same keyword arguments that returns an object
  with `text` and `usage`. `benchmark/run.py` does this for you.
- **BREAKING:** `describe_graph`, `describe_ebm` and `explain_instance` no longer take `**kwargs`. Three option classes
  replace them, as keyword-only arguments. `graph=GraphOptions(...)` (module `ebaml.llm.options`) holds the options of
  `EBMGraph.from_ebm` and `EBMGraph.to_text`. `prompt=PromptOptions(...)` (module `ebaml.llm.options`) holds
  `expert_description`, `graph_description`, `dataset_description` and `task_description`. `local=LocalOptions(...)`
  (module `ebaml.local.facts`, only for `explain_instance`) holds the options of the local facts. A misspelled option was
  silently ignored before. It now raises a `TypeError`. For example, change `describe_ebm(ebm, dataset_description="x")`
  to `describe_ebm(ebm, prompt=PromptOptions(dataset_description="x"))`.
- **BREAKING:** `ebaml.graphs.ebm_graph` no longer imports `baml_sdk`. `EBMGraph.to_text` takes a new keyword argument,
  `check_budget`, a function that compares the estimated tokens of the text with `max_tokens` and
  `overbudget_percent`. `baml_sdk.check_token_budget` is such a function. If you call `EBMGraph.to_text` yourself and
  pass no `check_budget`, the function checks nothing: it writes the text once, and does not warn or simplify. Pass
  `check_budget=baml_sdk.check_token_budget` to get the old behavior. `describe_graph` and `describe_ebm` pass it for you.
- **BREAKING:** `facts_to_texts` moved from the local facts module to the new module `ebaml.local.fact_text`.
  `ebaml.local.facts` now computes the facts only, and `ebaml.local.fact_text` renders them. Change
  `from ebaml.local import facts_to_texts` to `from ebaml.local.fact_text import facts_to_texts`.

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
