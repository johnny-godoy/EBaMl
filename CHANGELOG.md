# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- scriv-insert-here -->

<a id='changelog-0.2.3'></a>
## [0.2.3] - 2026-10-30

### Fixed

- A local explanation accepts a row of the training data whose categories are numbers or booleans, such as a `pd.Categorical` of `1, 2, 3` or of `True, False`. interpret keeps these categories as text, so such a row raised `InvalidInstanceError: Unknown category`. A value that cannot be a category, such as a complex number, raises `InvalidInstanceError`.

- A local explanation raises `InvalidInstanceError` when the value of a continuous feature is not a number, such as `"abc"`. It raised a bare `ValueError` or `TypeError`, which `except EBaMlError` did not catch.

- `EBMGraph.to_text` with a boolean format says how many x-axis values the feature has, and that the format needs exactly two. The message said "more than x-axis values", which was also wrong for a feature with one value.

<a id='changelog-0.2.2'></a>
## [0.2.2] - 2026-10-30

### Fixed

- `EBMGraph.from_ebm`, `describe_graph` and `describe_ebm` read the scores of the right feature when `exclude` or `remove_terms` leaves a feature without a term. The graph of a feature showed the scores of another feature, or raised an `IndexError`. A feature without a term raises `UnsupportedModelError`, and `describe_ebm` skips it.
- The feature importances that `describe_ebm` sends to the summary pair each feature with its own importance when a feature has no term.
- The local explanations name the features without a term in their `UnsupportedModelError`, and do not say that the model has interactions.

<a id='changelog-0.2.1'></a>
## [0.2.1] - 2026-10-29

### Added

- A proof page shows that only the differences between the classes of a bin change the predicted probabilities of a multiclass classifier, and that its graphs keep the predictions of the EBM. A Lean proof checks the claims.

<a id='changelog-0.2.0'></a>
## [0.2.0] - 2026-10-29

### Added

- `model_kind(ebm)` reads the kind of a fitted EBM as a `BinaryClassifier`, `MulticlassClassifier` or `Regressor`. Each kind gives the names of the outputs, the meaning of a score and the conversion of scores to the output.

- `EBMGraph.from_ebm` reads the graph of a multiclass classifier: one score per bin and per class, centred over the classes of each bin. `EBMGraph.classes` names the classes, and `to_text` writes one series per class after a `Class:` line. `from_text` reads that text back.
- `GraphOptions.classes` and the `classes` argument of `EBMGraph.from_ebm` select the classes of a multiclass graph, because the text of a model with many classes is long.

- `describe_ebm` and `describe_graph` log an `interactions_skipped` warning that names the interaction terms they skip, because the descriptions cover the main effects only.

- `describe_graph` and `describe_ebm` describe multiclass classifiers. `GraphOptions(classes=[...])` selects the classes to describe.

- `Prediction.margin` is the log-odds of the predicted class: the margin of the decision, which means the same for every model kind.
- `Counterfactual.new_decision` holds the class that the prediction flips to.
- `SufficientReason.minimum` tells whether no smaller set fixes the prediction. It is always true for a binary classifier.
- The proof page of the local explanations defines the counterfactual, the sufficient reason and the completions of regressors and multiclass classifiers.

### Changed

- `EBMGraph.to_text` raises `EBMGraphShapeError` when the scores and the standard deviations do not have the same shape.

- The prompt of `describe_graph` and `describe_ebm` says how to read the graph of a multiclass classifier: one series per class, where only the differences between the classes of a bin change the predicted probabilities. The prompts of binary classifiers and regressors do not change.

- **BREAKING:** The result types of the local explanations have field names that fit every model kind, so the explanations of regressors and multiclass classifiers can come later with no breaking change. The JSON texts for the prompts, and so the explanations of binary classifiers, do not change.

  | Type | Old field | New field |
  | --- | --- | --- |
  | `Prediction` | `label` | `decision` |
  | `Prediction` | `probability` | `output` |
  | `Prediction` | `logit` | `score` |
  | `Prediction` | `logit_std` | `score_std` |
  | `Counterfactual` | `new_probability` | `new_output` |

- The type hints name the real types instead of `object`, so pyrefly and beartype check them: class labels are `ClassLabel`, the values of an instance are `FeatureValue`, and `Contribution.value` and `Change.from_value` are `str`. `ebaml.core.custom_types` defines the aliases.

### Fixed

- The local explanations reject an EBM with two classes whose link is not `logit` with `UnsupportedModelError`, because it is not a binary classifier. Every rejection starts with "Local explanations need a binary classifier."

- A local explanation raises `InvalidInstanceError` for a NaN of a NumPy float type and for `pd.NA`. A NumPy NaN got the last bin, and `pd.NA` raised a `TypeError`. A `Decimal` value is accepted.

<a id='changelog-0.1.1'></a>
## [0.1.1] - 2026-10-27

### Fixed

- `GraphOptions(x_axis_precision=...)` no longer fails on categorical features. The precision rounds interval ends only, and category names stay as they are.

- The graph text escapes the keys as JSON. A category name with a quote or a backslash gave the LLM invalid JSON.

- The simplification of a large graph stops at 10%. It tried 10.1% before it raised the error.

- The completion share of a local explanation counts a completion at a logit of exactly 0 as the negative class. The share was too low for a negative prediction.

- The `usage` log event of `describe_graph` and `describe_ebm` has `truncated_attempts`, the number of attempts that `max_tokens` cut off. The token usage leaves out those attempts, because the provider error carries no count, so it is a lower bound.

<a id='changelog-0.1.0'></a>
## [0.1.0] - 2026-10-26

### Added

- `describe_graph`, `describe_ebm` and `explain_instance` take `model=` to choose the LLM of one call, as `ModelOptions` or
  the `"provider/model"` shorthand. The default model stays OpenRouter's `openrouter/free`.
- `ebaml.llm.options.ModelOptions` sets the provider, the model, the environment variable of the key, the base URL,
  the temperature, the output limit and the headers.
- `benchmark/run.py --model provider/model` runs the benchmark on another model.

- Importing `ebaml.llm.descriptors` runs `baml generate` when the generated `baml_sdk` package is missing. If the package
  was generated by another BAML toolchain than `baml-bridge` expects, the import selects the expected toolchain with
  `baml toolchain use` and generates the package again. If the agent skill check stops `baml generate`, the import runs it
  again with the check off.
- If the import cannot fix the cause, the new `SDKUnavailableError` (also an `ImportError`) lists the steps that fix it:
  install the BAML toolchain, put it on `PATH`, or run `baml generate` in the checkout. It also shows the end of the
  command output.
- `describe_graph`, `describe_ebm` and `explain_instance` raise `MissingAPIKeyError`, `APIKeyRejectedError`,
  `RateLimitedError` or `ProviderUnreachableError` (all `LLMRequestError`) when the provider fails a request for a reason
  that the user can fix. The message names the cause, the environment variable to export where it applies, and the steps
  that fix it. These calls raised the raw `BamlError` for these causes.

### Changed

- The BAML entry points `describe_graph_cot`, `summarize_ebm`, `describe_ebm`, `answer_benchmark_question` and
  `explain_local` take an optional `model: ModelSpec`. The new BAML function `check_model` rejects an unknown provider.
- EBaMl needs the BAML toolchain 0.23 and `baml-bridge` 0.23 or later.

### Fixed

- The benchmark reads a number with no digit before the decimal point (`.5`) as 0.5, not as 5, so `is_correct`
  accepts such an answer.

## [0.0.3] - 2026-10-25

### Added

- The README and the home page of the documentation have a table of the model types that each feature supports:
  binary classifiers, regressors, multiclass classifiers and models with interactions.
- `ebaml.EBaMlError` is the base class of every error that `ebaml` raises on purpose, so `except EBaMlError` catches
  all of them. Every error class of the package derives from it.
- New error classes replace the plain `ValueError` errors: `OptionRangeError`, `InvalidInstanceError`,
  `UnsupportedModelError`, `GraphTextError` and `EBMGraphShapeError`. Each one also derives from `ValueError`. The README section "Errors" lists when each one is raised.
- `EBMGraph.from_text` raises a `GraphTextError` for a line that is not valid JSON. It raised the `JSONDecodeError` of
  `orjson` before, which is also a `ValueError`.

### Fixed

- `explain_instance` raises an `UnsupportedModelError` for a regressor. It raised an `AttributeError` for the missing `classes_`.
- `EBMGraph.to_text` writes the confidence level of the bounds with its decimals, for example `87.5%`. It rounded the
  level to a whole percent before, so `EBMGraph.from_text` computed the standard deviations at the wrong level.
- `EBMGraph.from_ebm` and `EBMGraph.simplify` return an instance of the subclass when you call them on a subclass of
  `EBMGraph`. They returned a plain `EBMGraph` before, so the runtime type check raised an error.
- `parse_str_tuple_to_float_tuple` raises a `ValueError` for a text with more or fewer than two values. It returned a
  tuple of any length before, for example three ends for the interval `"(1, 2, 3)"`.
- `TaskResult.accuracy` is `0.0` for a task with no outcomes. It raised a `ZeroDivisionError` before, and so did
  `format_results`.

### Changed

- **BREAKING:** `facts_to_texts` returns a `FactTexts` object with the fields `prediction`, `counterfactual`,
  `sufficient_reason` and `completions`, instead of a `dict`. `parse_graph_text` returns a `ParsedGraph` object instead of a
  tuple of five values.
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
