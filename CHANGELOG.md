# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- scriv-insert-here -->

<a id='changelog-0.6.7'></a>
## [0.6.7] - 2026-11-10

### Added

- `DescribeGraphRequest`, `DescribeEBMRequest` and `ExplainInstanceRequest` hold the input of `describe_graph`, `describe_ebm` and `explain_instance`. Import them from `ebaml`.

### Changed

- **BREAKING:** `describe_graph`, `describe_ebm` and `explain_instance` take one request object and no other argument. Wrap the old arguments in the matching request: `describe_ebm(DescribeEBMRequest(ebm, model="openai/gpt-4o-mini"))`. The fields keep the names, the order and the defaults of the old arguments, and `graph`, `prompt`, `local`, `model` and `language` stay keyword-only.

<a id='changelog-0.6.6'></a>
## [0.6.6] - 2026-11-09

### Added

- `describe_ebm(ebm, progress_bar=True)` shows a progress bar on the error stream, with one step for each feature and one for the summary. If the stream is not a terminal, the bar writes one line for each step.
- `run_task(..., progress=True)` shows a progress bar of the questions of a benchmark task. `benchmark/run.py` shows a bar by default. Use `--no-progress` to turn it off.

<a id='changelog-0.6.5'></a>
## [0.6.5] - 2026-11-09

### Added

- The `ebaml-chat` command: a chat in the terminal with an agent that explains an EBM. Ask about the model or about one row of the data, and ask follow-up questions. The server starts once and stays up for the whole chat.
- `chat_with_mcp` and `chat_with_tools` in BAML, and `ebaml.llm.descriptors.chat_with_mcp` in Python, for the same chat.

<a id='changelog-0.6.4'></a>
## [0.6.4] - 2026-11-09

### Changed

- Internal loops in graph simplification, local explanations and the MCP model summary use `enumerate` and array conversion in place of index loops. The output stays the same.

<a id='changelog-0.6.3'></a>
## [0.6.3] - 2026-11-09

### Added

- `explain_with_mcp` in BAML: an agent that starts `ebaml-mcp`, lets a language model call its tools, and returns the report with the list of tool calls. `explain_with_tools` runs the same loop on tools that you supply.

<a id='changelog-0.6.2'></a>
## [0.6.2] - 2026-11-09

### Changed

- Local explanation prompts no longer overclaim. A failed counterfactual search says that no realistic change flips the prediction. A missing sufficient reason says that the search found none. Completion shares may fall as inputs become known.

<a id='changelog-0.6.1'></a>
## [0.6.1] - 2026-11-09

### Added

- `LocalOptions.search` chooses the algorithm that solves the covering problems of the sufficient reason with several margins and of the counterfactual of a chosen class: `MipCover` (the default: branch and bound in a MIP solver, with `stop=Stop.OPTIMAL` or `Stop.FIRST` and a `time_limit`), `LocalSearchCover` (a seeded local search from the greedy set) or `GreedyCover`. The last two need no MIP solver.
- `MipBackend` is the protocol of the optimization library behind `MipCover` and the lower bound. `OrToolsBackend` runs OR-Tools with HiGHS, SCIP or CP-SAT (`OrToolsSolver`). Another library plugs in as a class with the same `solve` method.
- `SufficientReason.lower_bound` and `Counterfactual.lower_bound` give the fewest inputs that any set needs, from the LP relaxation. A set whose size equals the bound is marked as the smallest set.
- `Audience.lower_bound` decides whether the facts give the lower bound: `DataScientist` writes `smallest_possible_number_of_changes` and `smallest_possible_number_of_features`, and `Layperson` leaves them out. The prompts tell the LLM that it may state the gap when the set is not proved smallest.
- `SearchLimitError` tells that a solver ended with no set of changes for a chosen class and no proof that none exists, and names the fix.

### Changed

- **BREAKING:** `LocalOptions.reason_search` is replaced by `LocalOptions.search`, and `Counterfactual` and `SufficientReason` have the required field `lower_bound`.
- The counterfactual of a chosen class is the smallest set of changes over every choice of bins when the solver proves it, and `is_smallest_set` is then true.
- **BREAKING:** `Audience` has the abstract method `lower_bound`, so a custom audience must define it.
- When `is_smallest_set` is false, the prompts tell the LLM to describe the set without a word about its size, in place of calling it minimal.
- The package depends on `ortools`, and the lock moves `protobuf` to 6.33.

### Removed

- **BREAKING:** `ebaml.local.reason_search` with `ReasonSearch`, `Minimal`, `ExactOrMinimal` and `Selection`. Use a `CoveringSolver` in `LocalOptions.search`.

<a id='changelog-0.6.0'></a>
## [0.6.0] - 2026-11-08

### Added

- `LocalOptions.target_class` asks, for a multiclass classifier, for the changes after which a chosen class is the predicted class. `reason_search` chooses the features. The result is never marked as the smallest set, and it has no proof.

### Changed

- The counterfactual prompt tells the LLM when the set of changes is not the smallest, and that a flip of a class means that another class becomes the predicted class. The counterfactual facts have the key `is_smallest_set`, and the key `minimum_number_of_changes` is now `number_of_changes`.
- `Counterfactual.minimum` is false for the counterfactual of a chosen target class.

<a id='changelog-0.5.2'></a>
## [0.5.2] - 2026-11-08

### Added

- `explain_instance`, `compute_local_facts` and the MCP tool `explain_prediction` explain a prediction of a multiclass classifier without interactions. The decision is the predicted class. The counterfactual is the fewest changes after which another class has a higher score, and the sufficient reason keeps the class against every other class. `LocalOptions.reason_search` chooses the search of the sufficient reason. The Lean proof of these facts is not done yet.
- `OutputKind.CLASS_PROBABILITY` names the output of a multiclass classifier. The texts for the data scientist call the score of the predicted class `class_score`.

<a id='changelog-0.5.1'></a>
## [0.5.1] - 2026-11-07

### Changed

- The `mcp` extra requires `mcp>=2.3.0`, and `ebaml-mcp` runs on the `MCPServer` class of `mcp` 2. The tool names and
  argument schemas are the same.

- The lower bounds of the dependencies are the releases that the tests cover: `narwhals>=2.28.0`, `pandas>=3.1.0`, and,
  for the `mcp` extra, `cyclopts>=5.3.2`, `joblib>=1.6.0` and `polars>=2.1.0`.

### Security

- The `mcp` extra and the project environment hold `mcp` 2.3.0 and the fixed `opentelemetry` releases, because the
  dependency overrides that held semgrep's old pins are gone.

<a id='changelog-0.5.0'></a>
## [0.5.0] - 2026-11-07

### Added

- A Lean proof and a paper cover the local explanations of regressors: the shift of the intercept, the counterfactual of the answer "inside an interval", and the two searches of the sufficient reason (`Minimal` and `ExactOrMinimal`). The "Proof of the local explanations" page states what the proof shows and what it leaves out.

### Fixed

- A fresh `pip install ebaml` no longer installs a `baml-bridge` that rejects the generated `baml_sdk` with a version
  skew error. `ebaml` now requires `baml-bridge==0.23.0`, the version of the toolchain that generates the SDK.

- The wheel and the source distribution no longer contain the `.gitignore` file that `baml generate` writes into
  `baml_sdk`.

- `pip install "ebaml[mcp]"` no longer installs `mcp` 2.x, which has no `FastMCP` and made `ebaml-mcp` fail at start-up.
  The `mcp` extra now requires `mcp>=1.30.0,<2`.

<a id='changelog-0.4.4'></a>
## [0.4.4] - 2026-11-07

### Added

- `explain_instance` and `compute_local_facts` explain a prediction of a regressor. `LocalOptions(decision=...)` sets the question about the predicted value: `Threshold(value)`, or `Interval(low, high)` for a prediction below or above the interval. The default asks whether the value is above the average prediction. The sufficient reason and the counterfactual of a regressor have no proof yet, and an interval that contains the prediction raises `UnsupportedModelError`.
- The MCP tool `explain_prediction` explains a regressor.

- A regressor can explain the answer "inside" of `Interval(low, high)`. `LocalOptions.reason_search` chooses how to find the sufficient reason of such a decision: `Minimal()` (the default) returns a set from which no input can be removed, and `ExactOrMinimal()` returns the smallest set up to a limit on the number of features.
- The sufficient reason text says whether the set is the smallest or only minimal, and the prompt tells the LLM to use the matching word.

### Changed

- **BREAKING:** `Audience.class_output` and `Audience.margin` take the `OutputKind` of the model as a second argument. A custom audience must add it. `Prediction` has two new fields, `threshold` and `output_kind`, and `points_towards` reads the threshold.
- The error for a multiclass classifier says: "Local explanations need a binary classifier or a regressor."
- The local prompts tell the LLM that a prediction can be a number, and the layperson guidance allows the predicted value and the limits of the question.

- **BREAKING:** `Interval` is open at the low end and closed at the high end, as `Threshold` is. The answers are "at most `low`", "above `low` and at most `high`" and "above `high`". `Side` holds a tuple of `Margin` objects in place of one sign and one threshold.

<a id='changelog-0.4.3'></a>
## [0.4.3] - 2026-11-06

### Added

- `ebaml-mcp` takes `--host` and `--port` for the `streamable-http` transport. The defaults are `127.0.0.1` and `8000`.

<a id='changelog-0.4.2'></a>
## [0.4.2] - 2026-11-06

### Fixed

- `ebaml-mcp` writes its log events to stderr. Before, they went to stdout and could corrupt the JSON-RPC stream of the stdio transport.
- The `max_tokens` argument of the `get_graph` tool takes effect. The server merges the bins of a continuous graph until its estimated size fits the budget, and it raises an error if the graph cannot fit.

<a id='changelog-0.4.1'></a>
## [0.4.1] - 2026-11-05

### Added

- `ebaml-mcp --model` loads a skops file as well as a joblib file, and reads the format from the file. The server trusts only the EBM classes, so a skops file runs no code.

### Changed

- The MCP server reads the `--data` CSV file with polars, and `ModelStore` accepts a frame of any library that narwhals supports (pandas, polars, pyarrow). ebaml no longer imports pandas outside the type of `pd.NA`.

- `ebaml-mcp` and `benchmark/run.py` use cyclopts for the command line. The flags stay the same, and the `mcp` extra installs cyclopts. A wrong value now gives a usage error with the list of valid choices.

<a id='changelog-0.4.0'></a>
## [0.4.0] - 2026-11-05

### Added

- The `ebaml-mcp` command serves a fitted EBM as an MCP server (install the `mcp` extra). An agent can read the model and its graphs, and explain one prediction with the exact local facts, with no LLM key. With `--llm`, it can also ask for LLM descriptions.

<a id='changelog-0.3.3'></a>
## [0.3.3] - 2026-11-03

### Added

- `ebaml` now exports the functions, the option classes, the local-explanation facts and the audience helpers that the docs use, so `from ebaml import describe_ebm` works. `import ebaml` still loads no BAML code: `describe_graph`, `describe_ebm`, `explain_instance` and `LocalExplanation` load on first use.

<a id='changelog-0.3.2'></a>
## [0.3.2] - 2026-11-02

### Added

- `TokenEstimator` estimates the tokens of a graph text from its words, bytes and digits, and is a `BudgetCheck`. It can learn its weights from the token counts of real calls with `observe`.
- `GraphOptions(token_learner=TokenLearner(learning=True))` makes `describe_graph` and `describe_ebm` check the token budget with the weights of the model that you chose, and learn the weights of each model that answers. The learned weights persist in `~/.cache/ebaml/token_weights.json`. Learning is off by default.
- A table of token weights for each model ships with the package (`src/ebaml/llm/token_weights.json`), and `scripts/fit_token_weights.py` adds a new model to it.
- `sample_graph_tokens` sends one request with a graph text and returns its token samples, at a small cost.
- `TokenUsage.samples` lists the token samples of the calls of a run. Each sample has the provider, the model, the request body and the input tokens. It never holds a header.
- `TokenFitError` is raised when token weights cannot be fitted or read.

### Changed

- The default token estimate of a graph text is now 4.13 tokens for each word and 0.27 for each digit, with a margin above the fit, and no longer 1 token for 3 characters. The old rule gave 43% too few tokens on real graph texts. Graphs near `max_tokens` are now simplified at a different size, so the text that the LLM reads can differ.

### Removed

- **BREAKING:** The BAML functions `estimate_tokens` and `check_token_budget` and the BAML class `TokenBudget` are gone. Use `ebaml.graphs.token_budget` in Python.
- **BREAKING:** `TokenUsage` has a new required field `samples`, so code that builds a `TokenUsage` must pass it.

<a id='changelog-0.3.1'></a>
## [0.3.1] - 2026-11-01

### Added

- `explain_instance` takes `language=` (default `"English"`), so the three local explanations come in the language of the reader. It works with every audience.

### Changed

- The `Audience` docstring and the guide show an executive audience in place of a French one. Use `language=` to change the language.

<a id='changelog-0.3.0'></a>
## [0.3.0] - 2026-11-01

### Added

- `Audience` is a base class that a user subclasses to explain a prediction to a new kind of reader. A subclass sets the instruction for the LLM (`guidance`) and the keys that the facts of the prompt get. `Layperson` and `DataScientist` are the two built-in audiences, and a subclass of either changes one rule. `explain_instance` and `facts_to_texts` accept an `Audience` object.
- The module `ebaml.local.audience` has the helpers of the built-in audiences: `likelihood_word`, `strength_word`, `round_value` and `points_towards`.

### Changed

- **BREAKING:** `Audience` is no longer an enum. Use `Layperson()` and `DataScientist()` in place of `Audience.LAYPERSON` and `Audience.DATA_SCIENTIST`. The names `"layperson"` and `"data_scientist"` still work as strings.
- **BREAKING:** The BAML functions `explain_local`, `LocalCounterfactual`, `LocalSufficientReason`, `LocalCompletions` and `local_user_msg` take the text `audience_guidance: string` in place of the name of the audience, and `audience_guidance` is no longer a BAML function.

<a id='changelog-0.2.6'></a>
## [0.2.6] - 2026-10-31

### Changed

- The counterfactual facts that the local explanation prompts read put the class after the changes and its likelihood (or probability, for a data scientist) in one object, `after_the_changes`, with the same field names as the prediction. With the probability in a key of its own, a model read it as the probability of the old class.

<a id='changelog-0.2.5'></a>
## [0.2.5] - 2026-10-31

### Added

- `ebaml.graphs.token_budget.check_token_budget` checks the token budget of a graph text in Python, with no LLM and no key. Pass it as `check_budget` to `EBMGraph.to_text`, which then simplifies a graph that is too large. `estimate_tokens` gives the estimate alone. Both follow the rule of the BAML functions of the same name, so you do not need to import the generated `baml_sdk`.

<a id='changelog-0.2.4'></a>
## [0.2.4] - 2026-10-31

### Changed

- The default output limit of a call is 12000 tokens, and 16000 for the first turn of a conversation and for the benchmark. A reasoning model counts its hidden reasoning in this limit, and at 6000 and 8000 tokens some replies ended at the limit with no answer.

- The counterfactual facts that the local explanation prompts read name the class that the prediction changes to (`new_predicted_class`), and the likelihood of that class is `likelihood_of_new_predicted_class` (`probability_of_new_predicted_class` for a data scientist). The LLM no longer has to guess which class the likelihood is about.

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
