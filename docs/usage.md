---
description: The functions of EBaMl that call an LLM, with their options and their errors.
---

# Usage

This page shows the functions that call the LLM, their options and their errors. To install EBaMl and set an API key
first, see [Setup](setup.md).

```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import DescribeEBMRequest, DescribeGraphRequest, PromptOptions, describe_ebm, describe_graph

ebm = ExplainableBoostingClassifier().fit(X, y)

print(describe_graph(DescribeGraphRequest(ebm, feature_index=0)))
print(describe_ebm(DescribeEBMRequest(ebm, num_sentences=30, prompt=PromptOptions(dataset_description="..."))).text)
```

`describe_ebm` returns an `EBMDescription`. Its field `text` is the description. Its fields `skipped_terms` and
`skipped_features` name the interaction terms and the features without a graph that the description leaves out. The text
says so too, and the same names reach the log as `interactions_skipped` and `features_skipped` warnings.

A graph can be turned into text with `EBMGraph.to_text` and back with `EBMGraph.from_text`, which also reads a graph
embedded in a longer prompt. It needs the confidence bounds, and it only reads continuous features.

Each function takes one request: `DescribeGraphRequest`, `DescribeEBMRequest` or `ExplainInstanceRequest`. Three keyword-only fields set the options. `graph=GraphOptions(...)` sets the options of `EBMGraph.from_ebm` and
`EBMGraph.to_text`, for example `max_tokens`. `prompt=PromptOptions(...)` sets the descriptions that the prompts use
(`expert_description`, `graph_description`, `dataset_description` and `task_description`). `model=` selects the LLM: see
[Choosing the model](providers.md). The requests, `GraphOptions`, `PromptOptions` and `ModelOptions` are importable from `ebaml`. Token usage reported by the provider is logged at `INFO` on the `ebaml.llm.descriptors` logger.

A multiclass classifier has one graph series per class. `GraphOptions(classes=[...])` selects the classes to describe,
because the text of a model with many classes is long. In every bin, the scores have the same mean over all the classes
of the model. Only the differences between the classes change the predicted probabilities.
The [multiclass notebook](multiclass-notebook.md) shows each option on the iris data. The [model kinds notebook](model-kinds-notebook.md) compares the graphs and the local
explanations of a binary classifier, a multiclass classifier and a regressor.

<!-- invisible-code-block: python
import numpy as np

y_multiclass = np.select([X["age"] < 30, X["age"] < 60], ["young", "middle"], "old")
-->

```python
from ebaml import GraphOptions

multiclass = ExplainableBoostingClassifier(interactions=0).fit(X, y_multiclass)
print(describe_graph(DescribeGraphRequest(multiclass, feature_index=0, graph=GraphOptions(classes=["young", "old"]))))
```

## Text, vision and mixed modes

`GraphOptions(mode=...)` sets how each graph reaches the LLM. The mode is a member of `GraphMode`, and it applies to
`describe_graph` and `describe_ebm`.

| Mode | The request sends | Model |
| --- | --- | --- |
| `GraphMode.TEXT` (default) | The graph text of `EBMGraph.to_text`. | Every model. |
| `GraphMode.VISION` | A PNG image of the graph, with no simplification, and no graph text. | A model that reads images. |
| `GraphMode.MIXED` | The graph text, and a PNG image of the same simplified graph. | A model that reads images. |

The images need the extra `vision`: `pip install "ebaml[vision]"`. A call in vision or mixed mode without the extra
raises `MissingExtraError` before any request.

```python
from ebaml import GraphMode

print(describe_graph(DescribeGraphRequest(ebm, feature_index=0, graph=GraphOptions(mode=GraphMode.VISION))))
print(describe_graph(DescribeGraphRequest(ebm, feature_index=0, graph=GraphOptions(mode=GraphMode.MIXED))))
```

A picture gives no exact values. In vision mode, the prompt tells the LLM to call each value that it reads from the
image an estimate. In mixed mode, the prompt tells the LLM to take every number from the text and to use the image for
the shape of the graph. Every mode names the interaction terms that the description leaves out.

The image and the text of a graph agree bin for bin. Draw the image yourself with `graph_png` or `graph_svg`.
`GraphImageOptions` sets the title, the title of the score axis, the feature format and the confidence level.

```python
from ebaml import EBMGraph, GraphImageOptions, graph_png, model_kind

graph = EBMGraph.from_ebm(ebm, 0)
png = graph_png(graph, GraphImageOptions(score_axis=model_kind(ebm).score_axis))
```

### A model that reads no images

`GraphOptions(vision_fallback=...)` takes a member of `VisionFallback`. It sets what a call in vision or mixed mode
does when the model reads no images. The
call knows this in one of two ways: `ModelOptions(vision=False)` says so before any request, or the provider rejects
the image.

| `vision_fallback` | The call |
| --- | --- |
| `VisionFallback.RAISE` (default) | Raises `VisionUnsupportedError`. Its message names the model and lists the steps that fix the cause. |
| `VisionFallback.TEXT` | Describes each graph from its text, and emits one `VisionFallbackWarning`. |

The default raises, because a caller who asks for an image must not get a description of the text in silence. The
first model that reads no images decides for the whole call: in `describe_ebm`, the graphs that wait switch to text, or
send no request with `VisionFallback.RAISE`. A rejected request is not retried.

```python
import warnings

from ebaml import ModelOptions, VisionFallback, VisionFallbackWarning, VisionUnsupportedError

text_only = ModelOptions(provider="openai", model="text-only-model", vision=False)
try:
    describe_graph(DescribeGraphRequest(ebm, 0, graph=GraphOptions(mode=GraphMode.VISION), model=text_only))
except VisionUnsupportedError as error:
    print(error)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", VisionFallbackWarning)
    graph = GraphOptions(mode=GraphMode.VISION, vision_fallback=VisionFallback.TEXT)
    print(describe_graph(DescribeGraphRequest(ebm, 0, graph=graph, model=text_only)))
assert [warning.category for warning in caught] == [VisionFallbackWarning]
```

To turn the warning into an error, call `warnings.simplefilter("error", VisionFallbackWarning)`. The
[vision notebook](vision-notebook.md) draws the image of each model kind and compares the three modes.

## Explaining one prediction

```python
from ebaml import DescribeEBMRequest, ExplainInstanceRequest, describe_ebm, explain_instance

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)  # binary classifier, no interactions
global_explanation = describe_ebm(DescribeEBMRequest(ebm)).text  # computed once, prepended to every local prompt

explanation = explain_instance(
    ExplainInstanceRequest(ebm, X.iloc[0].to_dict(), global_explanation, audience="layperson")
)
print(explanation.counterfactual)  # what would have to change to flip the prediction
print(explanation.sufficient_reason)  # which inputs settle the prediction on their own
print(explanation.completions)  # how certain the outcome is as inputs become known
```

The facts are computed exactly from the scores and confidence intervals of the EBM (`ebaml.local`); the LLM only puts
them in words. They are the three queries of [arXiv:2010.12265](https://arxiv.org/abs/2010.12265): minimum change
required, minimum sufficient reason and count completions. `audience="layperson"` (default) gives no model numbers;
`audience="data_scientist"` quotes scores, probabilities and confidence intervals. Subclass `ebaml.Audience` to add a
reader with its own rules. Pass `language="French"` (any language name) to get the explanations in that language. It supports every kind of model
without interactions, and `explanation.facts` holds the numbers behind the text. The [Local explanations](local-explanations.md) page explains what each explanation means, the options and the limits,
and the [local explanations notebook](local-notebook.md) walks through them on the Titanic data.

## Errors

Every error that `ebaml` raises on purpose derives from `ebaml.EBaMlError`. Use `except EBaMlError` to catch the errors
of `ebaml` apart from the errors of other code. Each error for a bad value also derives from `ValueError`, so an
`except ValueError` catches it too.

| Error | `ValueError` | Raised when |
| --- | --- | --- |
| `InvalidOptionError` | yes | An option is not one of its valid values, for example an unknown `Normalization`. |
| `OptionRangeError` | yes | A number option is outside its range, for example `confidence_level=1`. |
| `InvalidInstanceError` | yes | The instance to explain lacks a value, or has a category that the EBM did not see. |
| `UnsupportedModelError` | yes | A local explanation gets an EBM with interactions, or one that is not a classifier or a regressor. A graph gets `classes`, and the model has one output. `model_kind` gets an EBM that is not fitted. |
| `GraphTextError` | yes | `EBMGraph.from_text` cannot read the text of a graph. |
| `EBMGraphError` | no | The base class of the errors of a graph. |
| `EBMGraphShapeError` | yes | The bins of a graph do not fit its scores, the scores and the standard deviations have different shapes, or a boolean format gets more than two bins. A graph image gets continuous bins that do not touch. |
| `EBMGraphTokenLimitError` | no | The text of a graph is over the token budget, and its format cannot be simplified. |
| `EBMGraphSimplificationError` | no | The text of a graph is still over the token budget at the largest simplification. |
| `TokenFitError` | yes | The token weights cannot be fitted from an empty list of samples, or a weight is negative or not finite. |
| `SDKUnavailableError` | no | `baml_sdk` is missing, and `ebaml` cannot run `baml generate`. The error also derives from `ImportError`. |
| `MissingExtraError` | no | A feature needs an extra that is not installed, for example a graph image without the extra `vision`. The error also derives from `ImportError`. |
| `LLMRequestError` | no | The base class of the errors of a provider. Its message lists the steps that fix the cause. |
| `MissingAPIKeyError` | no | A provider needs an API key and the environment has none. |
| `APIKeyRejectedError` | no | A provider answers a request with "unauthorized" or "forbidden". |
| `RateLimitedError` | no | A provider limits the rate of the requests. |
| `ProviderUnreachableError` | no | A request does not reach the provider. |
| `VisionUnsupportedError` | no | A call in vision or mixed mode reaches a model that reads no images, and `vision_fallback` is `VisionFallback.RAISE`. |

`EBMGraphShapeError`, `EBMGraphTokenLimitError` and `EBMGraphSimplificationError` derive from `EBMGraphError`. The five provider errors derive from `LLMRequestError`. A call with `vision_fallback=VisionFallback.TEXT` emits a `VisionFallbackWarning` (a `UserWarning`) in place of `VisionUnsupportedError`. `ebaml` exports all of these classes.

## Behaviour

- **Model:** `openrouter/free` picks a free model for each request, so the model can change between calls and between
  the turns of one conversation. Quality and speed vary with it.
- **Retries:** transient failures (rate limits, network errors, 5xx) are retried up to 3 times. Requests the provider
  rejects are not retried.
- **Concurrency:** `describe_ebm` describes features concurrently, at most `max_concurrency` at a time (default 8).
  Free models are rate limited, so lower it (`describe_ebm(DescribeEBMRequest(ebm, max_concurrency=2))`) if you see rate-limit errors.
- **Progress:** `describe_ebm(DescribeEBMRequest(ebm, progress_bar=True))` shows a progress bar on the error stream. It counts one step for
  each feature and one for the summary. If the error stream is not a terminal, the bar writes one line for each step,
  so a log file or an agent that polls the console sees the progress. The benchmark and `fit_token_weights.py` show a bar
  by default. Use `--no-progress` to turn it off.
- **Token budget:** there is no tokenizer before a call, so the check estimates the tokens of a graph text from its
  words and its digits. The default weights are 4.13 tokens for each word and 0.27 for each digit. They come from 100
  graph texts that one current model counted. The estimate has a margin above the fit, so it is more often too high
  than too low. Other models count other numbers of tokens, so the estimate is a guide. Going a little over
  `max_tokens` only warns. Going more than `overbudget_percent` (default 50) over makes `to_text` simplify the graph, or
  raise if it cannot. The provider's own error is the real limit. `describe_graph`, `describe_ebm` and `explain_instance` pass the
  check. A direct call of `EBMGraph.to_text` checks nothing, unless you pass `check_budget`, for example
  `ebaml.check_token_budget`. That check needs no LLM and no key. To choose the weights, pass a
  `TokenEstimator` instead. To use the weights of each model and to learn them, pass
  `GraphOptions(token_learner=TokenLearner(learning=True))`. The learner starts each model at the weights of a table
  that ships with the package. Then it moves them toward the tokens of the graphs in your real calls. The weights
  persist in `~/.cache/ebaml/token_weights.json`. Learning is off by default, because learned weights change the graph
  text from one call to the next. `scripts/llm/token_weights.py` adds the weights of a new model to the table.

The `model` argument selects the LLM of a call: see [Choosing the model](providers.md).
