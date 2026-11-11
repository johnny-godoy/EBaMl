# Local explanations

`describe_ebm` explains how an EBM works *in general*. A local explanation answers a different question: **why did the
model predict this for this one input?**

```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import DescribeEBMRequest, ExplainInstanceRequest, PromptOptions, describe_ebm, explain_instance

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)  # a binary classifier, no interactions
global_explanation = describe_ebm(DescribeEBMRequest(ebm, prompt=PromptOptions(dataset_description="..."))).text  # once per model

explanation = explain_instance(ExplainInstanceRequest(
    ebm,
    X.iloc[0].to_dict(),      # the value of every feature, by name
    global_explanation,
    audience="layperson",     # or "data_scientist", or your own Audience
))
print(explanation.counterfactual)
print(explanation.sufficient_reason)
print(explanation.completions)
```

A worked example, with its output, is in the [local explanations notebook](local-notebook.md).

## Facts are computed, words are generated

The design has one rule: **nothing that can be computed is left to the LLM.**

```text
EBM + one input
      │  compute_local_facts (Python, deterministic)
      ▼
Facts: prediction, counterfactual, sufficient reason, completions
      │  facts_to_texts(audience)
      ▼
One JSON text per explanation ──┐
                                ├──▶ BAML explain_local (3 LLM calls) ──▶ LocalExplanation
Global explanation (describe_ebm) ┘                                         (3 texts + the facts)
```

1. `compute_local_facts` reads the scores and confidence intervals of the EBM and works out the facts. It is
   deterministic: the same model and input always give the same facts.
2. `facts_to_texts` renders each fact as JSON for the audience. This is where numbers are
   removed for a layperson.
3. `explain_local` (BAML, `baml_src/local_explanations.baml`) makes three LLM calls concurrently, one per
   explanation. Each prompt starts with the **global explanation** of the model, so the LLM knows what the features
   mean, then the prediction, a question, and the facts.

The result is a `LocalExplanation` with the three texts and `facts`, the numbers behind them, so any sentence can be
checked against the data.

## The three explanations

They are the explainability queries of Barceló, Monet, Pérez and Subercaseaux,
[*Model Interpretability through the Lens of Computational Complexity*](https://arxiv.org/abs/2010.12265). An EBM without interactions predicts `intercept + Σ score_i(value_i)`, one independent effect per feature, which
makes the first two queries solvable exactly: the [proof](local-explanations-proof.md) shows that the sorting rules below find the smallest set.

| Explanation | Query in the paper | Question | How it is computed |
| --- | --- | --- | --- |
| `counterfactual` | Minimum change required | What is the *fewest* inputs to change so that the prediction flips? | Exact when `minimality` proves it. |
| `sufficient_reason` | Minimum sufficient reason | What is the *fewest* inputs that settle the prediction, whatever the other inputs are? | Exact when `minimality` proves it. |
| `completions` | Count completions | If only some inputs are known, what share of the possible completions get this prediction? | Estimated by sampling. |

Each result names the guarantee that it has. `minimality` says why the set is, or is not, known to be the smallest.
Sorting proves it for a decision with one margin: a binary classifier, and a regressor with a threshold. A decision with several margins (a
multiclass classifier, the answer "inside" of an interval) and the counterfactual of a target class use a search, and
`minimality` is then a proof only if the search proves it. The two sections below describe the case with one margin.
[Regressors](#regressors) and [Multiclass classifiers](#multiclass-classifiers) describe the search.

| `MinimalityStatus` | Meaning |
| --- | --- |
| `SORTING_PROOF` | One margin. Sorting finds the smallest set, and the [proof](local-explanations-proof.md) covers it. |
| `SOLVER_PROOF` | The MIP solver proved that no smaller set is sufficient. The proof rests on the certificate of the solver. |
| `AT_LOWER_BOUND` | The set has as many members as the lower bound, so no smaller set exists. The bound rests on the LP solver. |
| `UNPROVED` | The search found a set, but it did not prove that the set is the smallest. A smaller set can exist. |

`MinimalityStatus.is_smallest` is false only for `UNPROVED`. The prompts read this boolean as `is_smallest_set`.

### Counterfactual

For each feature, take its best alternative value: the one that lowers the score in favour of the current prediction
the most. Sort these drops from largest to smallest and apply them until the prediction flips. Because the effects
add up independently, the *k* largest drops are the best *k* changes, so the first set that flips is the smallest
possible. If even changing every feature does not flip it, `facts.counterfactual` is `None`.

### Sufficient reason

With nothing known, the worst case puts every feature at the value that works hardest against the prediction.
Knowing a feature's value removes that worst case for it and "guarantees" a gain. Sort the gains from largest to
smallest and fix features until the worst case still keeps the prediction. That set is the smallest possible, and
**every other feature is irrelevant to the outcome**: whatever values it takes, the prediction cannot change. If even
fixing all the features does not guarantee the prediction (the model is too uncertain), `facts.sufficient_reason` is
`None`.

### Completions

Reveal the features one at a time, in the same order as the sufficient reason, and for each step estimate the share of
completions that get the same prediction. Step 0 is the base rate with nothing known. Only the first `max_steps`
features are revealed. If all of them are, the last share is 100% by definition (every input known).

Each share is an estimate from `n_samples` random completions, under these assumptions:

- **Independent inputs.** Each unknown feature is sampled alone, so a completion can be a combination that the data
  does not contain.
- **Training weights.** A bin is drawn with its share of the training data of the EBM (`bin_weights_`).
- **Every bin.** The sample uses all the bins of a feature. The counterfactual and the sufficient reason use only the
  values that `min_support` allows, so their domain is smaller.
- **Fitted scores.** The sample uses the scores as the model fitted them. `confidence_level` does not change a share.
- **A fixed seed.** The same `seed` gives the same shares. Another seed, or another `n_samples`, gives another
  estimate of the same share.

A share can fall when an input becomes known: the value of the instance can be less favourable than most values of
that input. The sampling error of a share `p` is about `sqrt(p * (1 - p) / n_samples)`. Unlike the other two,
**these are estimates and must be described as approximate.**

## Confidence

The EBM stores a standard deviation for every score. `confidence_level` (default `0.95`) gives a number `z` of
standard deviations. Each claim has its own uncertainty model: it names the scores that move by `z` deviations
**against the claim being made**.

| Claim | What moves by `z` deviations |
| --- | --- |
| Counterfactual | The effect of the new value of each changed feature, against the flip. The other features keep their fitted effects. |
| Sufficient reason | Every effect, against the prediction: a fixed feature at its value, and a free feature at its worst value. |
| Interval of the probability or of the value | The total score. Its deviation (`score_std`) is that of a sum of independent errors. |
| Margin of a multiclass classifier | The difference of two class scores. Its deviation is the sum of the two deviations, because an EBM stores no covariance. |

A higher `confidence_level` makes a counterfactual and a sufficient reason harder to find: the set gets larger, or the
search finds none. It never turns a failed claim into a valid one. A claim holds under its own model only: a
counterfactual does not cover the uncertainty of the features that it leaves unchanged. `confidence_level=0` moves
no score, so no claim covers the uncertainty of the model.

## The facts

`explanation.facts` is a `LocalFacts`. Its fields have names that fit every model kind:

| Field | Meaning for a binary classifier | Meaning for a regressor |
| --- | --- | --- |
| `prediction.decision` | The predicted class. | The answer to the question about the value, such as `"above 10"`. |
| `prediction.output`, `lower`, `upper` | The probability of the predicted class, and its confidence interval. | The predicted value, and its confidence interval. |
| `prediction.score`, `score_std` | The log-odds of the second class, and its standard deviation. | The predicted value, and its standard deviation. |
| `prediction.margin` | The log-odds of the predicted class. It is positive for every prediction. | The distance of the value from the threshold, on the side of the decision. |
| `prediction.threshold` | `0`: the log-odds at which the class changes. | The threshold of the question. |
| `prediction.output_kind` | `OutputKind.PROBABILITY` | `OutputKind.VALUE` |
| `counterfactual.changes` | The changes, as `Change(feature, from_value, to_value, score_change)`. | The same. |
| `counterfactual.new_decision`, `new_output` | The class after the changes, and its probability. | The opposite answer, and the predicted value after the changes. |
| `sufficient_reason.features` | The features that fix the prediction. | The features that fix the answer. |
| `sufficient_reason.guaranteed_margin` | The worst case of the log-odds of the predicted class with these features fixed. | The worst case of the distance from the threshold, with these features fixed. |
| `counterfactual.minimality`, `sufficient_reason.minimality` | A `MinimalityStatus`: `SORTING_PROOF`, `SOLVER_PROOF`, `AT_LOWER_BOUND` or `UNPROVED`. It is always `SORTING_PROOF` for one margin. | The same. It can be `UNPROVED` for the answer "inside" of an interval. |
| `counterfactual.lower_bound`, `sufficient_reason.lower_bound` | No set has fewer members than this number. | The same. |
| `completions` | One `CompletionStep(fixed, share)` for each step. | The same. |

<!-- invisible-code-block: python
from ebaml import compute_local_facts

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
instance = X.iloc[0].to_dict()
-->

```python
facts = compute_local_facts(ebm, instance)
print(facts.prediction.decision, round(facts.prediction.output, 2))
if facts.counterfactual is not None:
    print(facts.counterfactual.new_decision, [change.feature for change in facts.counterfactual.changes])
```

## Regressors

A regressor predicts a number, so the user says which question about the number to explain. `LocalOptions.decision` takes
a `Decision`:

| Decision | Question | Answers |
| --- | --- | --- |
| `Threshold(value)` | Is the predicted value above `value`? | "above `value`", "at most `value`". |
| `Interval(low, high)` | Is the predicted value at most `low`, inside the interval, or above `high`? | "at most `low`", "above `low` and at most `high`", "above `high`". |
| `None` (default) | Is the predicted value above the intercept? | The intercept is the average prediction on the training data. |

The counterfactual is the fewest changes after which the answer is another one. The sufficient reason is a set of
inputs that keep the answer for every value of the other inputs. A classifier accepts no `decision`, because its
predicted class is the decision.

The answer "inside" has two margins, one for each end of the interval. The counterfactual is the smaller of the two
answers for the margins, and it is exact. The sufficient reason must keep both margins positive at the same time.
Sorting does not find the smallest set for that. The problem is a covering problem, and no fast exact algorithm for
it exists unless P = NP ([proof page](local-explanations-proof.md)). `LocalOptions.search` chooses the algorithm. Each
class is one algorithm:

| Solver | Algorithm | `minimality` is a proof when |
| --- | --- | --- |
| `MipCover(backend=OrToolsBackend(), stop=Stop.OPTIMAL, time_limit=10.0)` (default) | Branch and bound over all sets in a MIP solver. `OrToolsBackend(OrToolsSolver.SCIP)` or `OrToolsSolver.CP_SAT` choose another solver than HiGHS. `stop=Stop.FIRST` ends at the first set smaller than the greedy set. | The solver proves the optimum, or the size equals the lower bound. |
| `LocalSearchCover(iterations=100, destroy_share=0.1, seed=None)` | Starts from the greedy set, removes a random part of it, builds it again with the greedy, and keeps the smallest set. `None` uses `LocalOptions.seed`. It needs no MIP solver. | The size equals the lower bound. |
| `GreedyCover()` | The greedy algorithm of set cover, then the removal of each input that the set does not need. It needs no MIP solver and no time limit. | The size equals the lower bound. |

Every solver checks the empty set first and reports a lower bound, `lower_bound`: no set has fewer inputs. The bound
comes from the LP relaxation of the problem, which the backend solves. `GreedyCover(bound_backend=None)` and
`LocalSearchCover(bound_backend=None)` run no LP, and their bound is `1`. If the time limit of `MipCover` ends before
the solver proves the optimum, it returns the best set that it found, and `minimality` is `UNPROVED` unless the set
reaches the bound. A decision with one margin ignores `search`, because sorting finds its smallest set exactly. When `minimality` is
`UNPROVED`, the prompt tells the LLM not to call the set the smallest, and a data scientist gets the lower bound in the
facts.

The MIP solver is behind the `MipBackend` protocol: a class with a method `solve(model, settings)` that solves a
`CoveringModel` and returns a `ModelResult`. `OrToolsBackend` is the backend that ships with the package. To use another
optimization library, write a class with this method and pass it as `MipCover(backend=...)`.

<!-- invisible-code-block: python
from ebaml import Threshold, LocalOptions
from interpret.glassbox import ExplainableBoostingRegressor

regressor = ExplainableBoostingRegressor(interactions=0).fit(X, X["age"] * 0.5 + X["fare"] * 0.01)
-->

```python
facts = compute_local_facts(regressor, instance, LocalOptions(decision=Threshold(30.0)))
print(facts.prediction.decision, round(facts.prediction.output, 1))
```

The answer "inside" of an interval uses the solver. This example asks for the local search, which needs no MIP solver
and proves less:

```python
from ebaml import Interval, LocalSearchCover

options = LocalOptions(decision=Interval(20.0, 40.0), search=LocalSearchCover())
reason = compute_local_facts(regressor, instance, options).sufficient_reason
if reason is not None:
    print(reason.features, reason.minimality, reason.lower_bound)
```

`explain_instance` takes the same options with `local=LocalOptions(decision=Threshold(30.0))`. The explanations of a
regressor use the same sorting algorithms as the explanations of a classifier. The
[proof page](local-explanations-proof.md) says which claims have a proof.

## Multiclass classifiers

A multiclass classifier has one score for each class. The decision is the predicted class `c`. The class holds while its
score is above the score of every other class `j`, so each pair `(c, j)` is one margin: the score of `c` minus the score
of `j`. Each margin is a sum of one term for each feature, as the margin of a binary classifier is.

- The counterfactual is the fewest changes after which another class has a higher score than `c`. It is the smallest of
  the answers for the margins, so sorting finds it for each margin. `new_decision` is the class that is predicted
  after the changes.
- The sufficient reason must keep every margin positive at the same time, so `LocalOptions.search` chooses the
  algorithm, as for the answer "inside" of a regressor. `MipCover()` is the default.
- The completions are estimated by sampling, as for every kind.
- `LocalOptions.target_class` asks for the fewest changes after which a chosen class is the predicted class. All the
  margins of that class must then hold at once. Each plausible bin of a feature is a possible change, and a set of
  changes moves each changed feature to one bin. `LocalOptions.search` finds the set over every choice of bins. With
  `MipCover()`, the set is the smallest unless the time limit ends first. If the solver finds no set in its time limit
  and cannot prove that none exists, `compute_local_facts` raises a `SearchLimitError`, which names the fix.
  `GreedyCover` and `LocalSearchCover` raise it too when a change that lowers a margin stops the greedy.
- EBMs store no covariance between the scores of two classes. The confidence interval of a margin therefore uses the sum
  of the two standard deviations, which is an upper bound for any correlation.

<!-- invisible-code-block: python
import numpy as np

y_multiclass = np.select([X["age"] < 30, X["age"] < 60], ["young", "middle"], "old")
multiclass = ExplainableBoostingClassifier(interactions=0).fit(X, y_multiclass)
-->

```python
facts = compute_local_facts(multiclass, instance)
print(facts.prediction.decision, round(facts.prediction.output, 2))
print(facts.sufficient_reason)
```

To ask for the changes that lead to another class, pass that class as `target_class`:

```python
other = next(label for label in multiclass.classes_ if label != facts.prediction.decision)
print(compute_local_facts(multiclass, instance, LocalOptions(target_class=other)).counterfactual)
```

A multiclass classifier accepts no `decision`. The [proof page](local-explanations-proof.md) lists the claims that the
proof covers for a multiclass classifier. On an exact tie between the two highest classes, the proof does not apply.

## Audience

An audience is an instance of a subclass of `Audience`. Two audiences ship with the package. Pass one as an object, or
pass its name as a string:

| `audience` | What the LLM is given | What it is told |
| --- | --- | --- |
| `"layperson"` or `Layperson()` (default) | Words instead of model numbers: effects are `strongly`/`moderately`/`slightly` `towards`/`against`, probabilities are `very likely`, `unlikely`, ... Values of the person's own inputs (their age) stay. | No numbers that describe the model, no jargon. |
| `"data_scientist"` or `DataScientist()` | Scores, standard deviations, the probability and its confidence interval, the logit, the confidence level. | Quote the numbers, and say how uncertain a claim is. |

Removing the numbers *before* the prompt is deliberate: an instruction not to repeat a number is weaker than not
having the number. The name is validated, so `audience="child"` raises `InvalidOptionError` before any LLM call.

### Language

The audience decides who reads the explanation. The `language` argument decides in which language the reader gets it.
Pass any language name that the LLM knows. The facts, the numbers and the audience rules stay the same, so you can
combine every audience with every language:

```python
explanation = explain_instance(ExplainInstanceRequest(ebm, X.iloc[0].to_dict(), global_explanation, audience="layperson", language="French"))
```

The [languages notebook](languages-notebook.md) shows one passenger in several languages. The quality of the text depends on how well the LLM writes the language. To change the language and the rules of the
reader together, write your own audience (below).

### Your own audience

An audience has two jobs, and the base class `Audience` has one method for each part of them:

- `guidance()` returns the instruction that tells the LLM how to write for this reader, and why.
- `contribution`, `prediction`, `change`, `flip`, `class_output`, `margin`, `guarantee`, `support`, `lower_bound`,
  `share` and `sample` return the keys that the audience adds to one kind of fact. `facts_to_texts` builds the keys
  that all audiences share (the names of the features, their values).
- `flip`, `guarantee`, `support` and `sample` state the conditions of the facts: whether the scores moved by their
  uncertainty (`confidence_level`), whether rare values were left out (`min_support`), and the size of the sample
  (`n_samples`). The prompts read these keys. The text then claims a condition only when the options set it.

All the methods are abstract, so the author decides on every model number. To change one rule of a built-in audience,
subclass it and override that method. This audience keeps the words of the layperson and writes for a chief executive:

```python
from typing import override

from ebaml import Layperson


class Executive(Layperson):
    @override
    def guidance(self) -> str:
        return (
            "The reader is a chief executive who has one minute, so open with the bottom line in one sentence. "
            "Then name the one or two inputs that matter most, because the reader decides on the result and not "
            "on the method. Use business words, and mention only values of the person's own inputs."
        )
```

The same way, `Layperson` with a new `guidance` gives a child (ELI5), a reader of Simplified Technical English
(ASD-STE100) or any other role. Start from `DataScientist` for a role that reads numbers, such as a machine learning
engineer.

An audience with new rules overrides the methods of the facts. This one is for a loan officer: it gives the probability
as a percentage in place of the probability, the confidence interval and the logit, and it drops the scores of the
contributions. The changes, the flip and the margin keep the numbers of the data scientist.

```python
from ebaml.core.custom_types import JSONValue
from ebaml import DataScientist, round_value


class LoanOfficer(DataScientist):
    @override
    def guidance(self) -> str:
        return "The reader is a loan officer, so quote percentages and skip the statistics of the model."

    @override
    def contribution(self, contribution, prediction) -> dict[str, JSONValue]:
        return {}

    @override
    def prediction(self, prediction, confidence_level) -> dict[str, JSONValue]:
        return {"percent": round_value(100 * prediction.output)}
```

`ebaml` also exports the helpers of the built-in audiences: `likelihood_word`, `strength_word`,
`sample_size_word`, `round_value` and `points_towards`. Pass the object to `explain_instance(ExplainInstanceRequest(..., audience=LoanOfficer()))`. The
[custom audiences notebook](audiences-notebook.md) walks through six audiences (a child, a chief executive, a reader of Simplified Technical English, a loan officer, a machine learning engineer and a regulator) and shows their facts side by side.

You can see exactly what the LLM reads, without calling it:

<!-- invisible-code-block: python
instance = X.iloc[0].to_dict()
-->

```python
from ebaml import DataScientist
from ebaml import compute_local_facts
from ebaml import facts_to_texts

facts = compute_local_facts(ebm, instance)
texts = facts_to_texts(facts, DataScientist())
print(texts.counterfactual)
```

```json
{
  "flip_possible": true,
  "changes": [
    {"feature": "sex", "from": "male", "to": "female", "score_change": 2.437}
  ],
  "number_of_changes": 1,
  "is_smallest_set": true,
  "after_the_changes": {"predicted_class": 1, "probability": 0.589},
  "flip_holds_at_confidence": 0.95,
  "rare_values_left_out": true,
  "min_support": 0.01
}
```

For a layperson, the same fact has no scores and no confidence level, and the probability becomes a word:

```json
{
  "flip_possible": true,
  "changes": [{"feature": "sex", "from": "male", "to": "female"}],
  "number_of_changes": 1,
  "is_smallest_set": true,
  "after_the_changes": {"predicted_class": 1, "likelihood": "about as likely as not"},
  "flip_holds_even_if_the_model_is_somewhat_off": true,
  "rare_values_left_out": true
}
```

(The example is passenger 0 of the notebook: a 35-year-old man in third class, with a survival probability of 0.11.)

## Options

The function takes one `ExplainInstanceRequest(ebm, instance, global_explanation, audience, num_sentences, *, language, local, prompt, model)`.
The fields `language`, `local`, `prompt` and `model` are keyword-only. `local` is a `LocalOptions` from `ebaml`. `prompt` is a
`PromptOptions` from `ebaml`. The table shows where each setting goes:

| Field | Default | Meaning |
| --- | --- | --- |
| `audience` | `"layperson"` | An `Audience`, or the name `"layperson"` or `"data_scientist"`. See [Audience](#audience). |
| `num_sentences` | `5` | The most sentences in each of the three explanations. |
| `language` | `"English"` | The language of the explanations. See [Language](#language). |
| `local.confidence_level` | `0.95` | See [Confidence](#confidence). In `[0, 1)`. |
| `local.min_support` | `0.01` | Share of the training data a value needs to be offered as an alternative. A category needs this share; for a continuous feature the lowest and highest `min_support` of the data are trimmed. It keeps freak values with extreme scores out of the counterfactual and out of the worst case of the sufficient reason. `0` allows every value. |
| `local.max_steps` | `5` | The most features revealed one by one in `completions`. |
| `local.n_samples` | `10000` | Samples used for `completions`. |
| `local.seed` | `0` | Seed of the sampling, so that the facts are reproducible. |
| `prompt.expert_description`, `prompt.dataset_description` | see `default_graph_options` | Used in the prompts. A dataset description helps. |

`explain_instance` takes `model=` like `describe_ebm`: see [Choosing the model](providers.md). Token usage of the three calls is logged at `INFO` on the
`ebaml.llm.descriptors` logger as `usage`.

To compute the facts alone, with no LLM and no key, call `compute_local_facts(ebm, instance, LocalOptions(...))`.

## What the guarantees mean, and what they do not

"Exact" has a precise meaning here, and the prompts tell the LLM so:

- **It is exact for this model, not for the real world.** The sufficient reason says nothing about what would happen
  to a real person, only that the EBM's prediction cannot change.
- **It is exact over the values allowed by `min_support`**, and over the confidence level. `confidence_level=0` and
  `min_support=0` give the paper's definition over all values; the notebook checks the sufficient reason by random sampling, and the unit tests check both exact explanations against enumeration of every input.
- **The free features vary independently.** A sufficient reason covers *every* combination of the remaining values,
  including combinations that never occur in the data (a first-class passenger who paid the cheapest fare). That is
  what makes it a guarantee, and also why it can need more features than intuition suggests.
- **`completions` is an estimate** under the assumptions of [Completions](#completions): independent inputs, the
  training weights of every bin, the fitted scores and a fixed seed.

## Limits

- **Classifiers and regressors only.** Another kind of model raises `UnsupportedModelError`, which is a `ValueError`. The
  [definitions for every model kind](local-explanations-proof.md#definitions-for-every-model-kind) say what the
  explanations of each kind mean. The [model kinds notebook](model-kinds-notebook.md) shows the fields of the result
  types.
- **No interactions.** Fit with `interactions=0`. With pairwise terms the effects no longer add up per feature, so the
  greedy solutions would not be exact; the function raises `ValueError` instead of answering approximately.
- **Every input is needed.** A missing value, a feature name that is absent or a category not seen in training raises
  `ValueError`. Impute, or drop the row.
- **The words are not guaranteed.** The facts are exact; the LLM can still misread them. The prompts forbid inventing
  values and tell it which claims are guarantees and which are estimates, and `explanation.facts` lets you check.
