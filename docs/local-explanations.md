# Local explanations

`describe_ebm` explains how an EBM works *in general*. A local explanation answers a different question: **why did the
model predict this for this one input?**

```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml.llm.descriptors import describe_ebm, explain_instance
from ebaml.llm.options import PromptOptions

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)  # a binary classifier, no interactions
global_explanation = describe_ebm(ebm, prompt=PromptOptions(dataset_description="..."))  # once per model

explanation = explain_instance(
    ebm,
    X.iloc[0].to_dict(),      # the value of every feature, by name
    global_explanation,
    audience="layperson",     # or "data_scientist"
)
print(explanation.counterfactual)
print(explanation.sufficient_reason)
print(explanation.completions)
```

A worked example, with its output, is in the [local explanations notebook](local-notebook.md).

## Facts are computed, words are generated

The design has one rule: **nothing that can be computed is left to the LLM.**

```text
EBM + one input
      │  ebaml.local.facts (Python, deterministic)
      ▼
Facts: prediction, counterfactual, sufficient reason, completions
      │  ebaml.local.fact_text.facts_to_texts(audience)
      ▼
One JSON text per explanation ──┐
                                ├──▶ BAML explain_local (3 LLM calls) ──▶ LocalExplanation
Global explanation (describe_ebm) ┘                                         (3 texts + the facts)
```

1. `compute_local_facts` reads the scores and confidence intervals of the EBM and works out the facts. It is
   deterministic: the same model and input always give the same facts.
2. `facts_to_texts` (module `ebaml.local.fact_text`) renders each fact as JSON for the audience. This is where numbers are
   removed for a layperson.
3. `explain_local` (BAML, `baml_src/local_explanations.baml`) makes three LLM calls concurrently, one per
   explanation. Each prompt starts with the **global explanation** of the model, so the LLM knows what the features
   mean, then the prediction, a question, and the facts.

The result is a `LocalExplanation` with the three texts and `facts`, the numbers behind them, so any sentence can be
checked against the data.

## The three explanations

They are the explainability queries of Barceló, Monet, Pérez and Subercaseaux,
[*Model Interpretability through the Lens of Computational Complexity*](https://arxiv.org/abs/2010.12265). An EBM without interactions predicts `intercept + Σ score_i(value_i)`, one independent effect per feature, which
makes the first two queries solvable exactly.

| Explanation | Query in the paper | Question | How it is computed |
| --- | --- | --- | --- |
| `counterfactual` | Minimum change required | What is the *fewest* inputs to change so that the prediction flips? | Exact. |
| `sufficient_reason` | Minimum sufficient reason | What is the *fewest* inputs that settle the prediction, whatever the other inputs are? | Exact. |
| `completions` | Count completions | If only some inputs are known, what share of the possible completions get this prediction? | Estimated by sampling. |

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
completions that get the same prediction. The unknown features are sampled independently from the training
distribution of the EBM (`bin_weights_`) with a fixed seed. Step 0 is the base rate with nothing known. Only the first `max_steps` features are revealed; if all of them are, the last share is 100% by definition (every input known). Unlike the other two, **these are estimates and must be described as
approximate.**

## Confidence

The EBM stores a standard deviation for every score. `confidence_level` (default `0.95`) moves each score by
`z` standard deviations **against the claim being made**:

- a **counterfactual** must flip the prediction even if the effects of the changed features are at the unfavourable
  edge of their confidence interval, so it is harder to find;
- a **sufficient reason** must keep the prediction even if the fixed features sit at the unfavourable edge and the
  free ones take their worst values, so it needs more features, or does not exist.

`confidence_level=0` ignores the uncertainty. The probability of the prediction also gets a confidence interval from
the variance of the sum of the effects (`logit_std`), which assumes the errors of the effects are independent.

## Audience

| `audience` | What the LLM is given | What it is told |
| --- | --- | --- |
| `"layperson"` (default) | Words instead of model numbers: effects are `strongly`/`moderately`/`slightly` `towards`/`against`, probabilities are `very likely`, `unlikely`, ... Values of the person's own inputs (their age) stay. | No numbers that describe the model, no jargon. |
| `"data_scientist"` | Scores, standard deviations, the probability and its confidence interval, the logit, the confidence level. | Quote the numbers, and say how uncertain a claim is. |

Removing the numbers *before* the prompt is deliberate: an instruction not to repeat a number is weaker than not
having the number. The choice is validated, so `audience="child"` raises `InvalidOptionError` before any LLM call.

You can see exactly what the LLM reads, without calling it:

```python
from ebaml.core.custom_types import Audience
from ebaml.local.facts import compute_local_facts
from ebaml.local.fact_text import facts_to_texts

facts = compute_local_facts(ebm, instance)
texts = facts_to_texts(facts, Audience.DATA_SCIENTIST)
print(texts["counterfactual"])
```

```json
{
  "flip_possible": true,
  "changes": [
    {"feature": "sex", "from": "male", "to": "female", "score_change": 2.437}
  ],
  "minimum_number_of_changes": 1,
  "flip_holds_at_confidence": 0.95,
  "probability_of_new_prediction": 0.589
}
```

For a layperson, the same fact has no scores and no confidence level, and the probability becomes a word:

```json
{
  "flip_possible": true,
  "changes": [{"feature": "sex", "from": "male", "to": "female"}],
  "minimum_number_of_changes": 1,
  "flip_holds_even_if_the_model_is_somewhat_off": true,
  "likelihood_of_new_prediction": "about as likely as not"
}
```

(The example is passenger 0 of the notebook: a 35-year-old man in third class, with a survival probability of 0.11.)

## Options

The signature is `explain_instance(ebm, instance, global_explanation, audience, num_sentences, *, local, prompt)`.
The arguments `local` and `prompt` are keyword-only. `local` is a `LocalOptions` from `ebaml.local.facts`. `prompt` is a
`PromptOptions` from `ebaml.llm.options`. The table shows where each setting goes:

| Argument | Default | Meaning |
| --- | --- | --- |
| `audience` | `"layperson"` | `"layperson"` or `"data_scientist"` (or an `Audience`). |
| `num_sentences` | `5` | The most sentences in each of the three explanations. |
| `local.confidence_level` | `0.95` | See [Confidence](#confidence). In `[0, 1)`. |
| `local.min_support` | `0.01` | Share of the training data a value needs to be offered as an alternative. A category needs this share; for a continuous feature the lowest and highest `min_support` of the data are trimmed. It keeps freak values with extreme scores out of the counterfactual and out of the worst case of the sufficient reason. `0` allows every value. |
| `local.max_steps` | `5` | The most features revealed one by one in `completions`. |
| `local.n_samples` | `10000` | Samples used for `completions`. |
| `local.seed` | `0` | Seed of the sampling, so that the facts are reproducible. |
| `prompt.expert_description`, `prompt.dataset_description` | see `default_graph_options` | Used in the prompts. A dataset description helps. |

The LLM client is the `Router` in `baml_src/talk_to_ebm.baml`, shared with the global explanation: see
[Changing the LLM provider](providers.md). Token usage of the three calls is logged at `INFO` on the
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
- **`completions` is an estimate**, and treats the unknown features as independent, as the EBM does.

## Limits

- **Binary classifiers only.** Other tasks raise `ValueError`. A regressor has no "other outcome" to flip to.
- **No interactions.** Fit with `interactions=0`. With pairwise terms the effects no longer add up per feature, so the
  greedy solutions would not be exact; the function raises `ValueError` instead of answering approximately.
- **Every input is needed.** A missing value, a feature name that is absent or a category not seen in training raises
  `ValueError`. Impute, or drop the row.
- **The words are not guaranteed.** The facts are exact; the LLM can still misread them. The prompts forbid inventing
  values and tell it which claims are guarantees and which are estimates, and `explanation.facts` lets you check.
