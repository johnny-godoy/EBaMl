---
description: "Why an EBM needs words: EBaMl puts the graphs and the facts of a prediction in plain text for each reader."
---

# Why EBaMl?

An [EBM](what-is-an-ebm.md#ebm) is a [glass-box model](what-is-an-ebm.md#glass-box-model): a person can read every
part of it. But a model with many features has many [graphs](what-is-an-ebm.md#shape-function), and each graph can
have hundreds of bins. A customer who asks "why was my loan denied?" does not want a graph. EBaMl puts the graphs and
the predictions of an EBM into words, and it computes in Python every fact that the words state.

This page follows one model from the graph to the words. The model is an EBM of synthetic loan data, with the features
`income`, `debt_ratio`, `age` and `late_payments`. The docs build fits it and computes the graph, the graph text and
the facts below (`scripts/docs/figures.py`). The texts in the boxes marked "Example" come from one run of an LLM. Another
run, or another model, gives other words.

## From a graph to a description

`interpret` shows the graph of a feature as a picture:

![The graph of income in an EBM of synthetic loan data. The score goes up from about -1.6 at low incomes, crosses zero
near 40,000, and stays near +1.3 above 55,000. The confidence band is widest at the highest incomes, where the data is
sparse.](generated/income-graph.svg){ width="640" height="360" }

Many LLMs read only text. A vision model can read the picture, but it estimates each value from the axes. So EBaMl
writes the graph as text, which every model reads and which gives the exact numbers. `EBMGraph.to_text` lists the
score and the confidence interval of each bin. The text below drops the bins that change the score by less than 5 % of its range
(`EBMGraph.simplify`):

```text
--8<-- "docs/generated/income-graph.txt"
```

`describe_graph` sends this text to the LLM in a chain of three turns, and returns a description of the graph:

!!! example "Example: the description of the graph of `income` (one run of `openrouter/free`)"
    The income function shows a strong overall increase from approximately –1.6 at the lowest earnings to positive
    values, crossing zero near $40,000–$41,000 and indicating that lower incomes substantially hinder loan approval
    while higher incomes generally help. Between roughly $55,000 and $95,000 the curve plateaus at a high positive level
    of about +1.3 to +1.4, reflecting a strong and fairly stable benefit from moderate-to-high income. The most striking
    feature is what occurs above $95,000, where the function unexpectedly declines, dropping to around +1.29 for incomes
    between $95,400 and $116,500 and further to about +1.03 in the top bin $133,300–$159,500, which contradicts the
    intuitive expectation that more income always strengthens approval odds. This decline coincides with very wide
    confidence intervals, especially in the highest bins, suggesting the pattern is driven by sparse data rather than a
    robust real effect. [...]

`describe_ebm` describes each graph in the same way, and then writes a summary of the whole model.

## From a prediction to an explanation

A [local explanation](what-is-an-ebm.md#local-explanation) explains one prediction. Take this applicant, whom the
model denies:

```json
--8<-- "docs/generated/applicant.json"
```

`interpret` gives the contribution of each feature to this prediction. EBaMl answers three questions that a person
asks (Barceló et al., 2020): what must change to flip the decision, which inputs settle the decision on their own, and
how certain the decision becomes as the inputs become known. Python computes the answers from the scores and the
confidence intervals of the EBM. For a data scientist, the facts of the first two questions are:

```json
--8<-- "docs/generated/counterfactual.json"
```

```json
--8<-- "docs/generated/sufficient-reason.json"
```

`is_smallest_set` says that no smaller set of features works. The [proof of the local
explanations](local-explanations-proof.md) shows this for the search of the code. The LLM only puts the facts into words.
For a layperson, the facts have no scores, and the probabilities become words:

!!! example "Example: the explanation for a layperson (one run of `openrouter/free`)"
    **What would change the decision.** The prediction is "denied," and it is very likely, mainly because an income of
    $31,500 and a debt ratio of 0.46 each push strongly toward denial, one late payment adds a moderate push, and age 67
    pushes slightly the other way. To get a different answer, only one thing has to change: income would need to rise
    from $31,500 into the range from $82,900 up to just under $83,150. That is the smallest change that works — nothing
    smaller flips the outcome, and no adjustment to the debt ratio, late payments, or age is needed. [...]

    **What settles the decision.** This model's prediction of "denied" is already settled by just two inputs: the
    applicant's income of $31,500 and debt ratio of 0.46. The other two inputs — one late payment and age 67 — cannot
    change this outcome for any realistic values they might take, because rare extreme values are excluded from the
    guarantee. [...] This statement describes how this particular model behaves, not what would happen in the real
    world.

The result of `explain_instance` holds the facts next to the words (`explanation.facts`), so a person can check each
sentence. The words of an LLM can still be wrong. [Local explanations](local-explanations.md) says what each
guarantee means and what it does not mean.

To compute the facts alone, with no LLM and no API key, call `compute_local_facts`:

```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import DataScientist, compute_local_facts, facts_to_texts

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
facts = compute_local_facts(ebm, X.iloc[0].to_dict())
print(facts_to_texts(facts, DataScientist()).counterfactual)  # the JSON that the LLM reads
```

## What EBaMl adds to `interpret`

| Task | `interpret` | EBaMl |
| --- | --- | --- |
| Read a graph | A picture of the graph. | The graph as text, and a description of the graph in words (`describe_graph`). |
| Read the whole model | One picture for each graph, and the importance of each feature. | A summary of the model in words (`describe_ebm`). |
| Explain one prediction | The contribution of each feature. | A counterfactual, a sufficient reason and completions, computed in Python and put in words for a layperson or a data scientist (`explain_instance`). |
| Change the model | Methods of the model for whole terms: `monotonize`, `scale`, `remove_terms`. | Edits of single bins and ranges of a graph, as in GAM Changer, with a report, a history and business constraints (`EditSession`). See [Editing a model](editing.md). |
| Give the model to an agent | No tool for agents. | An MCP server with tools to read, explain and edit the model (`ebaml-mcp`). See [MCP server](mcp.md). |
| Choose the LLM | Not applicable. | Any provider from Python with `model=`: OpenRouter, OpenAI, Anthropic, Gemini, Ollama and more. See [Choosing the model](providers.md). |

EBaMl is based on [TalkToEBM](https://github.com/interpretml/TalkToEBM) (Lengerich et al., 2023), which showed that an
LLM can read the graph text of an EBM. The [home page](index.md#how-ebaml-differs-from-talktoebm) says how the two
differ.

## Next steps

1. Install EBaMl and set an API key: [Setup](setup.md).
2. Describe a graph and a model: [Usage](usage.md).
3. Explain one prediction: [Local explanations](local-explanations.md).

## References

- Barceló, P., Monet, M., Pérez, J. and Subercaseaux, B. (2020). Model Interpretability through the Lens of
  Computational Complexity. NeurIPS 2020. [arXiv:2010.12265](https://arxiv.org/abs/2010.12265).
- Lengerich, B. J., Bordt, S., Nori, H., Nunnally, M. E., Aphinyanaphongs, Y., Kellis, M. and Caruana, R. (2023). LLMs
  Understand Glass-Box Models, Discover Surprises, and Suggest Repairs.
  [arXiv:2308.01157](https://arxiv.org/abs/2308.01157).
