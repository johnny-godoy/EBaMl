# EBaMl

<!-- --8<-- [start:intro] -->
EBaMl explains an [Explainable Boosting Machine](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/#ebm) (EBM) in plain words. An EBM is a
[glass-box model](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/#glass-box-model) for tabular data: it adds one learned
[graph](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/#shape-function) for each feature, so a person can read how each feature moves a
prediction. EBaMl gives the graphs and the predictions of an EBM to a large language model (LLM), and returns text that
a data scientist, a customer or a regulator can read.

![The graph of income in an EBM of synthetic loan data. The score goes up from about -1.6 at low incomes, crosses zero
near 40,000, and stays near +1.3 above 55,000.](https://johnny-godoy.github.io/EBaMl/generated/income-graph.svg)

- **Describe a model.** `describe_graph` and `describe_ebm` turn each graph, and the whole model, into a description.
- **Explain one prediction.** `explain_instance` gives a [local explanation](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/#local-explanation):
  what must change to flip the decision, which inputs settle it, and how certain it is. Python computes these facts
  from the scores of the EBM, and the LLM only puts them in words.
- **Edit a model.** `EditSession` changes the graphs to meet a requirement of the business, with a report and a history.
- **Give the model to an agent.** `ebaml-mcp` is an MCP server with tools to read, explain and edit the model.
- **Use any LLM.** The default is the free router of [OpenRouter](https://openrouter.ai). Pass `model=` to use OpenAI,
  Anthropic, Gemini, a local Ollama server or another provider. The LLM gets the graph text and, for a local
  explanation, the values of the input. To keep the data on your machines, use a local model.

EBaMl is based on [TalkToEBM](https://github.com/interpretml/TalkToEBM). Python reads the EBM, and
[BAML](https://docs.boundaryml.com) holds the prompts and the LLM clients.
<!-- --8<-- [end:intro] -->

## Why an EBM, and why EBaMl

**Why an EBM instead of XGBoost?** A model of boosted trees, such as XGBoost, is a black box: tools such as SHAP
estimate why it made a prediction. An EBM is a sum of one score for each feature (and for a few pairs of features), so
the contribution of each term is exact, and a person can find a wrong pattern in a graph and edit it. The InterpretML paper says that an EBM "can be
as accurate as many blackbox models" ([Nori et al., 2019](https://arxiv.org/abs/1909.09223)). The fit is somewhat
slower. [What is an EBM?](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/) has the details and the sources.

**What does EBaMl add to `interpret`?** `interpret` fits the EBM, draws its graphs and gives the contribution of each
feature to a prediction. EBaMl adds the words: a description of each graph and of the model, and a local explanation
that a customer can read. It also adds the counterfactual and the sufficient reason of a prediction, edits of single
bins with a history, and the MCP server. [Why EBaMl?](https://johnny-godoy.github.io/EBaMl/why-ebaml/) shows one example from the model to the words.

## Quick start

<!-- --8<-- [start:quick-start] -->
Install the package, and set a free [OpenRouter key](https://openrouter.ai/settings/keys) in your shell:

```bash
pip install ebaml
export OPENROUTER_API_KEY=sk-or-...
```

Fit an EBM, and ask for a description of the graph of its first feature:

```python
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_breast_cancer

from ebaml import DescribeGraphRequest, describe_graph

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
print(describe_graph(DescribeGraphRequest(ebm, feature_index=0)))
```

The free router picks a model for each request, so the text changes from run to run.
<!-- --8<-- [end:quick-start] -->

## Supported models

<!-- --8<-- [start:supported-models] -->
| Feature | Binary classifier | Regressor | Multiclass classifier | Interactions |
| --- | :---: | :---: | :---: | :---: |
| Graph descriptions (`describe_graph`) | ✅ | ✅ | ✅ | ⚠️ |
| Model summary (`describe_ebm`) | ✅ | ✅ | ✅ | ⚠️ |
| Graph text (`EBMGraph`) | ✅ | ✅ | ✅ | ⚠️ |
| Local explanations (`explain_instance`) | ✅ | ✅ | ✅ | ❌ |
| MCP server (`ebaml-mcp`) | ✅ | ✅ | ✅ | ⚠️ |
| Editing (`EditSession`) | ✅ | ✅ | ✅ | ⚠️ |

✅ supported. ⚠️ partly supported: the main effects are described, and the interaction terms are skipped.
`describe_graph` and `describe_ebm` name them in the prompt, so the description says that it leaves them out, and they
log an `interactions_skipped` warning. ❌ not supported yet. Full support is on the
[roadmap](https://github.com/johnny-godoy/EBaMl/blob/main/ROADMAP.md).
<!-- --8<-- [end:supported-models] -->

## Learn more

The [documentation](https://johnny-godoy.github.io/EBaMl/) holds the guides:

| Page | Read it to |
| --- | --- |
| [What is an EBM?](https://johnny-godoy.github.io/EBaMl/what-is-an-ebm/) | Learn what an EBM is, and why a team uses one. |
| [Why EBaMl?](https://johnny-godoy.github.io/EBaMl/why-ebaml/) | See one graph and one prediction go from the model to the words. |
| [Setup](https://johnny-godoy.github.io/EBaMl/setup/) | Install from PyPI or from a source checkout, set an API key, and fix setup errors. |
| [Usage](https://johnny-godoy.github.io/EBaMl/usage/) | Describe a graph or a model, and read the options and the errors. |
| [Local explanations](https://johnny-godoy.github.io/EBaMl/local-explanations/) | Explain one prediction, and learn what each guarantee means. |
| [Editing a model](https://johnny-godoy.github.io/EBaMl/editing/) | Change the graphs of an EBM to meet a requirement of the business. |
| [Choosing the model](https://johnny-godoy.github.io/EBaMl/providers/) | Use another LLM provider with `model=`. |
| [MCP server](https://johnny-godoy.github.io/EBaMl/mcp/) | Give an EBM to an agent, and chat with it in the terminal (`ebaml-chat`). |
| [Notebooks](https://johnny-godoy.github.io/EBaMl/notebooks/) | Run each feature step by step in a marimo notebook. |
| [API reference](https://johnny-godoy.github.io/EBaMl/api/) | Read every public function and class. |

## Contributing

Read [CONTRIBUTING.md](https://github.com/johnny-godoy/EBaMl/blob/main/CONTRIBUTING.md) and the
[development guide](https://johnny-godoy.github.io/EBaMl/development/). The [roadmap](https://github.com/johnny-godoy/EBaMl/blob/main/ROADMAP.md)
lists the planned releases. EBaMl is free software under the
[MIT license](https://github.com/johnny-godoy/EBaMl/blob/main/LICENSE).
