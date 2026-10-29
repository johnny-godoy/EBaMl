# EBaMl

--8<-- "README.md:intro"

## Supported models

--8<-- "README.md:supported-models"

## Where to go next

- [Setup](setup.md): install the BAML toolchain, set an API key, generate the SDK, troubleshooting.
- [Choosing the model](providers.md): OpenRouter is only the default. Pass `model=` to use OpenAI, Anthropic, Gemini, Ollama and more.
- [Usage](usage.md): describe a graph or a whole model, and how the model, retries and token budget behave.
- [Local explanations](local-explanations.md): why did the model predict this for this input? A counterfactual, a sufficient reason and the share of completions, computed exactly and put in words for a layperson or a data scientist.
- [Notebook](notebook.md): a worked example on the Titanic data, with its output.
- [Local explanations notebook](local-notebook.md): the same, one passenger at a time, with its output.
- [Model providers notebook](providers-notebook.md): the same request sent to OpenRouter, OpenAI, Anthropic, Ollama and a gateway, side by side.
- [Multiclass notebook](multiclass-notebook.md): the graphs and the descriptions of an EBM with three classes.
- [Model kinds notebook](model-kinds-notebook.md): a binary classifier, a multiclass classifier and a regressor side by side.
- [Benchmark](benchmark.md): how well does a model read EBM graphs?
- [API reference](api.md): every public function and class.

## How EBaMl differs from TalkToEBM

[TalkToEBM](https://github.com/interpretml/TalkToEBM) builds the prompts in Python and calls OpenAI or Azure
directly. EBaMl keeps the Python that has to touch the EBM (extracting and simplifying the graphs) and moves the rest
to BAML: the prompts, the LLM clients with their retries, the chain-of-thought pipelines and the token accounting. The
provider and model are changed in a few lines of a `.baml` file, and the pipelines can be tested offline on a scripted client.
