# EBaMl

--8<-- "README.md:intro"

## Start here

Choose the page for what you need:

<div class="grid cards" markdown>

-   **I do not know what an EBM is**

    ---

    Learn what an Explainable Boosting Machine is, how it makes a prediction, and what it costs.

    [What is an EBM?](what-is-an-ebm.md)

-   **I must explain decisions to people**

    ---

    See one prediction go from the model to a text for a layperson, with the facts behind each sentence.

    [Why EBaMl?](why-ebaml.md), then [Local explanations](local-explanations.md)

-   **I know EBMs, and I want the API**

    ---

    Install the package, describe a graph or a model, and choose the LLM.

    [Setup](setup.md), [Usage](usage.md) and [API reference](api.md)

-   **I build an agent**

    ---

    Give an EBM to an agent through the MCP server: tools to read, explain and edit the model.

    [MCP server](mcp.md)

</div>

## Quick start

--8<-- "README.md:quick-start"

The [notebooks](notebooks.md) show each feature step by step, with the output of each step.

## Supported models

--8<-- "README.md:supported-models"

## How EBaMl differs from TalkToEBM

[TalkToEBM](https://github.com/interpretml/TalkToEBM) builds the prompts in Python and calls OpenAI or Azure
directly. EBaMl keeps the Python that has to touch the EBM (extracting and simplifying the graphs) and moves the rest
to BAML: the prompts, the LLM clients with their retries, the chain-of-thought pipelines and the token accounting. The
pipelines can be tested offline on a scripted client. EBaMl also adds the local explanations, the editing of a model,
the MCP server and the choice of the LLM from Python.
