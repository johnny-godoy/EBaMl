---
description: "Choose the LLM of each call with the model argument: providers, options, keys and limits."
---

# Choosing the model

<!-- invisible-code-block: python
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)
-->

The default model is OpenRouter's free router, `openrouter/free`. It needs `OPENROUTER_API_KEY`. The `model` argument
of `describe_graph`, `describe_ebm` and `explain_instance` selects another model for one call. EBaMl sends every LLM
request of that call to that model. You do not change the package.

## Quick start

Pass the shorthand `"provider/model"`:

```python
from ebaml import DescribeEBMRequest, describe_ebm

print(describe_ebm(DescribeEBMRequest(ebm, model="anthropic/claude-haiku-4-5")).text)
```

Pass `ModelOptions` to set more:

```python
from ebaml import ModelOptions

model = ModelOptions(provider="anthropic", model="claude-haiku-4-5", temperature=0.2, max_output_tokens=8000)
print(describe_ebm(DescribeEBMRequest(ebm, model=model)).text)
```

EBaMl splits the shorthand at the first `/`. The text `"openrouter/openai/gpt-4o-mini"` selects the provider
`openrouter` and the model `openai/gpt-4o-mini`. EBaMl checks the provider before the first paid request. It raises
`InvalidOptionError` if the provider is not in the table below.

## Providers

The provider is a BAML provider prefix. Set the environment variable of the key in the shell that runs Python.

| Provider | Default key variable | Graph images | Notes |
| --- | --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY` | OpenAI format | The model name is the OpenRouter name, for example `openai/gpt-4o-mini`. `openrouter/free` routes a request with an image to a free model that reads images. |
| `openai` | `OPENAI_API_KEY` | OpenAI format | |
| `anthropic` | `ANTHROPIC_API_KEY` | Anthropic format | |
| `google` | `GOOGLE_API_KEY` | Gemini format | The Gemini API. |
| `ollama` | None | OpenAI format | A local server. `base_url` defaults to `http://localhost:11434/v1`. |
| `openai-generic` | None | OpenAI format | Any OpenAI-compatible server. `base_url` is required. Set `api_key_env` if the server needs a key. |
| `azure` | `AZURE_OPENAI_API_KEY` | OpenAI format | The model name is the name of the deployment. Set `base_url` to the deployment, or set `AZURE_OPENAI_ENDPOINT`. |
| `vertex` | None | Gemini format | Google Cloud credentials (`GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth`). The project comes from `GOOGLE_CLOUD_PROJECT`. |
| `bedrock` | None | Bedrock format | The AWS credentials of the machine. The region comes from `AWS_REGION`. `api_key_env` has no effect. |

The column "Graph images" gives the format in which each provider gets the PNG of a graph in vision and mixed mode
([Text, vision and mixed modes](usage.md#text-vision-and-mixed-modes)). The provider takes the image only if the model
reads images. Set `ModelOptions(vision=False)` for a model that reads no images.

Run `baml describe <client>.new`, for example `baml describe aws.BedrockClient.new`, to see all settings of a client.

## Options

| Field | Meaning | Default |
| --- | --- | --- |
| `provider` | The BAML provider prefix from the table. | Required. |
| `model` | The name of the model at the provider. | Required. |
| `api_key_env` | The name of the environment variable that holds the key. | The variable of the provider. |
| `base_url` | The address of the server. | The address of the provider. |
| `temperature` | The sampling temperature. | 0.7 (0.0 in the benchmark). |
| `max_output_tokens` | The most tokens of one reply. | See below. |
| `headers` | Headers to add to each request. | None. |
| `vision` | Whether the model reads images. `False` applies `GraphOptions.vision_fallback` before any request of a call in vision or mixed mode. | `None`: the call sends the image. |

A field that you set wins over the default of the call. The output limit has two defaults:

| Calls | `max_output_tokens` |
| --- | --- |
| Turn 1 of a graph description, the draft of the summary, and the two turns of the benchmark | 16000 |
| All other calls, including the local explanations | 12000 |

## Examples

```python
# OpenRouter, a pinned model (the free router picks a new model for each request)
describe_ebm(DescribeEBMRequest(ebm, model="openrouter/openai/gpt-4o-mini"))

# OpenAI
describe_ebm(DescribeEBMRequest(ebm, model="openai/gpt-4o-mini"))

# Google Gemini
describe_ebm(DescribeEBMRequest(ebm, model="google/gemini-2.5-flash"))

# A local model with Ollama. It needs no key.
describe_ebm(DescribeEBMRequest(ebm, model="ollama/llama3.1"))

# A company gateway with an OpenAI-compatible API
gateway = ModelOptions(
    provider="openai-generic",
    model="my-model",
    base_url="https://llm.example.com/v1",
    api_key_env="MY_GATEWAY_KEY",
    headers={"x-team": "ebm"},
)
describe_ebm(DescribeEBMRequest(ebm, model=gateway))
```

The accuracy benchmark has the flag `--model`:

```bash
uv run python benchmark/run.py accuracy --model anthropic/claude-haiku-4-5 --limit 10
```

## Keys

Put the key in an environment variable, in the shell or with `os.environ`. `ModelOptions` holds the name of the
variable, never the key, so the key does not appear in a `repr`, a log line or a traceback. EBaMl reads the variable
only when it sends a request.

## Limits

- One call uses one model. The turns of a call cannot use different models.
- Python cannot set up fallback chains or round robin. A BAML client object cannot cross from Python to BAML, so
  Python sends plain data and BAML builds the client. A BAML client declaration can use `ai.clients.Fallback`: see
  "Change the default model" in `CONTRIBUTING.md`.
- A reasoning model counts its hidden tokens in `max_output_tokens`. Keep the limit large, or the reply can end before
  the answer.
- Quality, speed and cost depend on the model. Small models can ignore the number of sentences that you ask for.
- EBaMl does not read the input of a model from the provider. In vision or mixed mode, EBaMl sends the image unless
  `ModelOptions(vision=False)` says that the model reads no images, and a model that rejects the image raises
  `VisionUnsupportedError` or falls back to text.

Every client has the retry policy of the default clients: EBaMl retries a transient failure up to 3 times.

To try the providers in a notebook, see the [model providers notebook](providers-notebook.md).
