# EBaMl

<!-- --8<-- [start:intro] -->
BAML for Explainable Boosting Machines. Python extracts the graphs from an EBM, and BAML holds the prompts, the LLM
clients and the chain-of-thought pipelines.

EBaMl is based on [TalkToEBM](https://github.com/interpretml/TalkToEBM) and adds local
explanations (why did the model predict this for this input?) computed exactly and put in words for a layperson or a
data scientist, a benchmark that scores answers automatically, a configurable LLM provider, token budgets and typed,
structured logging.

LLM calls go through [OpenRouter](https://openrouter.ai)'s free router (`openrouter/free`), so no paid API key is needed.

<!-- --8<-- [end:intro] -->
## Supported models

<!-- --8<-- [start:supported-models] -->
| Feature | Binary classifier | Regressor | Multiclass classifier | Interactions |
| --- | :---: | :---: | :---: | :---: |
| Graph descriptions (`describe_graph`) | ✅ | ✅ | ❌ | ⚠️ |
| Model summary (`describe_ebm`) | ✅ | ✅ | ❌ | ⚠️ |
| Graph text (`EBMGraph`) | ✅ | ✅ | ❌ | ⚠️ |
| Local explanations (`explain_instance`) | ✅ | ❌ | ❌ | ❌ |

✅ supported. ⚠️ partly supported: the main effects are described, and the interaction terms are skipped without a
warning. ❌ not supported yet. Full support is on the roadmap.
<!-- --8<-- [end:supported-models] -->
## Setup

<!-- --8<-- [start:setup] -->
### Requirements

- Python 3.13 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- The BAML toolchain 0.22 or later (step 1).
- An API key for an LLM provider. The default is [OpenRouter](https://openrouter.ai)'s free router, which needs no
  paid plan. You can use another provider instead, see "Changing the LLM provider".

### 1. Install the BAML toolchain

```bash
curl -fsSL https://pkg.boundaryml.com/install.sh | sh -s
```

The installer prints a line to add to your shell profile. Add it, then open a new shell:

```bash
. "$HOME/.baml/env"
```

Check it:

```bash
baml --version   # toolchain 0.22 or later
```

If `baml` resolves to an older install (Homebrew `baml`, or `baml-py` in a virtualenv), remove it or put `~/.baml/bin`
first on `PATH`. The old CLIs have no `describe`, `generate add` or `agent` commands.

If a command stops with "the BAML agent skill is required", run `baml agent install` and retry.

### 2. Get the code and install the Python dependencies

```bash
git clone <repository-url> EBaMl
cd EBaMl
uv sync
```

### 3. Set the API key of your LLM provider

For the default provider, OpenRouter:

1. Create an account at <https://openrouter.ai> and open <https://openrouter.ai/settings/keys>.
2. Create a key. Keys start with `sk-or-`.
3. Export it in the shell you run the code from:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

The key is read from the environment when a request is sent. It is never written to a `.baml` file.

Using another provider? Set its variable instead (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) and edit the clients as
described in "Changing the LLM provider". Local models (Ollama) need no key.

### 4. Generate the Python SDK

```bash
baml generate
```

This writes the `src/baml_sdk/` package from `baml_src/`, next to `ebaml`, so it imports wherever the project is
installed. Rerun it every time a `.baml` file changes, including when you change the provider.

### 5. Check the setup

```bash
OPENROUTER_API_KEY=unused baml test -x root::Live   # offline tests on a scripted client, no network
baml run main                         # prints "hello from baml"
```

These two do not call an LLM, so they pass without a working key or provider. They check the toolchain and the SDK.

### 6. Live test (optional)

Makes real requests to your provider (9 on the OpenRouter free router, so it can be slow or rate limited). It does
nothing without `OPENROUTER_API_KEY`, so with another provider run the example under "Usage" instead.

```bash
OPENROUTER_API_KEY=sk-or-... uv run python scripts/live_openrouter.py
```

It fits a small EBM, runs the three-turn chain directly and checks that the provider reports token usage, then runs
`describe_graph` and `describe_ebm`. It exits non-zero on the first failed assertion.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `baml: command not found` | The profile line from step 1 is not loaded. Open a new shell or run `. "$HOME/.baml/env"`. |
| `unknown command: describe` (or `generate add`, `agent`) | An older `baml` is first on `PATH`. See step 1. |
| `ModuleNotFoundError: baml_sdk` | The SDK is not generated. Run `baml generate` (step 4). |
| An error that names an environment variable | The key of the provider in `baml_src/talk_to_ebm.baml` is not exported in this shell. |
| Rate-limit errors from the free router | Lower the concurrency, `describe_ebm(ebm, max_concurrency=2)`, or pin a paid model. |

<!-- --8<-- [end:setup] -->
## Usage

<!-- --8<-- [start:usage] -->
```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml.llm.descriptors import describe_ebm, describe_graph
from ebaml.llm.options import PromptOptions

ebm = ExplainableBoostingClassifier().fit(X, y)

print(describe_graph(ebm, feature_index=0))
print(describe_ebm(ebm, num_sentences=30, prompt=PromptOptions(dataset_description="...")))
```

A graph can be turned into text with `EBMGraph.to_text` and back with `EBMGraph.from_text`, which also reads a graph
embedded in a longer prompt. It needs the confidence bounds, and it only reads continuous features.

Two keyword-only arguments set the options. `graph=GraphOptions(...)` sets the options of `EBMGraph.from_ebm` and
`EBMGraph.to_text`, for example `max_tokens`. `prompt=PromptOptions(...)` sets the descriptions that the prompts use
(`expert_description`, `graph_description`, `dataset_description` and `task_description`). Both classes are in
`ebaml.llm.options`. Token usage reported by the provider is logged at `INFO` on the `ebaml.llm.descriptors` logger.

### Explaining one prediction

```python
from ebaml.llm.descriptors import describe_ebm, explain_instance

ebm = ExplainableBoostingClassifier(interactions=0).fit(X, y)  # binary classifier, no interactions
global_explanation = describe_ebm(ebm)  # computed once, prepended to every local prompt

explanation = explain_instance(ebm, X.iloc[0].to_dict(), global_explanation, audience="layperson")
print(explanation.counterfactual)      # what would have to change to flip the prediction
print(explanation.sufficient_reason)   # which inputs settle the prediction on their own
print(explanation.completions)         # how certain the outcome is as inputs become known
```

The facts are computed exactly from the scores and confidence intervals of the EBM (`ebaml.local`); the LLM only puts
them in words. They are the three queries of [arXiv:2010.12265](https://arxiv.org/abs/2010.12265): minimum change
required, minimum sufficient reason and count completions. `audience="layperson"` (default) gives no model numbers;
`audience="data_scientist"` quotes scores, probabilities and confidence intervals. It supports binary classifiers
without interactions, and `explanation.facts` holds the numbers behind the text. The "Local explanations" page of the documentation explains what each explanation means, the options and the limits,
and `notebooks/local_explanations.py` walks through them on the Titanic data.

### Errors

Every error that `ebaml` raises on purpose derives from `ebaml.EBaMlError`. Use `except EBaMlError` to catch the errors
of `ebaml` apart from the errors of other code. Each error for a bad value also derives from `ValueError`, so an
`except ValueError` catches it too.

| Error | `ValueError` | Raised when |
| --- | --- | --- |
| `InvalidOptionError` | yes | An option is not one of its valid values, for example an unknown `Normalization`. |
| `OptionRangeError` | yes | A number option is outside its range, for example `confidence_level=1`. |
| `InvalidInstanceError` | yes | The instance to explain lacks a value, or has a category that the EBM did not see. |
| `UnsupportedModelError` | yes | A local explanation gets an EBM with interactions, or one that is not a binary classifier. |
| `GraphTextError` | yes | `EBMGraph.from_text` cannot read the text of a graph. |
| `EBMGraphError` | no | The base class of the errors of a graph. |
| `EBMGraphShapeError` | yes | The bins of a graph do not fit its scores, or a boolean format gets more than two bins. |
| `EBMGraphTokenLimitError` | no | The text of a graph is over the token budget, and its format cannot be simplified. |
| `EBMGraphSimplificationError` | no | The text of a graph is still over the token budget at the largest simplification. |

The last three graph errors derive from `EBMGraphError`. All classes are in `ebaml.core.exceptions`, and `ebaml` exports
them too.

<!-- --8<-- [end:usage] -->
## How it works

| Where | What |
| --- | --- |
| `src/ebaml/core/` | Shared kernel: types and errors. It imports no other part of `ebaml`. |
| `src/ebaml/graphs/` | Graph extraction and simplification (numpy, `interpret`), and the text the LLM reads. |
| `src/ebaml/llm/` | The edge: the calls into BAML (`descriptors.py`), their options and the retry after a cut-off reply. |
| `baml_src/talk_to_ebm.baml` | Prompts, LLM clients, the three-turn description chain, the summary, `describe_ebm`, token accounting. |
| `src/ebaml/local/`, `baml_src/local_explanations.baml` | Local explanations: the facts of one prediction computed deterministically, their rendering as JSON texts, and the prompts that put them in words. |
| `baml_src/talk_to_ebm_test.baml` | Offline tests on a scripted client. |
| `baml.toml` | Generator config for the Python SDK. |
| `src/ebaml/benchmark.py`, `benchmark/` | The benchmark: loading, answer parsing and scoring, and the data and runner (see [Benchmark](#benchmark)). |
| `notebooks/` | Example notebooks, written with marimo. |
| `docs/`, `zensical.toml` | The documentation, built with Zensical (see [Development](#development)). |
| `tests/unit/` | Python unit tests (100% coverage required). `scripts/live_openrouter.py` is the optional live test. |
| `scripts/` | Maintenance scripts, see [scripts/README.md](scripts/README.md). |
| `.github/` | CI workflow and the scripts that turn failures into reports. |
| `src/baml_sdk/` | Generated. Do not edit. |

<!-- --8<-- [start:behaviour] -->
- **Model:** `openrouter/free` picks a free model for each request, so the model can change between calls and between
  the turns of one conversation. Quality and speed vary with it.
- **Retries:** transient failures (rate limits, network errors, 5xx) are retried up to 3 times. Requests the provider
  rejects are not retried.
- **Concurrency:** `describe_ebm` describes features concurrently, at most `max_concurrency` at a time (default 8).
  Free models are rate limited, so lower it (`describe_ebm(ebm, max_concurrency=2)`) if you see rate-limit errors.
- **Token budget:** there is no tokenizer before a call, so the check uses a rough estimate (about 3 characters per
  token). Going a little over `max_tokens` only warns. Going more than `overbudget_percent` (default 50) over makes
  `to_text` simplify the graph, or raise if it cannot. The provider's own error is the real limit. The functions in
  `ebaml.llm.descriptors` pass the BAML check. A direct call of `EBMGraph.to_text` checks nothing, unless you pass
  `check_budget`, for example `baml_sdk.check_token_budget`.

<!-- --8<-- [end:behaviour] -->
## Notebook

[`notebooks/titanic.py`](notebooks/titanic.py) is a [marimo](https://marimo.io) notebook that ports TalkToEBM's example.
It trains an EBM on the Titanic dataset (downloaded from OpenML), asks the LLM to describe a graph and the whole model,
and shows the graph and text API (`from_ebm`, `simplify`, `to_text`, `from_text`). The LLM cells run only when you press
their button and need `OPENROUTER_API_KEY`.

```bash
uv run marimo edit notebooks/titanic.py   # interactive
uv run marimo run notebooks/titanic.py    # as an app
```

[`notebooks/local_explanations.py`](notebooks/local_explanations.py) is the place to learn the local explanations: pick a
passenger, see the facts with no API key, move the confidence slider, check the guarantees against random completions,
compare what the LLM reads for a layperson and a data scientist, and (with a key) get the explanations in words.

```bash
uv run marimo edit notebooks/local_explanations.py
```

## Benchmark

The benchmark follows [TalkToEBM's benchmark](https://github.com/interpretml/TalkToEBM/tree/main/benchmark) that checks how well a
model reads EBM graphs: function recognition (easy and hard), the largest jump, monotonicity, reading a value and the
widest confidence interval. It makes real requests, so it needs `OPENROUTER_API_KEY`, and it asks two LLM turns per
question (6 tasks, 480 questions in all).

```bash
uv run python benchmark/run.py --task monotonicity --limit 10 --output results.json
```

See [benchmark/README.md](benchmark/README.md) for the tasks, the scoring and what it leaves out.

## Changing the LLM provider

<!-- --8<-- [start:changing-model] -->
OpenRouter's free router is only the **default**. The prompts and the pipelines do not depend on it: every LLM call goes
through a BAML client, and BAML ships clients for many providers. To switch, edit the clients at the top of
`baml_src/talk_to_ebm.baml`, then run `baml generate`. Nothing in Python changes.

### The clients

| Client | Used by | Default settings |
| --- | --- | --- |
| `RouterBase` | the `Router` client: turns 2 and 3 of a graph description, and the shortening of the summary | `temperature = 0.7`, `max_tokens = 6000` |
| `RouterLongBase` | the `RouterLong` client: turn 1 of a graph description, and the draft summary | `temperature = 0.7`, `max_tokens = 8000` |
| `RouterBenchmarkBase` | the `RouterBenchmark` client of the benchmark | `temperature = 0.0`, `max_tokens = 8000` |

`Router`, `RouterLong` and `RouterBenchmark` wrap the matching `*Base` client in `ai.clients.Retry`, which retries
transient failures up to 3 times. Change the `*Base` lines only. The retry wrappers keep working with any provider.
The three can use different providers, for example a cheap model for `RouterBase` and a stronger one for the benchmark.

### Examples

Replace the right-hand side of a `*Base` line. Keep each client's own `temperature` and `max_tokens`.

```baml
// OpenRouter, a pinned model (https://openrouter.ai/models)
client RouterBase = openai.OpenRouterClient.new(model = "<provider>/<model>", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 6000);

// OpenAI
client RouterBase = openai.ChatClient.new(model = "gpt-4o-mini", api_key = env.OPENAI_API_KEY, temperature = 0.7, max_tokens = 6000);

// Anthropic
client RouterBase = anthropic.Client.new(model = "claude-haiku-4-5", api_key = env.ANTHROPIC_API_KEY, temperature = 0.7, max_tokens = 6000);

// Google Gemini (the output limit is called max_output_tokens)
client RouterBase = google.GeminiClient.new(model = "gemini-2.5-flash", api_key = env.GEMINI_API_KEY, temperature = 0.7, max_output_tokens = 6000);

// A local model through Ollama (no key; the URL defaults to http://localhost:11434/v1)
client RouterBase = openai.OllamaClient.new(model = "llama3.1", temperature = 0.7, max_tokens = 6000);

// Any other OpenAI-compatible endpoint (vLLM, LM Studio, a company gateway, ...)
client RouterBase = openai.GenericClient.new(model = "<model>", base_url = "https://<host>/v1", api_key = env.MY_GATEWAY_KEY, temperature = 0.7, max_tokens = 6000);
```

BAML also has `openai.AzureClient`, `aws.BedrockClient`, `google.VertexClient` and `claude_code.ClaudeCodeClient`. Run
`baml describe <name>` (for example `baml describe aws.BedrockClient.new`) to see every parameter of a client.

Set the environment variable you named in `api_key = env.<NAME>` in the shell you run Python from. Keys are read when a
request is sent and are never stored in a `.baml` file.

To fall back to a second provider when the first one fails, wrap both in `ai.clients.Fallback`:

```baml
client RouterBase = ai.clients.Fallback {
    members: [
        anthropic.Client.new(model = "claude-haiku-4-5", api_key = env.ANTHROPIC_API_KEY, temperature = 0.7, max_tokens = 6000),
        openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, temperature = 0.7, max_tokens = 6000),
    ],
};
```

### After the change

```bash
baml generate                 # regenerate src/baml_sdk/
baml fmt                      # keep the .baml files formatted
OPENROUTER_API_KEY=unused baml test -x root::Live    # offline tests; they use a scripted client and ignore your provider
baml test -i root::Live    # live prompt tests; they send real requests with your key
```

Then run your usual code. `describe_graph` and `describe_ebm` need no other change.

!!! note
    Quality, speed and cost depend on the model. Small and free models may ignore the requested number of sentences
    or fail the benchmark's answer formats. The default `openrouter/free` also picks a different model for each
    request, so results are not reproducible: pin a model to compare runs.

### Using a client you build yourself

All BAML entry points (`describe_graph_cot`, `summarize_ebm`, `describe_ebm`, `answer_benchmark_question`) take an
optional `client: ai.Client`. When it is `null` they use the clients above. The offline tests pass a scripted client
this way. From BAML code you can pass any `ai.Client`, including your own implementation of the interface. The Python
wrappers in `ebaml.llm.descriptors` do not expose this parameter, so from Python the clients in
`baml_src/talk_to_ebm.baml` are the way to configure the provider.

<!-- --8<-- [end:changing-model] -->

## Development

<!-- --8<-- [start:development] -->

```bash
baml fmt          # format .baml files
baml check        # compile check
baml test -x root::Live   # run the offline BAML tests
baml generate     # regenerate src/baml_sdk/
uv run pytest     # Python unit tests, fails under 100% coverage
prek run --all-files                       # every hook: formatters, ruff, vulture, pyrefly, baml fmt, ...
uv run python scripts/mutation.py          # mutation testing of all modules, fails if any mutant survives
uv run python scripts/mutation.py --base origin/main   # the same, but only for modules changed since main
uv run python scripts/mutation.py --module src/ebaml/local/facts.py   # the same, but only for one module
uv run python -m scripts.check_ignores     # every lint/type/test workaround needs a reason written next to it
uv run python -m scripts.compile_ignores   # all workarounds and their reasons as one Markdown report to audit
uv run --group docs python -m scripts.build_docs         # build the docs; fails on a broken link or a docstring warning
uv run --group docs python -m scripts.build_docs serve   # preview the docs (exports the notebooks, so the first run is slow)
```

CI (`.github/workflows/ci.yml`) runs prek, secret and security scans (trufflehog, semgrep, trivy), `baml check` and
`baml test`, pytest with 100% coverage, and mutation testing of the changed modules with a 100% score. A weekly run tests all modules, one job for each module. Lint and type workarounds
(`# noqa`, `# type: ignore`, skipped tests, ...) must carry a written reason next to them.
<!-- --8<-- [end:development] -->
