# Plan: custom model providers (0.1.0)

You are a senior Python and BAML engineer. You will implement one milestone of `ROADMAP.md` in the EBaMl repository.
Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
Users install EBaMl as a library. The only way to change the LLM today is the README section "Changing the LLM
provider", which tells users to edit `baml_src/talk_to_ebm.baml` and run `baml generate`. That means modifying the
library. Make the model a run-time choice instead: a user passes a model description from Python to `describe_graph`,
`describe_ebm` and `explain_instance` (and the benchmark script), and EBaMl sends every LLM request of that call to
that model. The default (OpenRouter's `openrouter/free`) stays exactly as it is when the user passes nothing.
</goal>

<verified_facts>
These were tested against `baml toolchain 0.22.2` in a scratch copy of the project. They decide the design, so do not
spend time re-discovering them, but re-check them quickly if `baml --version` differs.

1. Every BAML entry point already takes `client: ai.Client? = null`, and `run_counted` passes it to
   `ai.Agent.new(client = client)`. The offline tests in `baml_src/talk_to_ebm_test.baml` use it with scripted BAML
   clients (`EchoClient {}` and others).
2. **A client object cannot cross the Python/BAML boundary in either direction.** Passing
   `baml_sdk.vendor.openai.OllamaClient.new(...)` from Python fails with
   `TypeError: host value type openai.OllamaClient is not a member of declared union ai.Client | null`. Wrapping it in
   `ai.clients.Retry.new(inner=...)` from Python fails the same way. A BAML function that returns `ai.Client` also fails
   when its result is decoded in Python. So Python must send **plain data**, and BAML must build the client.
3. A parameter typed `ai.Selector?` (alias of `ai.Client | string | baml.env.Ref`) resolved with
   `ai.clients.resolve(client)` accepts a `"provider/model"` string from Python (`"ollama/llama3.1"` reached the
   Ollama endpoint). An unknown prefix raises `ai.errors.InvalidRequest` that lists the valid prefixes (`openai`,
   `openai-chat`, `openai-images`, `ollama`, `openrouter`, `anthropic`, `typesafeai`, and more; print the full list
   once). A string selector carries **no** settings: no temperature, no output limit, no base URL, no key variable.
4. There is no call-time override of temperature or output limit (`ai.Agent.new` takes only `max_steps`,
   `schema_attempts`, `client`, `tool_errors`, `on_event`). Settings must be set on the client when it is built.
5. Vendor constructors read no environment variable at construction. `api_key` accepts `ai.Credential`
   (`string | baml.env.Ref`), and `baml.env.ref(name)` compiles inside a BAML function, so a key can be named by its
   environment variable and read only when the request is sent. With `api_key` unset, each vendor reads its own
   default variable (`OpenRouterClient` reads `OPENROUTER_API_KEY`).
6. Field names differ by vendor: `openai.*` and `anthropic.Client` use `max_tokens` (Anthropic defaults to 8192),
   `google.GeminiClient` uses `max_output_tokens`. Use `baml describe <name>` (for example
   `baml describe aws.BedrockClient.new`) to read each constructor.
7. `baml check` and `baml generate` may need `--agent-skill-check off` in a cloud container, or run
   `baml agent install` first.
</verified_facts>

<design>
The design follows `.claude/rules/solid.md`. Keep to it, and change it only if a verified fact above turns out wrong.
If you change it, write why in the PR description.

<python_side>
Add a frozen `attrs` class `ModelOptions` to `src/ebaml/llm/options.py`, next to `GraphOptions` and `PromptOptions`.
It is pure data, imports nothing from `baml_sdk`, and follows the existing pattern of one options class per part of the
work (SOLID S and I).

```python
@attrs.frozen
class ModelOptions:
    provider: str                       # a BAML provider prefix: "openai", "anthropic", "openrouter", "ollama", ...
    model: str                          # the provider's model name
    api_key_env: str | None = None      # the NAME of the environment variable that holds the key; None = provider default
    base_url: str | None = None         # for OpenAI-compatible gateways, local servers, Azure, ...
    temperature: float | None = None    # None = the default of each call (0.7, or 0.0 for the benchmark)
    max_output_tokens: int | None = None  # None = the default of each call (6000 or 8000)
    headers: Mapping[str, str] = attrs.field(factory=dict)

    @classmethod
    def parse(cls, value: "ModelOptions | str") -> "ModelOptions": ...  # "anthropic/claude-haiku-4-5" -> ModelOptions
```

- Name the output limit `max_output_tokens`, not `max_tokens`, because `GraphOptions.max_tokens` is already the token
  budget of the graph text. Two meanings for one word break the writing rules and confuse users.
- Take the key only by the name of its environment variable. The key then never sits in a Python object, a log line, a
  traceback or a `repr`. Users who type a key in a notebook already set `os.environ`, as the notebooks do.
- `parse` splits a string on the first `/` only (`"openrouter/openrouter/free"` gives provider `openrouter`, model
  `openrouter/free`). Raise `InvalidOptionError` (from `ebaml.core.exceptions`) for a string with no `/`, or an empty
  provider or model. Do not keep a provider list in Python: BAML is the single source of truth for which providers
  exist (see below), so the two lists cannot drift.
- Validate ranges in the style of the existing option classes (`OptionRangeError` for `max_output_tokens < 1` or a
  negative temperature).

Change the public functions in `src/ebaml/llm/descriptors.py`:

- `describe_graph`, `describe_ebm` and `explain_instance` get a keyword-only `model: ModelOptions | str | None = None`.
  A string is the `"provider/model"` shorthand. Follow `.claude/rules/types.md` (the union accepts the plain string, as
  `Audience | str` already does).
- Add one private helper that converts `ModelOptions | None` to `baml_sdk.ModelSpec | None`, and pass the result as
  `model=` to every `baml_sdk` call that sends requests (`describe_graph_cot`, `summarize_ebm`, `explain_local`).
  `None` sends `None`, so the default clients run.
- Check the model before the first paid call. `describe_ebm` converts every graph to text first "so that errors surface
  before any LLM call is paid for"; keep that promise. Call a cheap BAML function (below, `check_model`) once, and
  translate its `BamlError` for an unknown provider into `InvalidOptionError` with the BAML message (raise from the
  original error).
- Update the docstrings: "The LLM client is configured in `baml_src/...`" becomes a description of `model` and its
  default.
- `benchmark/run.py` gets `--model provider/model`. `src/ebaml/benchmark.py` stands alone and must not import from
  `ebaml`, so build the `baml_sdk.ModelSpec` in `run.py` and bind it with `functools.partial`.
</python_side>

<baml_side>
Add to `baml_src/talk_to_ebm.baml`:

```baml
class ModelSpec {
    provider: string,
    model: string,
    api_key_env: string?,
    base_url: string?,
    temperature: float?,
    max_output_tokens: int?,
    headers: map<string, string>?,
}

class CallDefaults {
    temperature: float,
    max_output_tokens: int,
}
```

- One function builds the vendor client from a `ModelSpec` and a `CallDefaults`: a field the user set wins, else the
  default of the call applies. It maps `max_output_tokens` to each vendor's field name, and `api_key_env` to
  `baml.env.ref(name)`. It wraps the result in `ai.clients.Retry.new(inner = ..., max_attempts = 3)`, so a custom
  model keeps the same retry policy as the default clients.
- Look the provider up in a table if BAML can hold constructor lambdas in a `map<string, ...>` (SOLID O: a new provider
  is a new row). If BAML cannot, use one `match` on `provider` and write a comment that says why. Cover at least
  `openai`, `anthropic`, `openrouter`, `ollama`, `google` (Gemini), `openai-generic` (any OpenAI-compatible endpoint;
  it needs `base_url`), `azure`, `vertex` and `bedrock`. Use the same prefix names that `ai.clients.resolve` uses, so
  the docs can say "the BAML provider prefix".
- An unknown provider throws `baml.errors.InvalidArgument` whose message lists the supported providers. Expose
  `check_model(spec: ModelSpec)` that builds and discards the client, for the Python check above.
- Keep the three call defaults in one place: `RouterBase` = 0.7 / 6000, `RouterLongBase` = 0.7 / 8000,
  `RouterBenchmarkBase` = 0.0 / 8000. Define three `CallDefaults` functions next to the `client` lines. If a declared
  client's fields can be read in a test (for example `RouterBase.max_tokens`), add an offline test that the defaults
  match the declared clients.
- Each entry point (`describe_graph_cot`, `summarize_ebm`, `describe_ebm`, `answer_benchmark_question`, and
  `explain_local` in `baml_src/local_explanations.baml`) gets `model: ModelSpec? = null` after `client`. Build the short
  and the long client once per call, and pass the right one to each `run_counted`, so turn 1 and the draft summary keep
  their larger output limit. Precedence: `client` (the test hook) > `model` > the function's declared client. Keep
  `client: ai.Client?`: the offline tests depend on it, and it is not exposed to Python users.
- Leave the prompts unchanged. This milestone changes where requests go, not what they say.
</baml_side>

<out_of_scope>
- Fallback chains, round robin and custom `ai.Client` implementations from Python. Fact 2 makes them impossible
  without a toolchain change. Document this limit.
- Different models for different turns of one call.
- An `EBAML_MODEL` environment variable that selects the model with no code. List it as a follow-up in the PR.
</out_of_scope>
</design>

<tests>
1. Offline BAML tests in `baml_src/talk_to_ebm_test.baml` (`OPENROUTER_API_KEY=unused baml test -x root::Live`): each
   supported provider builds without network access; user values win over call defaults; null fields take the call
   defaults; an unknown provider throws `InvalidArgument` naming the supported list; `client` takes precedence over
   `model`; `model = null` keeps the declared client.
2. Python unit tests in the mirrored files `tests/unit/ebaml/llm/test_options.py` and `test_descriptors.py`:
   `ModelOptions.parse` (first-slash split, the three error cases), range errors, the `headers` default; each public
   function passes the right `ModelSpec` (or `None`) to the SDK; a string `model` gives the same spec as the equal
   `ModelOptions`; an unknown provider raises `InvalidOptionError` before any paid call. Use `fake_sdk` from
   `tests/unit/ebaml/conftest.py` for every SDK fake: it checks each fake against the generated stub, so it catches a
   signature you forgot to regenerate.
3. Coverage stays at 100% (`uv run pytest`). Mutation testing of the changed modules has zero survivors:
   `uv run python scripts/mutation.py --base origin/main`.
4. A live prompt test in `baml_src/prompt_tests.baml` that calls `describe_graph_cot` with
   `ModelSpec { provider: "openrouter", model: "openrouter/free", ... }`, so the `Prompt tests` workflow proves the
   new path end to end. Run it with `baml test -i root::Live` only if a real `OPENROUTER_API_KEY` is set; otherwise say
   in the PR that it was not run locally.
</tests>

<documentation>
Users must be able to use any model from the docs alone, so the docs are a main deliverable. Write all prose in the
ASD-STE100 style of `.claude/rules/writing-style.md`: "model", "provider" and "client" each mean one thing.

1. Rewrite the README section between `<!-- --8<-- [start:changing-model] -->` and `[end:changing-model]`. Keep both
   markers, because `docs/providers.md` includes the section. Content, in this order:
   - The default model, and that `model=` changes it for one call with no change to the package.
   - A quick start: `describe_ebm(ebm, model="anthropic/claude-haiku-4-5")`, and the same call with `ModelOptions`.
   - A table of providers: prefix, the default key variable, and notes (Ollama needs no key; `openai-generic` needs
     `base_url`; the Azure, Vertex and Bedrock settings). Take the variable names from `baml describe`.
   - A table of the `ModelOptions` fields, and the call defaults (which turns use 6000 and which use 8000 output
     tokens).
   - Examples: OpenRouter with a pinned model, OpenAI, Anthropic, Gemini, local Ollama, a company gateway through
     `openai-generic` with headers, and the benchmark `--model` flag.
   - Keys: set the variable in the shell or with `os.environ`. EBaMl reads it only when it sends a request.
   - Limits: one model per call; no fallback chains from Python; reasoning models count hidden tokens against
     `max_output_tokens`, so keep it large.
   - Remove the instructions to edit `baml_src/`.
2. Retitle `docs/providers.md` ("Choosing the model") and its entry in `zensical.toml`. Check that `ModelOptions`
   renders in `docs/api.md`. Build the docs with `uv run --group docs python -m scripts.build_docs` (strict).
3. Add to `CONTRIBUTING.md` how to change the **default** model (the `*Base` lines), which is a maintainer task.
4. Update the `<architecture>` section of `CLAUDE.md`: the `*Base` lines set the default only, users pass
   `ModelOptions`, and `ModelSpec` is the data that crosses the boundary (with the reason from fact 2). Keep it short.
5. Update the notebook prose in `notebooks/titanic.py` and `notebooks/local_explanations.py` that says the key must be
   an OpenRouter key, and point to the provider docs.
</documentation>

<release>
This PR completes the 0.1.0 milestone. Follow `.claude/rules/changelog-and-versioning.md`: set `version = "0.1.0"`,
run `uv lock`, and add `## [0.1.0] - YYYY-MM-DD` to `CHANGELOG.md` with `Added` (the `model=` argument, `ModelOptions`,
the benchmark `--model` flag) and `Changed` (the new optional `model` argument of the BAML entry points). Mark the row
of 0.1.0 in `ROADMAP.md` as done.
</release>

<done>
- `baml fmt && baml check`, `OPENROUTER_API_KEY=unused baml test -x root::Live`, `uv run pytest` (100% coverage),
  `uv run python scripts/mutation.py --base origin/main` (zero survivors), the strict docs build, and
  `prek run --all-files` all pass.
- A call with no `model` sends the same requests as before (the existing offline and unit tests pass unchanged).
- A reader of `docs/providers.md` alone can run `describe_ebm` on Anthropic, a local Ollama model and an
  OpenAI-compatible gateway.
- Report what you ran, what passed, and anything you could not run (for example the live tests without a key).
</done>
