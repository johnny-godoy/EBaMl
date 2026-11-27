# Plan: a native async API next to the sync API (0.15.0)

You are a senior Python engineer with experience in asyncio, library API design and concurrent I/O. You will implement
one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, with
special care for `.claude/rules/solid.md`, `.claude/rules/reuse.md`, `.claude/rules/semantic-contracts.md` and
`.claude/rules/renames.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

<goal>
Every public function of `ebaml` that calls an LLM has two native forms: the sync function that exists today, and an
async function that awaits the async functions of `baml_sdk`. Neither form wraps the other: the async form does not run
the sync form in a thread, and the sync form does not start an event loop around the async form. Both forms share one
implementation of the pipeline logic (graph rendering, the shrinking retry, the vision fallback, the facts, the
parsing), so a fix in one reaches the other, and the two forms give the same result for the same request. The MCP
server uses the async forms, so it serves several requests at the same time. An application that already runs an event
loop (a web server, an agent framework, a notebook) awaits EBaMl's LLM requests instead of blocking on them.

`describe_ebm` also changes in both forms: a failure of one feature no longer discards the work on the others. The
description covers the features that succeeded and names the features that failed.
</goal>

<decisions>
The user made these decisions. They are settled: implement them as written, and do not ask again. If one of them
turns out to be impossible, stop and report the reason to the user with the evidence.

| Topic | Decision |
| --- | --- |
| Public names | A suffix: `describe_graph_async`, `describe_ebm_async`, `explain_instance_async`, `propose_difficulties_async`. Each is exported from `ebaml`, next to its sync twin. |
| Scope | The four public functions only. `sample_graph_tokens` and `chat_with_mcp` stay sync. Streaming (`_stream`, `on_event`) is out of scope. |
| One implementation | Generator pipelines with a sync and an async driver (decision details below). |
| Module layout | Three modules: the pipelines, the sync drivers, the async drivers (details below). |
| Failures in `describe_ebm` | Partial results, in **both** forms, so the two forms keep one contract. This is a **BREAKING** change of `describe_ebm` and `EBMDescription`. |
| Which failures are partial | Per-feature failures only. A failure that hits every request raises at once (table below). |
| Summary after failures | Summarize the features that succeeded, and name the failed ones in the summary prompt. If every feature failed, raise the error of the first failed feature. |
| CPU work in the async form | Run it inline, in the event loop. Document that these steps block the loop. Do not add `asyncio.to_thread` in this milestone: the benchmarks of 0.16.0 measure the steps and decide. |
| Sync form inside a running loop | If a `_sync` function of `baml_sdk` fails inside a running event loop (Jupyter, marimo), detect the running loop and raise a clear `EBaMlError` subclass that tells the user to await the `_async` form. Do not run the sync call in a thread. |
| Pull requests | Three pull requests, in order (see `<release>`). |

### One implementation for both forms

Write each pipeline once, with no I/O. Each pipeline is a generator that yields an `SDKCall` (an attrs value with the
name of the `baml_sdk` function and its keyword arguments) and receives the reply through `send`, or the error through
`throw`. The sync driver calls the `<name>_sync` function of `baml_sdk`, and the async driver awaits `<name>`. The
shrinking retry (`retry.py`), the vision fallback (`graph_views.py`) and the provider error mapping
(`provider_errors.py`) live in the generators, once. A unit test can drive a pipeline with plain replies and no fake of
`baml_sdk`. Search for an existing contract first (`reuse.md`), and register `SDKCall` and the driver contract in
`architecture/abstractions.yml`.

### Module layout

- A new module of `ebaml.llm` holds the pipelines and `SDKCall`. It does not import `baml_sdk`. The `baml_sdk` types
  that it needs reach it as values from the drivers. If it cannot work that way, stop and report it.
- `descriptors.py` keeps the sync drivers and the sync public functions.
- A new module holds the async drivers and the async public functions. It imports `baml_sdk`. Add it to the
  `ignore_imports` list of the import-linter contract with a `reason:` block, and update `architecture.md` and
  `CLAUDE.md`, which say that only `llm/descriptors.py` imports `baml_sdk`.
- Each module name must be new in `src/ebaml` (`architecture.md`), and each file stays at 800 lines or fewer (`size.md`).

### Partial results of `describe_ebm`

- Add a field to `EBMDescription` that names each feature whose description failed, with the error, for example
  `failed_features: list[FeatureFailure]`. `FeatureFailure` is an attrs class with the feature name and the
  `LLMRequestError` (`types.md`). Choose the names by `naming.md` and `renames.md`, and use the same name in the prompt
  key and the MCP output.
- Pass the names of the failed features to `summarize_ebm` in `baml_src/talk_to_ebm.baml`, as `omitted_terms` and
  `omitted_features` are passed today, so that the summary says which features it does not cover (semantic contract
  4). Extend `omitted_parts_msg` and add an offline BAML test that fails if the prompt leaves the failed features out.
- Log a `features_failed` warning that names them, like `interactions_skipped`.
- The MCP tools that call `describe_ebm` report the failed features in their output.
- Decide per-feature or global by a property of the error class (`solid.md`, "Tell, don't ask"), for example a class
  attribute `affects_every_request` on `LLMRequestError` and its subclasses. Do not branch on `isinstance` in the
  pipeline.

| Error | Kind | Why |
| --- | --- | --- |
| `MissingAPIKeyError`, `APIKeyRejectedError` | Global: raise at once | Every request fails the same way. |
| `ProviderUnreachableError` | Global: raise at once | No request reaches the provider. |
| `VisionUnsupportedError` | Global: raise at once | It concerns the model, not one graph. The vision fallback runs first. |
| `RateLimitedError` after the retries | Per feature | Other features can succeed. |
| A reply still cut off after the shrinking retry, or a reply that does not parse | Per feature | It concerns one graph. |

- When a global error happens, the sync form cancels the features that have not started, and the async form cancels
  the task group. The caller gets that one error, not an `ExceptionGroup`.
- If every feature fails with per-feature errors, raise the error of the first failed feature, in the order of the
  features, and send no summary request.
- `describe_graph` describes one feature, so it raises its error as it does today, in both forms.
</decisions>

<context>
- The public functions that call an LLM are `describe_graph`, `describe_ebm`, `explain_instance` and
  `propose_difficulties` in `src/ebaml/llm/descriptors.py`. `src/ebaml/__init__.py` loads them on first use through the
  `loaders` table of `__getattr__`, because the import of `descriptors` can run `baml generate`. Each takes one request
  class from `src/ebaml/llm/requests.py` (`DescribeGraphRequest`, `DescribeEBMRequest`, `ExplainInstanceRequest`,
  `ProposeDifficultiesRequest`). Both forms take the same request classes and return the same result classes.
- `baml_sdk` already generates both forms of each BAML function: `describe_graph_cot` is a coroutine function and
  `describe_graph_cot_sync` is its sync twin (`src/baml_sdk/__init__.pyi`). EBaMl calls only the `_sync` forms today.
- `describe_ebm` describes the features at the same time with a `ThreadPoolExecutor` of `max_concurrency` workers
  (`_describe_graphs`), counts the steps with `ebaml.llm.progress_bars.progress` in the thread that waits on
  `as_completed`, and then asks for the summary. It raises the first error and cancels the rest. progressbar2 is not
  thread safe (`progress-bars.md`).
- The async `describe_ebm_async` runs the features in an `asyncio.TaskGroup`, limits the requests in flight with an
  `asyncio.Semaphore(max_concurrency)`, and counts each finished feature in the event loop thread, so the progress bar
  needs no lock. It keeps the `max_concurrency` field of the request.
- The errors are in `src/ebaml/core/exceptions.py`. `src/ebaml/llm/provider_errors.py` maps provider errors to them.
- `src/ebaml/mcp/server.py` uses FastMCP. Its tools are sync functions except `save_model`, which runs a blocking step
  with `asyncio.to_thread`. `src/ebaml/mcp/llm_tools.py` calls `describe_ebm`, `describe_graph` and
  `propose_difficulties`. The tool names and argument schemas are public API, and a snapshot in the unit suite guards
  them.
- The semgrep rules ban a bare `asyncio.gather` or `create_task` and a `while True` loop with a sleep
  (`semgrep/ebaml.yml`).
- The unit tests fake `baml_sdk` with `fake_sdk` in `tests/unit/ebaml/conftest.py`, which checks each call against the
  generated signature. It does not cover a coroutine function yet. The integration tests run the real `baml_sdk` against
  the local server of `tests/conftest.py` (`fake_llm`). `tests/contract/test_sdk_calls.py` lists each called SDK
  function in `CALLED`. `tests/docs/conftest.py` fakes the LLM replies of the doc examples.
- `ebaml.benchmark` stands alone and takes a sync `AskQuestion`. Leave it sync. The benchmarks of 0.16.0 measure both
  forms.
</context>

<steps>
**Pull request 1: shared pipelines, no change of behavior** (`refactor(llm)`, label `no-changelog`)

1. Add `SDKCall`, the generator pipelines and the sync drivers in the layout of `<decisions>`. Move the logic of
   `descriptors.py` and `retry.py` into the pipelines. Every existing test passes unchanged, and `jscpd` finds no clone.
2. Add unit tests that drive each pipeline with plain replies and errors, with no fake of `baml_sdk`.
3. Register the shared contracts in `architecture/abstractions.yml`, and write the "Abstraction reuse" section of the
   PR.

**Pull request 2: the async API and the partial results**

4. Extend `fake_sdk` so that it checks an async SDK function against its generated signature and returns a coroutine.
   Add the async names to `CALLED` in `tests/contract/test_sdk_calls.py`. Give `tests/docs/conftest.py` async fakes of
   the LLM replies.
5. Add the async drivers and the four `_async` functions. Add each to `ebaml.__all__` and to the `loaders` table of
   `src/ebaml/__init__.py`, so that `import ebaml` stays fast. Give each a docstring with an `Examples` section that runs
   under Sybil. Use `asyncio.run` in the example, so that the reader can copy it into a script. Say in the docstring
   that the CPU steps run in the event loop and block it.
6. Add the partial results of `describe_ebm` to both forms, the class attribute that marks a global error, the summary
   prompt change and its offline BAML test, and the `features_failed` warning. Run `baml generate`, `baml fmt`,
   `baml check` and `OPENROUTER_API_KEY=unused baml test -x root::Live`.
7. Detect a running event loop in the sync forms where a `_sync` call of `baml_sdk` fails there, and raise the new error
   with its fix step. Add a row for it in `BUILTIN_BASE` of `tests/api/test_exceptions.py`. Test it inside a running
   loop. If every `_sync` call works inside a running loop, add no error, and write that result in the PR.
8. Prove that the two forms agree. Write one parametrized unit test that runs each pair on the same requests with the
   same fake replies, and asserts the same result, the same SDK calls in the same order (for `describe_ebm`, the same
   set), and the same error class with the same fix steps for each provider failure, including each row of the error
   table. Add an API test that fails when a public sync function that calls an LLM has no `_async` twin.
9. Test the partial results in both forms: one failed feature, all features failed (the first error, no summary
   request), and each global error (raised at once, the features that have not started are cancelled). Assert that the
   summary prompt names the failed features.
10. Keep the semantic contracts for both forms. Parametrize the guards of contract 4 in
    `tests/unit/ebaml/llm/test_descriptors.py` over both forms, and add a guard that a failed feature is named in the
    result and in the summary prompt. The facts come from the same code, so contracts 1, 2, 3, 5, 6 and 7 need no new
    oracle. Say so in the PR.
11. Write the integration tests. Run the async forms against `fake_llm`. Assert the number of requests, the turns of
    the conversation, and that `describe_ebm_async` never has more than `max_concurrency` requests in flight: record the
    peak in the local server, and give it a delay so that requests overlap. Assert that a cancelled call sends no further
    requests and leaves no task running. Give each test one reason to fail (`testing.md`).
12. Give each async function the same `progress_bar` keyword as its twin. Write the three tests of `progress-bars.md`
    for `describe_ebm_async`.
13. Update `tests/api/public_api.json` with `EBAML_UPDATE_SNAPSHOTS=1 uv run pytest tests/api --no-cov`, and review the
    diff. Add a changelog fragment: the `_async` functions under `Added`, and a **BREAKING:** entry under `Changed` for
    the partial results of `describe_ebm` and the new field of `EBMDescription`.

**Pull request 3: the MCP server, the docs and the notebook** (completes the milestone)

14. Make the LLM tools of `server.py` async functions that await the `_async` forms through `llm_tools.py`. Keep every
    tool name and argument schema, and check the snapshot. Report the failed features in the output of the tools that
    describe an EBM. Add a test with two tool calls in flight at the same time.
15. Add `docs/async.md`: when to choose each form, one example for each function, concurrency with `max_concurrency`,
    the partial results, cancellation and timeouts with `asyncio.timeout`, the error of the sync form inside a running
    loop, and the CPU steps that block the loop. Add it to the nav of `zensical.toml`. Link it from `docs/usage.md` and
    `docs/mcp.md`. Update every doc page that shows `EBMDescription`.
16. Add a marimo notebook that explains several instances at the same time with the `_async` forms, with each paid
    call behind a button (`notebooks.md`), and register it.
</steps>

<release>
- Pull request 1 has no user-visible change. It gets the label `no-changelog` and no fragment.
- Pull request 2 adds the fragment of step 13.
- Pull request 3 completes the milestone: it sets `version = "0.15.0"` in `pyproject.toml` and runs `uv lock`. Its
  fragment lists the concurrent MCP tools and their failed-feature output under `Changed`. It marks the 0.15.0 row of
  `ROADMAP.md` as done and deletes this plan.
</release>

<done>
- Each public function that calls an LLM has a native sync form and a native `_async` form, and an API test keeps it
  so.
- One implementation of each pipeline serves both forms, and `jscpd` finds no clone.
- The two forms give the same results, SDK calls and errors in the unit tests. The async form keeps `max_concurrency`
  and stops cleanly when cancelled in the integration tests.
- `describe_ebm` in both forms returns partial results, names the failed features in the result, the summary and the
  log, raises a global error at once, and raises when every feature fails.
- The MCP server serves LLM tool calls at the same time, with the same tool names and schemas.
- `prek run --all-files`, `uv run pytest`, the integration, API and contract suites, the offline BAML tests, mutation
  testing of the changed lines and the strict docs build pass.
</done>
