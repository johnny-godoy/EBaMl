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
implementation of the pipeline logic (graph rendering, the shrinking retry, the facts, the parsing), so a fix in one
reaches the other, and the two forms give the same result for the same request. The MCP server uses the async forms,
so it serves several requests at the same time. An application that already runs an event loop (a web server, an agent
framework, a notebook) calls EBaMl without blocking the loop.
</goal>

<context>
- The public functions that call an LLM are `describe_graph`, `describe_ebm`, `explain_instance` and
  `propose_difficulties` in `src/ebaml/llm/descriptors.py`. `src/ebaml/__init__.py` loads them on first use through the
  `loaders` table of `__getattr__`, because the import of `descriptors` can run `baml generate`. Each takes one request
  class from `src/ebaml/llm/requests.py` (`DescribeGraphRequest`, `DescribeEBMRequest`, `ExplainInstanceRequest`,
  `ProposeDifficultiesRequest`). `sample_graph_tokens` and `chat_with_mcp` call an LLM too, but they are not in
  `ebaml.__all__`: `chat_with_mcp` serves the `ebaml-chat` command. Decide with the user whether they get an async form.
- `baml_sdk` already generates both forms of each BAML function: `describe_graph_cot` is a coroutine function and
  `describe_graph_cot_sync` is its sync twin (`src/baml_sdk/__init__.pyi`). EBaMl calls only the `_sync` forms today.
  The async forms also accept `on_event`, and a `_stream` form exists. Streaming is out of scope for this milestone.
- `describe_ebm` describes the features at the same time with a `ThreadPoolExecutor` of `max_concurrency` workers
  (`_describe_graphs`), counts the steps with `ebaml.llm.progress_bars.progress` in the thread that waits on
  `as_completed`, and then asks for the summary. progressbar2 is not thread safe (`progress-bars.md`).
- `src/ebaml/llm/retry.py` holds `call_with_shrinking_graph`, which calls the LLM again with a smaller graph when a
  reply is cut off. `src/ebaml/llm/provider_errors.py` turns provider errors into `LLMRequestError` subclasses with a
  sync context manager. `src/ebaml/llm/graph_views.py` falls back from vision to text mode when a model has no image
  input (`job.switch.fall_back`). All of this logic must work in both forms without a copy.
- `descriptors.py` has 755 lines. The size limit is 800 lines (`size.md`), so the async form cannot go into the same
  file. The import-linter contract lets only `llm/descriptors.py` import `baml_sdk` (`architecture.md`). `jscpd`
  fails on a copy of 50 tokens and 5 lines in `src`, so a second copy of each function in a new file fails the hooks.
- `src/ebaml/mcp/server.py` uses FastMCP. Its tools are sync functions except `save_model`, which runs a blocking step
  with `asyncio.to_thread`. `src/ebaml/mcp/llm_tools.py` calls `describe_ebm`, `describe_graph` and
  `propose_difficulties`. The tool names and argument schemas are public API, and a snapshot in the unit suite guards
  them.
- The semgrep rules ban a bare `asyncio.gather` or `create_task` and a `while True` loop with a sleep (`semgrep/ebaml.yml`).
  Use `asyncio.TaskGroup`.
- The unit tests fake `baml_sdk` with `fake_sdk` in `tests/unit/ebaml/conftest.py`, which checks each call against the
  generated signature. It calls the fake and returns its result, so it does not cover a coroutine function yet. The
  integration tests run the real `baml_sdk` against the local server of `tests/conftest.py` (`fake_llm`).
  `tests/contract/test_sdk_calls.py` lists each called SDK function in `CALLED`. `tests/docs/conftest.py` fakes the LLM
  replies of the doc examples.
- `ebaml.benchmark` stands alone and takes a sync `AskQuestion`. The benchmark milestone (0.16.0) comes after this one
  and measures both forms, so leave `ebaml.benchmark` sync here.
</context>

<design>
Settle these decisions with the user before you write code. Show the options with a short example of each, and give
your recommendation.

1. **The public names.** Candidates: a prefix (`adescribe_ebm`, as in Django and LangChain), a suffix
   (`describe_ebm_async`), or a namespace (`from ebaml.aio import describe_ebm`). A namespace keeps one name for one job
   and gives a clean `__all__`, but two functions with the same name can confuse a reader of a traceback. A prefix
   sorts next to its sync twin in the docs. Whatever the choice, the result classes (`EBMDescription`,
   `LocalExplanation`, `DifficultyProposal`) and the request classes stay the same for both forms.
2. **One implementation for both forms.** Write each pipeline once, with no I/O, and drive it from two thin drivers.
   Recommended form: each pipeline is a generator that yields an `SDKCall` (an attrs value with the name of the SDK
   function and its arguments) and receives the reply or the error through `send` and `throw`. The sync driver calls the
   `_sync` function, and the async driver awaits the async function. The shrinking retry, the vision fallback and the
   provider error mapping then live in the generator, once. Compare this with the other options (a callable parameter
   for the SDK call, or code generation) and say why you chose one. Register the new shared contract in
   `architecture/abstractions.yml` (`reuse.md`) and search for an existing one first.
3. **The module layout.** Recommended layout: the pipelines move to a new module of `ebaml.llm` that does not import
   `baml_sdk`. `descriptors.py` keeps the sync drivers and public functions, and a new module holds the async ones.
   Only the two driver modules import `baml_sdk`. Add the second module to the `ignore_imports` list of the
   import-linter contract with a `reason:` block, and update `architecture.md` and `CLAUDE.md`, which say that only
   `llm/descriptors.py` imports `baml_sdk`. Each module name must be new in `src/ebaml` (`architecture.md`).
4. **CPU work inside the async form.** The facts of a local explanation run an OR-Tools search and a sampler, and a
   large graph takes time to render. Such a step blocks the event loop. Measure each step on the largest model of the
   tests. Run a step that takes more than about 50 ms with `asyncio.to_thread`, and write the measurement in a comment.
   The step itself stays the same function in both forms.
5. **Concurrency in `describe_ebm`.** The async form runs the features in an `asyncio.TaskGroup` and limits the
   requests in flight with an `asyncio.Semaphore(max_concurrency)`. It keeps the `max_concurrency` field of the request,
   so one request class serves both forms. It counts each finished feature in the event loop thread, so the progress bar
   needs no lock. If one feature fails, the task group cancels the others, and the error that reaches the caller is the
   same `LLMRequestError` as in the sync form, not an `ExceptionGroup`. Decide with the user how the async form reports
   a failure of several features at once.
</design>

<steps>
1. **Write the design note.** Put the decisions of `<design>` in the PR description, section "Abstraction reuse", and
   wait for the approval of the user.
2. **Refactor first, with no change of behavior.** Move the pipeline logic of `descriptors.py` and `retry.py` into the
   shared form, with the sync driver only. Every existing test must pass unchanged. Commit this as `refactor(llm)`
   commits of 400 lines or fewer, each with one purpose (`commits-and-prs.md`).
3. **Teach the test fakes about coroutines.** Extend `fake_sdk` so that it checks an async SDK function against its
   generated signature and returns a coroutine. Add the async names to `CALLED` in `tests/contract/test_sdk_calls.py`.
   Give `tests/docs/conftest.py` async fakes of the LLM replies.
4. **Add the async drivers and the public names.** Add each async function to the module of decision 3, to
   `ebaml.__all__` and to the `loaders` table of `src/ebaml/__init__.py`, so that `import ebaml` stays fast. Give each a
   docstring with an `Examples` section that runs under Sybil. Use `asyncio.run` in the example, so that the reader can
   copy it into a script.
5. **Prove that the two forms agree.** Write one parametrized unit test that runs each pair on the same requests with
   the same fake replies, and asserts the same result, the same SDK calls in the same order (for `describe_ebm`, the
   same set), and the same error class with the same fix steps for each provider failure. Add an API test that fails
   when a public sync function that calls an LLM has no async twin, so that the next LLM function gets both forms.
6. **Keep the semantic contracts for both forms.** Parametrize the guards of contract 4 in
   `tests/unit/ebaml/llm/test_descriptors.py` over both forms: an async entry point must name or reject an interaction
   or a removed term, as the sync one does. The facts come from the same code, so contracts 1, 2, 3, 5, 6 and 7 need no
   new oracle. Say so in the PR.
7. **Write the integration tests.** Run the async forms against `fake_llm`. Assert the number of requests, the turns of
   the conversation, and that `describe_ebm` never has more than `max_concurrency` requests in flight. Record the peak
   in the local server, and give it a delay so that requests overlap. Assert that a cancelled call sends no further
   requests and leaves no task running. Give each test one reason to fail (`testing.md`).
8. **Check the sync form inside a running loop.** A user of Jupyter or marimo calls the sync form from a cell while an
   event loop runs. Check that each `_sync` function of `baml_sdk` works there, and add a test. If it fails, report it to
   the user with the error. Do not wrap the sync form in a thread to hide it.
9. **Move the MCP server to the async forms.** Make the LLM tools of `server.py` async functions that await the async
   forms through `llm_tools.py`. Keep every tool name and argument schema, and check the snapshot. Add a test with two
   tool calls in flight at the same time.
10. **Progress bars.** Give each async function the same `progress_bar` keyword as its twin. Write the three tests of
    `progress-bars.md` for the async form of `describe_ebm`.
11. **Document it.** Add `docs/async.md`: when to choose each form, one example for each function, concurrency with
    `max_concurrency`, cancellation and timeouts with `asyncio.timeout`, and the CPU steps that run in a thread. Add it
    to the nav of `zensical.toml`. Link it from `docs/usage.md` and `docs/mcp.md`.
    Add a marimo notebook that explains several instances at the same time, with each paid call behind a button
    (`notebooks.md`), and register it.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.15.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment lists the async functions under `Added`, and the concurrent MCP tools under `Changed`. If decision 2 changes a
public signature, mark the entry **BREAKING:**. Update `tests/api/public_api.json` and review its diff. Mark the 0.15.0
row of `ROADMAP.md` as done, and delete this plan. The refactor of step 2 can come in an earlier pull request with the
label `no-changelog`.
</release>

<done>
- Each public function that calls an LLM has a native sync form and a native async form, and an API test keeps it so.
- One implementation of each pipeline serves both forms, and `jscpd` finds no clone.
- The two forms give the same results, SDK calls and errors in the unit tests, and the async form keeps
  `max_concurrency` and stops cleanly when cancelled in the integration tests.
- The MCP server serves LLM tool calls at the same time, with the same tool names and schemas.
- `prek run --all-files`, `uv run pytest`, the integration, API and contract suites, mutation testing of the changed
  lines and the strict docs build pass.
</done>
