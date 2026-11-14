# Test suites

Each suite answers one question, and each has its own guards. A change picks the suite by the question it must answer,
and a bug fix adds its guard to the suite where the bug shows (`bug-fixes.md`).

| Suite | Directory | Question | Faked | Guards |
| --- | --- | --- | --- | --- |
| Unit | `tests/unit/ebaml/` | Does one module do its job? | Its neighbours, and `baml_sdk` through `fake_sdk` | The mirrored layout (`check_layout`), 100 % coverage, mutation testing |
| Integration | `tests/integration/` | Do the parts work together, from an EBM to the text that the user reads? | Only the LLM, as a local server (`tests/conftest.py`) | `check_test_suites`: no fake of `baml_sdk` |
| API | `tests/api/` | Do the public names, signatures, errors, tools and commands keep their promises? | Nothing | `public_api.json`, the tool snapshot and `check_api_breaks` |
| Contract | `tests/contract/` | Does ebaml agree with what it does not own: the generated SDK, the provider protocol, the packaging? | Nothing, or the local server | The checks of the SDK against the sources, and the wire format |
| Docs | `README.md`, `docs/`, docstrings | Does each example in the docs run? | The LLM, as in `tests/docs/conftest.py` | Sybil |

`pytest` marks each test with the name of its suite (`tests/conftest.py`), so `uv run pytest -m integration` runs one
suite. `--strict-markers` rejects a misspelled name. CI runs the unit suite with the coverage gate, and the other
suites without coverage, so that a slow end-to-end test cannot stand in for a missing unit test.

## Integration tests

- Use the `fake_llm` and `llm_model` fixtures. The model is `openai-generic` with the address of the local server, and
  the fixture removes the keys of the real providers from the environment. A test that forgets the model finds no key,
  so a test never reaches a real provider.
- Run the real `baml_sdk` and the real BAML runtime. A fake of `baml_sdk` belongs to a unit test. `check_test_suites`
  fails on `fake_sdk` and on `setattr(baml_sdk, ...)` in this suite.
- Script the replies with `fake_llm.handler`, and read what the pipeline sent from `fake_llm.requests`. Assert on what
  crossed the boundary: the number of requests, the turns of a conversation, the facts in a prompt, the sampling
  options, and the error that the user sees.
- For each failure of a provider, assert the class of the error, the steps to fix it and the number of requests. A
  failure that is not retried (a rejected key) sends one request. A failure that is retried sends three.
- Give a test one reason to fail. A request count and a text are two tests.

## API tests

- A change of a public name changes `tests/api/public_api.json`. Review the diff, then update the snapshot with
  `EBAML_UPDATE_SNAPSHOTS=1 uv run pytest tests/api --no-cov`. A break needs a `**BREAKING:**` fragment
  (`changelog-and-versioning.md`).
- A new exception needs a row in `BUILTIN_BASE` of `test_exceptions.py`. A new command flag needs a row in
  `test_command_lines.py`. A new MCP tool needs a name in `test_mcp_tools.py` and a snapshot in the unit suite.

## Contract tests

- A contract test reads the other side from its source of truth: the stub of `baml_sdk`, the BAML files, the
  `pyproject.toml`, the requests of the local server. It never copies the other side into the test.
- A new call into `baml_sdk` needs a row in `CALLED` of `test_sdk_calls.py`, so that a reviewer sees each new crossing of
  the boundary.
- A value that two files must keep equal (the defaults of a model in `model_spec.baml` and in `talk_to_ebm.baml`) gets a
  test that reads both.
- If a contract test fails after `git pull`, run `baml generate` first: the generated SDK is not in git.

## Check

- `uv run python -m scripts.architecture.suites` checks the suites, the markers, `testpaths` and the CI step.
- Prove each new guard: break the code that it guards, watch the test fail, and restore the code (`bug-fixes.md`).
- Run the suites: `uv run pytest tests/integration tests/api tests/contract --no-cov`.
