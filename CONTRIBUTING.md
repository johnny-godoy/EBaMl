# Contributing

1. Install [uv](https://docs.astral.sh/uv/) and the BAML toolchain (see the README), then run `uv sync`, `baml generate`
   and `prek install`. `prek install` adds the hooks that check each commit and its message.
2. Make your change with tests. Run `prek run --all-files` and fix every failure; do not bypass the hooks.
3. If a function has an invariant, write a Hypothesis property test for it. CI loads the `ci` profile (see
   `tests/unit/conftest.py`), so CI draws the same examples every time. If a property finds a bug, keep the failing
   example as a separate regression test.
4. To fake a `baml_sdk` function in a test, wrap the fake in `fake_sdk` from `tests/unit/ebaml/conftest.py`. The
   wrapper checks each call against the stub that `baml generate` writes, so a changed signature fails the test.
5. A workaround (`# noqa`, `# type: ignore`, `pytest.mark.skip`, ...) needs a written reason next to it. Check with
   `uv run python -m scripts.check_ignores`.
6. For a user-visible change, run `uv run scriv create` and write the entries in the new file in `changelog.d/` (see
   `changelog.d/README.md`). Do not edit `CHANGELOG.md` or the version: the release workflow sets them after the merge.
   If the change is internal but touches `src/ebaml/`, `baml_src/` or `benchmark/`, add the label `no-changelog`.
7. Commit in small steps. Give each commit one purpose and a conventional commit message, for example
   `fix(graphs): keep the decimals of the confidence level`. `.claude/rules/commits-and-prs.md` has the rules.
8. Open a pull request with a conventional commit title, and fill the template. CI must pass.

## Change the default model

The default model is a maintainer choice. Users select another model with `model=` and need no change of the package.
The default sits in the `*Base` lines at the top of `baml_src/talk_to_ebm.baml`:

| Client | Used by | Default settings |
| --- | --- | --- |
| `RouterBase` | Turns 2 and 3 of a graph description, the shortening of the summary, and the local explanations | `temperature = 0.7`, `max_tokens = 6000` |
| `RouterLongBase` | Turn 1 of a graph description, and the draft of the summary | `temperature = 0.7`, `max_tokens = 8000` |
| `RouterBenchmarkBase` | The benchmark | `temperature = 0.0`, `max_tokens = 8000` |

1. Edit the `*Base` lines only. The clients `Router`, `RouterLong` and `RouterBenchmark` wrap them in
   `ai.clients.Retry`, so they keep their retry policy.
2. If you change a temperature or an output limit, change the same value in `short_defaults`, `long_defaults` or
   `benchmark_defaults` in `baml_src/model_spec.baml`. Those functions give the defaults of a call to a model that a user
   selects with `model=`.
3. Run `baml generate`, `baml fmt`, `baml check` and the offline tests: `OPENROUTER_API_KEY=unused baml test -x root::Live`.
4. A BAML client declaration can use `ai.clients.Fallback`, which Python cannot set up. For example, put the members
   `anthropic.Client.new(model = "claude-haiku-4-5", api_key = env.ANTHROPIC_API_KEY, ...)` and
   `openai.OpenRouterClient.new(model = "openrouter/free", api_key = env.OPENROUTER_API_KEY, ...)` in the `members` list.

To add a provider for `model=`, add a builder function and a row in `client_builders` in `baml_src/model_spec.baml`.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
