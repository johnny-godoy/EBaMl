# Contributing

1. Install [uv](https://docs.astral.sh/uv/) and the BAML toolchain (see the [Setup](https://johnny-godoy.github.io/EBaMl/setup/) page), then run `uv sync`, `baml generate`
   and `prek install`. `prek install` adds the hooks that check each commit and its message.
2. Make your change with tests. Run `prek run --all-files` and fix every failure; do not bypass the hooks.
3. If a function has an invariant, write a Hypothesis property test for it. CI loads the `ci` profile (see
   `tests/unit/conftest.py`), so CI draws the same examples every time. If a property finds a bug, keep the failing
   example as a separate regression test.
4. To fake a `baml_sdk` function in a test, wrap the fake in `fake_sdk` from `tests/unit/ebaml/conftest.py`. The
   wrapper checks each call against the stub that `baml generate` writes, so a changed signature fails the test.
5. Choose the test suite by its question: `tests/unit`, `tests/integration`, `tests/api`, `tests/contract` or
   `tests/site` (`.claude/rules/testing.md`). An integration test uses the real `baml_sdk` and a local server in place of the LLM.
   A change to a public name changes `tests/api/public_api.json`.
6. A workaround (`# noqa`, `# type: ignore`, `pytest.mark.skip`, ...) needs a written reason next to it. Check with
   `uv run python -m scripts.ignores.check`.
7. For a user-visible change, run `uv run scriv create` and write the entries in the new file in `changelog.d/` (see
   `changelog.d/README.md`). Do not edit `CHANGELOG.md` or the version: the release workflow sets them after the merge.
   If the change is internal but touches `src/ebaml/`, `baml_src/` or `benchmark/`, add the label `no-changelog`.
8. Commit in small steps. Give each commit one purpose and a conventional commit message, for example
   `fix(graphs): keep the decimals of the confidence level`. `.claude/rules/commits-and-prs.md` has the rules.
9. Open a pull request with a conventional commit title, and fill the template. CI must pass.

## Check the documentation site

The site suite, `tests/site`, builds the docs and checks the built site in a browser. It checks each page with axe in
the light and the dark color scheme, the keyboard, the layout at a phone width of 360 pixels, the headings, the images,
the frames, the links, the HTML and the Lighthouse scores. CI runs it on each pull request that changes the docs.

1. Install the pinned Node tools: `npm ci --prefix tests/site`.
2. Install Chromium for Playwright: `uv run playwright install chromium`. If Chromium is already on the machine, set
   `EBAML_CHROMIUM` to its executable instead.
3. Run the suite: `uv run --group docs pytest tests/site --no-cov`. The first build exports the notebooks and is slow.
4. To check a site that you built before, add `--prebuilt-site`. To also check the external links, add
   `--external-links`. A weekly workflow runs that check, because a remote site can fail for a day.

A failure of axe gives one block for each rule that a page breaks:

```text
color-contrast (serious): Elements must meet minimum color contrast ratio thresholds
  https://dequeuniversity.com/rules/axe/4.14/color-contrast?application=axeAPI
    .md-nav__link--active > .md-ellipsis
  Fix any of the following:
  Element has insufficient color contrast of 3.75 (foreground color: #526cfe, background color: #eef0ff, ...)
```

- The first line gives the ID of the rule, its impact and what the rule asks for.
- The link opens the page of the rule, with the reasons and the fixes.
- The indented lines give a CSS selector for each element that breaks the rule.
- The last lines tell what is wrong with the first element, for example the two colors and their contrast ratio.

The test name gives the page and the color scheme, for example `[api-dark]`. Fix the cause in the Markdown, in
`docs/stylesheets/extra.css` or in `zensical.toml`. If the cause is in the theme and no setting can fix it, ask the
maintainer before you add an exception. Do not turn off a rule to make the suite pass.

## Change the default model

The default model is a maintainer choice. Users select another model with `model=` and need no change of the package.
The default sits in the `*Base` lines at the top of `baml_src/talk_to_ebm.baml`:

| Client | Used by | Default settings |
| --- | --- | --- |
| `RouterBase` | Turns 2 and 3 of a graph description, the shortening of the summary, and the local explanations | `temperature = 0.7`, `max_tokens = 12000` |
| `RouterLongBase` | Turn 1 of a graph description, and the draft of the summary | `temperature = 0.7`, `max_tokens = 16000` |
| `RouterBenchmarkBase` | The benchmark | `temperature = 0.0`, `max_tokens = 16000` |

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

`docs/releasing.md` describes the release gates and how to recover from a failed release or ship a hotfix.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
