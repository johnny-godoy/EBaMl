# Plan: self-correcting token estimates

You are a senior Python engineer who knows online regression and the BAML runtime. You will replace the fixed token
estimate of EBaMl with an estimate that the token counts of real LLM calls can correct. Read `CLAUDE.md` and every file
in `.claude/rules/` first, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

<context>
<current_state>
- `EBMGraph.to_text` (`src/ebaml/graphs/ebm_graph.py`) estimates the tokens of a graph text and simplifies a
  continuous graph until it fits `max_tokens`. No tokenizer runs before the call, so the estimate is a guard rail and
  the provider error is the real limit. The estimate comes through the `BudgetCheck` protocol, a function
  `(*, text, max_tokens, overbudget_percent) -> TokenBudget`.
- `ebaml.graphs.token_budget` holds the default check: `estimate_tokens` is 3 characters per token, rounded up, and
  `check_token_budget` returns a `BudgetStatus` with `"ok"`, `"warn"` or `"over"`. It needs no `baml_sdk`. A hypothesis
  test keeps it equal to `estimate_tokens` and `check_token_budget` in `baml_src/talk_to_ebm.baml`. PR #77 added this
  module. If `src/ebaml/graphs/token_budget.py` does not exist on `main`, stop and ask the user, because the plan
  builds on it.
- `llm/descriptors.py` passes `check_token_budget` to `to_text`, and `estimate_tokens` to
  `llm/retry.py:call_with_shrinking_graph`. After a cut-off reply, the retry asks for a graph of 75% of the estimate.
- `run_counted` in `baml_src/talk_to_ebm.baml` runs one LLM function and returns a `Counted` (the text and a
  `TokenUsage`). It reads `RunResult.usage`, which sums the counts of every request of the run per model, retries
  included. It turns a count that the provider did not report into 0.
</current_state>

<runtime_facts>
The generated stubs (`src/baml_sdk/ai/__init__.pyi`, `src/baml_sdk/ai/events/__init__.pyi`) show what one run
records. Check them again, because a new BAML toolchain can change them.

- `RunResult.journal.entries()` lists the events of a run. Each `ModelCall` event is one HTTP request, with
  `resolved_model` (`ModelKey`: `provider`, `model`), `usage` (`Usage`, or null) and `http_request` (`baml.http.Request`:
  `method`, `url`, `headers`, `body`).
- `Usage` has `input_tokens`, `uncached_input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`,
  `output_tokens` and `reasoning_tokens`. Each one can be null.
- `resolved_model` names the model that answered. The default client `openrouter/free` routes each request to some free
  model, so the client that the user chose does not say which tokenizer counted the tokens.
- `http_request.headers` holds the API key. Read only `body` and `resolved_model` from a `ModelCall`, and never log,
  store or return the headers, because a key in a log or a weights file leaks to everyone who reads it.
</runtime_facts>

<decisions>
The user decided these points. Keep them.

1. The estimate is a linear model of text features: `tokens = alpha * word_count + beta * byte_count`. Start from
   default weights, and correct them online from the token counts that the LLM calls report.
2. The package ships default weights and a table of precomputed weights per model. A user can turn the online update
   off, and a precomputed entry can have its update disabled.
3. The token counts come from the BAML runtime, because only a real call knows how its tokenizer counted the text.
4. This work is one PR that builds on PR #77 and replaces its fixed rule.
</decisions>

<open_questions>
Ask the user each question below before you build the part that depends on it, and record the answer in the PR
description. Each answer changes the public API.

1. Do learned weights persist between sessions? If yes, the user names the file, and the package writes it only when
   updates are on.
2. May the default `check_token_budget` give a different number than 3 characters per token? A changed estimate
   simplifies some graphs at a different size, which changes the text that the LLM reads.
</open_questions>
</context>

<design>
<estimator>
Put the estimator in `ebaml.graphs`, because `to_text` uses it and domain code must not import `baml_sdk`
(`.claude/rules/architecture.md`). It takes plain numbers and knows nothing about BAML.

- Fit `tokens = gamma + alpha * word_count + beta * byte_count`. The intercept `gamma` is a recommendation, not a user
  decision: the count of a request covers the system message, the template and the message overhead, not only the
  graph. Measure whether it helps (see `<steps>`), and drop it if it does not.
- Word and byte counts grow together for one kind of text, so a plain online fit of both is unstable. Use recursive
  least squares with a prior that pulls toward the default or precomputed weights, so a few samples move the weights
  a little and many samples move them more. Graph text is mostly digits and punctuation. If you find features that
  separate better (counts of digits, punctuation, non-ASCII bytes), report the measurement to the user before you
  change the formula, because the formula is a user decision.
- An estimate that is too low lets an oversized graph through, and one that is too high only simplifies a graph a
  little more. So the budget check reads an upper bound (the fit plus a margin from the spread of the residuals), not
  the mean.
- Make the estimator an object that satisfies `BudgetCheck`, so `to_text(check_budget=estimator)` works with no change
  to `to_text`. Keep `estimate_tokens` and `check_token_budget` as module functions that use the default weights, so
  callers of PR #77 keep working.
- Updates are off by default. `to_text` simplifies a graph according to the estimate, so weights that drift make the
  same model and options give a different graph text over time. A user who wants learning turns it on and passes the
  object on.
- `describe_ebm` describes features in a thread pool, so guard the update with a lock.
- Return an attrs class for the weights and for each sample (`.claude/rules/types.md`), and give every hint its exact
  type, because `beartype` checks them at run time.
</estimator>

<samples>
- Collect one sample per `ModelCall` that has a usage. The target is the input count with the cache counts included:
  `uncached_input_tokens + cached_input_tokens + cache_write_input_tokens` when the provider reports them, else
  `input_tokens`. Skip a call whose counts are all null, and never use 0 for a missing count, because a 0 pulls the
  weights toward zero.
- Compute the features on the text of the messages in the request body, not on the raw JSON, because the JSON syntax
  and the request parameters are not part of the prompt that the tokenizer counts.
- Key the sample by `resolved_model`, so each model learns its own weights and `openrouter/free` does not mix
  tokenizers.
- Return the samples from the BAML entry points so Python can update the estimator: for example a field
  `samples: TokenSample[]` next to `usage` in `Counted`. A new field is additive. A renamed or removed field or entry
  point breaks the public API (`.claude/rules/changelog-and-versioning.md`).
- Put the code that reads the samples and updates the estimator in `ebaml.llm`, in `descriptors.py` or a new module of
  that layer. Only `llm/descriptors.py` may import `baml_sdk`.
</samples>

<precomputed_weights>
- Store the default weights and the table per model as data in the package (a JSON file read with `orjson`), each
  entry with the model key, the weights, whether its update is allowed, and how many samples it came from.
- Add a script in `scripts/` that fits the table from collected samples. Fitting sends no requests. Collecting samples
  sends real requests, so the collection step needs `OPENROUTER_API_KEY` and a user who agreed to pay for it.
</precomputed_weights>
</design>

<steps>
1. Measure before you build. Collect samples (features and input counts) from real calls, then compare the fixed rule
   with the fit with and without `gamma`, per model, on held-out samples. Report the error of each in the PR
   description. The rule in `.claude/rules/writing-style.md` applies: do not change an estimate on taste alone. Sending
   requests costs money, so ask the user before you collect, and say how many requests you plan.
2. Write the estimator and its unit tests in `ebaml.graphs`.
3. Change `run_counted` and the entry points in `baml_src/` to return the samples. Run `baml generate`, `baml fmt`,
   `baml check` and `OPENROUTER_API_KEY=unused baml test -x root::Live`. Add an offline BAML test in
   `baml_src/talk_to_ebm_test.baml` for a call with null counts.
4. Feed the samples to the estimator in `ebaml.llm`, with updates off by default.
5. Write the default weights, the table and the fitting script.
6. Update the docs and the changelog (see `<documentation>`).
</steps>

<tests>
- Unit tests for the estimator: the prior holds the weights near the defaults after one sample, many consistent
  samples move them to the true slope, a disabled update changes nothing, and concurrent updates from threads lose no
  sample.
- A test that a `ModelCall` with null counts gives no sample, and a test that the cache counts enter the target.
- A test that no sample, log event or weights file holds a header of `http_request`. Use a fake request whose headers
  hold a marker string, and check that the marker appears nowhere.
- Keep the hypothesis test of PR #77 for the default check, or replace it with one that states the new default rule.
- 100% coverage, and zero surviving mutants for the changed modules (`uv run python scripts/mutation.py --base
  origin/main`).
</tests>

<documentation>
- Update the "Token budget" item of the README section between the `behaviour` markers: what the estimate counts, the
  default weights, how to turn on the update, and that a learning estimator makes the graph text change over time.
- Add the estimator to `docs/api.md` under "Token budget".
- Write the prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
- Add a changelog fragment (`uv run scriv create`). Mark a changed default estimate under `Changed`.
</documentation>

<done>
- The PR description shows the measured error of the fixed rule and of the fit, per model, and the user's answers to
  the open questions.
- `to_text(check_budget=estimator)` works with a learning estimator, and the module functions use the default
  weights.
- No sample, log event or file holds an API key.
- `prek run --all-files`, `uv run pytest`, `uv run pytest README.md docs src --no-cov`, the offline BAML tests,
  mutation testing and the strict docs build pass.
</done>
