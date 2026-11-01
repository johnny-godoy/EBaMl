### Added

- `TokenEstimator` estimates the tokens of a graph text from its words, bytes and digits, and is a `BudgetCheck`. It can learn its weights from the token counts of real calls with `observe`.
- `GraphOptions(token_learner=TokenLearner(learning=True))` makes `describe_graph` and `describe_ebm` check the token budget with the weights of the model that you chose, and learn the weights of each model that answers. The learned weights persist in `~/.cache/ebaml/token_weights.json`. Learning is off by default.
- A table of token weights for each model ships with the package (`src/ebaml/llm/token_weights.json`), and `scripts/fit_token_weights.py` adds a new model to it.
- `sample_graph_tokens` sends one request with a graph text and returns its token samples, at a small cost.
- `TokenUsage.samples` lists the token samples of the calls of a run. Each sample has the provider, the model, the request body and the input tokens. It never holds a header.
- `TokenFitError` is raised when token weights cannot be fitted or read.

### Changed

- The default token estimate of a graph text is now 4.13 tokens for each word and 0.27 for each digit, with a margin above the fit, and no longer 1 token for 3 characters. The old rule gave 43% too few tokens on real graph texts. Graphs near `max_tokens` are now simplified at a different size, so the text that the LLM reads can differ.

### Removed

- **BREAKING:** The BAML functions `estimate_tokens` and `check_token_budget` and the BAML class `TokenBudget` are gone. Use `ebaml.graphs.token_budget` in Python.
- **BREAKING:** `TokenUsage` has a new required field `samples`, so code that builds a `TokenUsage` must pass it.
