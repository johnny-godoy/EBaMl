### Added

- `describe_graph`, `describe_ebm` and `explain_instance` take `model=` to choose the LLM of one call, as `ModelOptions` or
  the `"provider/model"` shorthand. The default model stays OpenRouter's `openrouter/free`.
- `ebaml.llm.options.ModelOptions` sets the provider, the model, the environment variable of the key, the base URL,
  the temperature, the output limit and the headers.
- `benchmark/run.py --model provider/model` runs the benchmark on another model.

### Changed

- The BAML entry points `describe_graph_cot`, `summarize_ebm`, `describe_ebm`, `answer_benchmark_question` and
  `explain_local` take an optional `model: ModelSpec`. The new BAML function `check_model` rejects an unknown provider.
- EBaMl needs the BAML toolchain 0.23 and `baml-bridge` 0.23 or later.
