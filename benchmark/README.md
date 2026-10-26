# benchmark

This benchmark follows the [benchmark of TalkToEBM](https://github.com/interpretml/TalkToEBM/tree/main/benchmark), which measures how
well an LLM reads the graphs of Explainable Boosting Machines. The data files are copied from there (TalkToEBM is MIT
licensed, see its `LICENSE`) and live in `data/` instead of `benchmark/benchmark/`.

## Tasks

Each file is a JSON list of `[prompt, solution]` pairs. The prompt holds the graph and ends in "Think step by step."

| Task | Questions | The model has to... | Solution |
| --- | --- | --- | --- |
| `function-recognition` | 50 | pick the function that the graph depicts | a letter, `a)` to `e)` |
| `function-recognition-hard` | 30 | the same, with harder functions | a letter |
| `jumps` | 100 | find the x-axis position of the largest jump | a number |
| `monotonicity` | 100 | tell if the graph is increasing, decreasing or not monotone | `Increasing`, `Decreasing` or `Not monotone` |
| `read-value` | 100 | read the mean value at an x-axis position | a number |
| `wide-confidence` | 100 | find the interval with the widest confidence bound | an interval, `(lower, upper)` |

## Running

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python benchmark/run.py                       # every task
uv run python benchmark/run.py --task jumps --limit 5 --output results.json
uv run python benchmark/run.py --model anthropic/claude-haiku-4-5   # needs ANTHROPIC_API_KEY
```

Questions are asked one at a time (free models are rate limited), two LLM turns each, at temperature 0: the model
reasons step by step, then states its final answer in the format of the task. The default model is OpenRouter's
`openrouter/free`. `--model provider/model` selects another model. The provider is a BAML provider prefix, and the
README section "Choosing the model" lists the prefixes and the key variable of each one.

`--output` saves every prompt, response and verdict. Token usage is the one the provider reports.

## Scoring

TalkToEBM's notebooks print each answer next to the solution and leave the comparison to the reader. Here
`ebaml.benchmark.is_correct` parses both and compares the values, so `(a)` matches `a)`, `0.50` matches `0.5` and
`Not monotone.` matches `Not monotone`, but `0.51` does not match `0.5`. An answer that cannot be parsed is wrong.

## Left out

- The notebooks (`benchmark/notebooks/`) that generated the questions from an `all_graphs.pkl`, which is not in the
  upstream repository, so the questions cannot be regenerated here.
- The saved results for GPT-4 Turbo, GPT-3.5 and Gemini 1.5 (`benchmark/results/`). They hold the raw responses next to
  the solutions, with no scores, and they come from other models, so they are not a baseline for this benchmark.
- `benchmark_utils.py`, which loads and trains the models on the datasets that the generator notebooks used.
