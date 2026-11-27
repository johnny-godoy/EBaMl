# Accuracy benchmark

The accuracy benchmark follows the [benchmark of TalkToEBM](https://github.com/interpretml/TalkToEBM/tree/main/benchmark), which measures how
well an LLM reads the graphs of Explainable Boosting Machines. The data files are copied from there (TalkToEBM is MIT
licensed, see its `LICENSE`) and live in `benchmark/accuracy/data/` instead of `benchmark/benchmark/`.

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
uv run python benchmark/run.py accuracy             # every task
uv run python benchmark/run.py accuracy --task jumps --limit 5 --output results.json
uv run python benchmark/run.py accuracy --model anthropic/claude-haiku-4-5   # needs ANTHROPIC_API_KEY
uv run python benchmark/run.py accuracy --no-progress                       # no progress bar on the error stream
uv run python benchmark/run.py accuracy --mode vision --model openrouter/google/gemma-4-31b-it:free   # images, see below
```

Questions are asked one at a time (free models are rate limited), two LLM turns each, at temperature 0: the model
reasons step by step, then states its final answer in the format of the task. The default model is OpenRouter's
`openrouter/free`. `--model provider/model` selects another model. The provider is a BAML provider prefix, and the
README section "Choosing the model" lists the prefixes and the key variable of each one.

`--output` saves every prompt, response and verdict. Token usage is the one the provider reports.

## Graph modes

`--mode` sets how each graph reaches the model, as `GraphOptions(mode=...)` does for `describe_graph`:

| Mode | The request sends |
| --- | --- |
| `text` (default) | The question as the data file holds it. |
| `vision` | The question with the values of the graph replaced by a note, and the PNG image of the graph. |
| `mixed` | The question as it is, and the PNG image of the graph. |

The image needs the extra `vision` and a model that reads images. It has the confidence band of the question if the
question has bounds. The questions of the function recognition tasks write their graph as `Graph: {...}` with no bounds,
and the other tasks write the graph text of `EBMGraph.to_text`.

The scorer compares exact values. A value that the model reads from an image is an estimate, so `read-value`, `jumps`
and `wide-confidence` are hard in vision mode by design: they measure how far an image alone takes the model.

## Scoring

TalkToEBM's notebooks print each answer next to the solution and leave the comparison to the reader. Here
`ebaml.benchmark.is_correct` parses both and compares the values, so `(a)` matches `a)`, `0.50` matches `0.5` and
`Not monotone.` matches `Not monotone`, but `0.51` does not match `0.5`. An answer that cannot be parsed is wrong.

## Results of the graph modes

One run on 2026-11-26, with the model `openrouter/nvidia/nemotron-3.5-super-vl:free`, a free model that reads images.
Each mode asked the first 10 questions of each task (`--limit 10`), at temperature 0. The table gives the number of
correct answers out of 10.

| Task | `text` | `vision` | `mixed` |
| --- | :---: | :---: | :---: |
| `function-recognition` | 10 | 10 | 10 |
| `function-recognition-hard` | 10 | 10 | 10 |
| `jumps` | 10 | 2 | 10 |
| `monotonicity` | 10 | 9 | 10 |
| `read-value` | 10 | 0 | 10 |
| `wide-confidence` | 10 | 2 | 10 |
| All tasks | 60 | 33 | 60 |

- Vision mode is as accurate as the text on the tasks about the shape of a graph: the function and the monotonicity.
- Vision mode fails the tasks that need an exact value. Its answers are close estimates, for example `0.022` for the
  solution `0.02253`, and the scorer counts only an exact value as correct.
- Mixed mode is as accurate as text mode on these questions. The text alone already answers every question, so the
  run shows no gain from the image.
- A request in vision mode has fewer input tokens: about 13,000 for 10 questions of a task, against 30,000 to 90,000 in
  text mode. The image costs fewer tokens than the values of a long graph.
- 3 answers of vision mode ran out of the token limit, and the scorer counted them as wrong.

The sample is small: 10 questions for each task, one model and one run. The results do not show that one mode is better
for other models or other questions. Run the command again with a larger `--limit` before you choose a mode on these
numbers:

```bash
uv run python benchmark/run.py accuracy --limit 10 --mode vision --model openrouter/nvidia/nemotron-3.5-super-vl:free
```

## Left out

- The notebooks (`benchmark/notebooks/`) that generated the questions from an `all_graphs.pkl`, which is not in the
  upstream repository, so the questions cannot be regenerated here.
- The saved results for GPT-4 Turbo, GPT-3.5 and Gemini 1.5 (`benchmark/results/`). They hold the raw responses next to
  the solutions, with no scores, and they come from other models, so they are not a baseline for this benchmark.
- `benchmark_utils.py`, which loads and trains the models on the datasets that the generator notebooks used.
