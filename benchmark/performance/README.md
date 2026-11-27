# Performance benchmark

The performance benchmark measures the time of the work of EBaMl around the LLM requests: the graph text, the facts
of a local explanation, an edit, the LLM calls with a local fake LLM, the MCP tools and the import of the package. The
LLM request itself takes most of the time of a real call, and EBaMl does not control it. The benchmark sends no real
request and needs no key.

## Run it

```bash
uv run python benchmark/run.py performance --output performance.json        # every workload
uv run python benchmark/run.py performance --workload "llm.*" --workload "graphs.render.titanic"
uv run python benchmark/run.py performance compare base.json head.json --markdown report.md
```

The first run fits the models and saves them in `benchmark/performance/.models/`, which takes about a minute. The
Titanic model needs the network once, to download the data from OpenML. Every later run loads the saved files, so two
runs measure the same models.

`--workload` takes a shell pattern of the workload names, and you can repeat it. pytest-benchmark runs one warm-up
round and then the rounds of each workload. The summary JSON has the same shape for every run: the median, the
interquartile range, the minimum and the mean time of each workload, with the commit, the Python version and the
machine.

## Workloads

`benchmark/performance/workloads.py` lists the workloads. A name is `group.call.options`.

| Group | Workloads | It measures |
| --- | --- | --- |
| `import` | `import.ebaml`, `import.ebaml_llm` | A new interpreter that imports `ebaml`, and that also loads the LLM functions and `baml_sdk`. |
| `graphs` | `graphs.render.<model>` | The graph text of every feature, as `describe_ebm` renders it before the requests. |
| `local` | `local.facts.<model>` | `compute_local_facts` for the first row, with the default options (the MIP search and 10,000 samples). |
| `edit` | `edit.monotonize.<model>` | A new `EditSession`, one `Monotonize` edit and its `edit_effect` on the training data. |
| `llm` | `llm.sync.<call>.<model>`, `llm.async.<call>.<model>` | `describe_ebm`, `explain_instance` and `propose_difficulties` with the fake LLM, in the sync and the async form. |
| `concurrency` | `concurrency.<form>.describe_ebm.max_concurrency-<n>` | `describe_ebm` of 100 features with 1, 4 and 16 requests at the same time. Each reply waits 10 ms, so the requests overlap. |
| `mcp` | `mcp.<tool>`, `mcp.session` | One call of each MCP tool on the Titanic model, through an in-memory client. `mcp.session` lists the tools only: it measures the start of the session that each call includes. |

The models are in `benchmark/performance/models.py`. Each has a fixed seed and no interactions.

| Model | Kind | Features | Bins |
| --- | --- | --- | --- |
| `titanic` | Binary classifier, the data of the notebooks | 7, two of them categorical | Default |
| `diabetes` | Regressor, the diabetes data of scikit-learn | 10 | Default |
| `multiclass` | Classifier with 5 classes, synthetic | 8 | 32 |
| `categorical` | Binary classifier, synthetic | 3 categorical with 50, 200 and 500 levels, and 3 continuous | Default |
| `features-<n>-bins-<b>` | Binary classifier, synthetic | 10, 100 or 500 | 32 or 1024 |

## Compare two runs

`compare` reads the median of each workload. A workload is slower when its median grows by more than the threshold
(25 % by default) and by more than the floor (2 ms by default). The floor keeps the jitter of the workloads that take a
few milliseconds out of the verdict. `--gate` makes the command exit with code 1 when a workload is slower.

## Continuous integration

`.github/workflows/performance.yml` runs the benchmark on a pull request that changes the code, every Monday, and on
demand.

- On a pull request, one job measures the base and the head on the same runner, with the workloads of the head. The
  base runs with its source first on `PYTHONPATH`, in the environment of the head. So the comparison shows the change
  of the code of EBaMl, and not a change of a dependency.
- The job writes the table of the changes to the run summary and to a comment on the pull request. It fails only when
  a workload is slower. It never compares with an absolute time, because a CI runner changes its speed from one job to
  the next.
- Each run keeps its summaries as an artifact for 90 days.

Calibrate the threshold from the first ten runs: read the largest change of a workload between two runs of the same
commit, and set the threshold of `compare` in the workflow to about twice that change, but not below 25 %. Until then,
the check is not a required check of `main`.

## Profiles and hotspots

One run on 2026-11-27, on a Linux container with 4 cores and Python 3.13, at the commit of the 0.15.0 release. A
sampling profiler (pyinstrument) and a deterministic profiler (cProfile) ran each slow workload, and memray measured
the memory. The share is the part of the time of the workload that the function takes. A profiler adds time to the
Python code, so the table gives the shares that a prototype of the fix confirmed, where one exists.

| # | Function | Share of the time | Workload | Median |
| --- | --- | --- | --- | --- |
| 1 | `TextFeatures.of` (`graphs/token_weights.py`): a Python loop over each character to count the digits | 19 to 29 % | `graphs.render.*` | 0.1 to 1.3 s |
| 2 | `EBMGraph.simplify` (`graphs/ebm_graph.py`): a Python loop over each bin, for each of up to 101 levels | about 50 % after fix 1 | `graphs.render.features-100-bins-1024` | 1.3 s |
| 3 | `confidence_factor` (`graphs/graph_text.py`): `scipy.stats.norm` for each class at each level | 11 to 46 % | `graphs.render.multiclass` | 0.76 s |
| 4 | `xy_to_json` (`graphs/graph_text.py`): one `orjson.dumps` call for each value | about 18 % after fix 1 | `graphs.render.*` | |
| 5 | `eval_terms` of `interpret`, through 22 calls of `predict` and `predict_proba`, about half of them repeated | about 75 % | `edit.monotonize.features-100-bins-1024` | 0.47 s |
| 6 | `sample_completions` (`local/completions.py`): the sampled scores are gathered again at each step | about 45 % | `local.facts.features-100-bins-1024` | 0.11 s |
| 7 | The import of `ebaml`: `skops.io`, `pandas` and the metrics of scikit-learn load eagerly | 0.65 s over `interpret`, 0.44 s more for the LLM functions | `import.*` | 1.75 to 2.2 s |
| 8 | The JSON schemas of pydantic for about 20 tools, built at each `create_server` | about 40 % of a tool call | `mcp.*` | 40 ms |
| 9 | The checks of beartype | 15 to 37 % of a call of a few milliseconds, 7 % or less of a slow call | `mcp.*`, `local.facts.*` | |
| 10 | `describe_ebm` renders every graph before its first request | the whole rendering | `llm.*.describe_ebm.*` | |

The memory is not a hotspot. The largest peak is 203 MB, for an edit of the model with 500 features and 1024 bins. The
graph text of that model allocates 9 GB in total, but each text is short-lived, and the peak stays at 77 MB.

beartype stays on for the whole package. A run with the hook off in a scratch copy saves 7 % or less on the slow
workloads, 15 to 37 % on the calls of a few milliseconds (an MCP tool, the local facts of a small model), and 0.4 s on
the first load of the LLM functions.

### Sync and async forms

`describe_ebm` and `describe_ebm_async` take the same time, and so do `explain_instance` and its async form, because
the same pipeline runs in both. With 10 ms of delay in each reply, `describe_ebm` of 100 features takes 3.6 s with
`max_concurrency=1`, 1.0 s with 4 and 0.46 s with 16, in both forms.

The async forms run the CPU steps in the event loop. A task that sleeps 1 ms and records each gap measured how long
the loop is blocked:

| Model | `describe_ebm_async`: longest block | `explain_instance_async`: longest block |
| --- | --- | --- |
| `titanic` | 17 ms | 5 ms |
| `diabetes` | 110 ms | 8 ms |
| `multiclass` | 671 ms | 13 ms |
| `features-100-bins-1024` | 1357 ms | 116 ms |
| `features-500-bins-32` | 325 ms | 284 ms |
| `features-500-bins-1024` | (over the token budget) | 585 ms |

`describe_ebm_async` renders every graph before its first request, so it blocks the loop once, for the sum of the
rendering of all graphs. `explain_instance_async` blocks the loop for the time of the local facts.
