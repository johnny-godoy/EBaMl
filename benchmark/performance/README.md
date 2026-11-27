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

The profiles of the workloads and the table of the hotspots are in the next section of this page when they exist.
