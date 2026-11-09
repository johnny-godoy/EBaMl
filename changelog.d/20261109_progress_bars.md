### Added

- `describe_ebm(ebm, progress_bar=True)` shows a progress bar on the error stream, with one step for each feature and one for the summary. If the stream is not a terminal, the bar writes one line for each step.
- `run_task(..., progress=True)` shows a progress bar of the questions of a benchmark task. `benchmark/run.py` shows a bar by default. Use `--no-progress` to turn it off.
