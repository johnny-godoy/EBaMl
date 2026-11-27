"""Measure the time of the work of EBaMl around the LLM requests, with a local fake LLM.

Usage: ``uv run python benchmark/run.py performance [--output FILE] [--workload PATTERN ...] [--models-dir DIR]
[--no-progress]``, and ``uv run python benchmark/run.py performance compare BASE HEAD [--markdown FILE] [--gate]``.
Sends no real request.

The command runs the workloads of ``workloads.py`` with pytest-benchmark and writes the summary of the run as JSON.
The first run fits the models of ``models.py`` and saves them, which takes about a minute. ``compare`` reads two
summaries and reports the change of the median time of each workload.
"""

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import attrs
import cyclopts
import orjson
import pytest
from scripts.fixed_command import run_fixed

import ebaml
from benchmark.performance.compare import DEFAULT_FLOOR_SECONDS, DEFAULT_THRESHOLD, compare, to_markdown
from benchmark.performance.models import DEFAULT_MODEL_DIR
from benchmark.performance.summary import read_summary, summarize, write_summary

BENCH_FILE = Path(__file__).resolve().parent / "bench_workloads.py"

app: cyclopts.App = cyclopts.App(name="performance", help=__doc__.splitlines()[0], result_action="return_value")


def _commit() -> str | None:
    """Read the commit of the measured ``ebaml``, which is the base in a comparison of a pull request.

    Returns
    -------
    str | None
        The commit of the repository that holds the imported ``ebaml``, or ``None`` outside a git repository.

    """
    result = run_fixed(["git", "rev-parse", "HEAD"], cwd=Path(ebaml.__file__).resolve().parent)
    return result.stdout.strip() if result.returncode == 0 else None


def _pytest_arguments(report: Path, workloads: Sequence[str], models_dir: Path, *, progress: bool) -> list[str]:
    """Build the arguments of the pytest run of the workloads.

    Parameters
    ----------
    report: Path
        The JSON report of pytest-benchmark.
    workloads: Sequence[str]
        The patterns of the workloads to run.
    models_dir: Path
        The directory of the saved models.
    progress: bool
        Whether pytest prints one line for each workload.

    Returns
    -------
    list[str]
        The arguments. They clear the options of the project, so the run measures no coverage.

    """
    return [
        str(BENCH_FILE),
        "-o",
        "addopts=",
        "-o",
        "python_files=bench_*.py",
        "-o",
        "python_functions=bench_*",
        "-p",
        "no:cacheprovider",
        "--benchmark-json",
        str(report),
        "--models-dir",
        str(models_dir),
        "-v" if progress else "-q",
        *(argument for pattern in workloads for argument in ("--workload", pattern)),
    ]


@app.default
def run(
    *,
    output: Path = Path("performance.json"),
    workload: Annotated[list[str], cyclopts.Parameter(negative_iterable=())] | None = None,
    models_dir: Path = DEFAULT_MODEL_DIR,
    progress: bool = True,
) -> int:
    """Run the workloads and write the summary of the run.

    Parameters
    ----------
    output: Path
        The JSON file of the summary.
    workload: list[str], optional
        Run only the workloads whose name matches this shell pattern, for example ``llm.*``. Repeat the flag for more
        patterns. The default runs every workload.
    models_dir: Path
        The directory of the saved models. A model that is not there is fitted and saved first.
    progress: bool
        Print one line for each workload as it ends. Use ``--no-progress`` to print only the table of results.

    Returns
    -------
    int
        The exit code of pytest: 0 if every workload ran.

    """
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "report.json"
        code = int(pytest.main(_pytest_arguments(report, workload or [], models_dir, progress=progress)))
        if code != 0:
            return code
        summary = summarize(orjson.loads(report.read_bytes()), _commit())
    write_summary(summary, output)
    sys.stdout.write(f"Wrote the timing of {len(summary.workloads)} workloads to {output}.\n")
    return 0


@cyclopts.Parameter(name="*")
@attrs.frozen(kw_only=True)
class CompareOptions:
    """The options of a comparison. Each field is one flag of the command.

    Attributes
    ----------
    threshold: float
        A workload is slower if its median grows by more than this share.
    floor_seconds: float
        A workload is not slower if its median grows by less than this many seconds.
    markdown: Path, optional
        Also write the report to this Markdown file.
    gate: bool
        Exit with code 1 if a workload is slower.

    """

    threshold: float = DEFAULT_THRESHOLD
    floor_seconds: float = DEFAULT_FLOOR_SECONDS
    markdown: Path | None = None
    gate: Annotated[bool, cyclopts.Parameter(negative=())] = False


@app.command(name="compare")
def compare_runs(base: Path, head: Path, /, options: CompareOptions | None = None) -> int:
    """Compare the summary of the base with the summary of the head, and print the change of each workload.

    Parameters
    ----------
    base: Path
        The summary of the base.
    head: Path
        The summary of the head.
    options: CompareOptions, optional
        The flags of the comparison.

    Returns
    -------
    int
        The exit code: 1 if ``--gate`` is on and a workload is slower, else 0.

    """
    options = CompareOptions() if options is None else options
    comparison = compare(
        read_summary(base),
        read_summary(head),
        threshold=options.threshold,
        floor_seconds=options.floor_seconds,
    )
    text = to_markdown(comparison, threshold=options.threshold)
    sys.stdout.write(text)
    if options.markdown is not None:
        options.markdown.write_text(text, encoding="utf-8")
    return 1 if options.gate and comparison.slower else 0
