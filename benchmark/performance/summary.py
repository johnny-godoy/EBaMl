"""The summary of a run of the performance benchmark: one JSON shape for every run, so that two runs compare.

pytest-benchmark writes a long report with the machine, the commit and every statistic. The summary keeps the fields
that a comparison reads, with the times in seconds.
"""

import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import attrs
import orjson

# Raise SCHEMA_VERSION when a field changes, so that a comparison of two runs of different shapes fails clearly.
SCHEMA_VERSION = 1


@attrs.frozen
class WorkloadTiming:
    """The time of the rounds of one workload.

    Attributes
    ----------
    name: str
        The name of the workload.
    group: str
        The group of the workload.
    rounds: int
        The number of measured rounds.
    median_seconds: float
        The median time of a round. A comparison reads this value, because one slow round does not move it.
    iqr_seconds: float
        The interquartile range of the times of a round: the spread of the middle half.
    min_seconds: float
        The time of the fastest round.
    mean_seconds: float
        The mean time of a round.

    """

    name: str
    group: str
    rounds: int
    median_seconds: float
    iqr_seconds: float
    min_seconds: float
    mean_seconds: float


@attrs.frozen
class RunSummary:
    """The summary of one run of the performance benchmark.

    Attributes
    ----------
    schema_version: int
        The version of this shape.
    created: str
        When the run ended, in ISO 8601 and UTC.
    commit: str | None
        The commit of the measured code, or ``None`` outside a git repository.
    python: str
        The version of Python.
    machine: str
        The machine and the operating system.
    workloads: list[WorkloadTiming]
        The timing of each workload of the run, in the order of the run.

    """

    schema_version: int
    created: str
    commit: str | None
    python: str
    machine: str
    workloads: list[WorkloadTiming]

    def timing(self, name: str) -> WorkloadTiming | None:
        """Find the timing of a workload.

        Parameters
        ----------
        name: str
            The name of the workload.

        Returns
        -------
        WorkloadTiming | None
            The timing, or ``None`` if the run did not measure the workload.

        """
        return next((timing for timing in self.workloads if timing.name == name), None)


def summarize(report: Mapping[str, object], commit: str | None) -> RunSummary:
    """Summarize the JSON report of pytest-benchmark.

    Parameters
    ----------
    report: Mapping[str, object]
        The parsed report of ``--benchmark-json``.
    commit: str | None
        The commit of the measured code.

    Returns
    -------
    RunSummary
        The summary.

    Raises
    ------
    TypeError
        If the report has no list of benchmarks.

    """
    entries = report["benchmarks"]
    if not isinstance(entries, Sequence):
        msg = "The report of pytest-benchmark has no list of benchmarks."
        raise TypeError(msg)
    timings = []
    for entry in entries:
        stats = entry["stats"]
        timings.append(
            WorkloadTiming(
                name=str(entry["param"]),
                group=str(entry["group"]),
                rounds=int(stats["rounds"]),
                median_seconds=float(stats["median"]),
                iqr_seconds=float(stats["iqr"]),
                min_seconds=float(stats["min"]),
                mean_seconds=float(stats["mean"]),
            ),
        )
    return RunSummary(
        schema_version=SCHEMA_VERSION,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        python=platform.python_version(),
        machine=f"{platform.machine()} {platform.system()} {platform.release()}",
        workloads=timings,
    )


def write_summary(summary: RunSummary, path: Path) -> None:
    """Write a summary as indented JSON.

    Parameters
    ----------
    summary: RunSummary
        The summary.
    path: Path
        The file to write.

    """
    path.write_bytes(orjson.dumps(attrs.asdict(summary), option=orjson.OPT_INDENT_2) + b"\n")


def read_summary(path: Path) -> RunSummary:
    """Read a summary that ``write_summary`` wrote.

    Parameters
    ----------
    path: Path
        The file.

    Returns
    -------
    RunSummary
        The summary.

    Raises
    ------
    ValueError
        If the file has another version of the shape.

    """
    data = orjson.loads(path.read_bytes())
    if data.get("schema_version") != SCHEMA_VERSION:
        msg = f"{path} has the summary version {data.get('schema_version')}, and this code reads {SCHEMA_VERSION}."
        raise ValueError(msg)
    workloads = [WorkloadTiming(**timing) for timing in data.pop("workloads")]
    return RunSummary(**data, workloads=workloads)
