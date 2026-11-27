"""Tests of the summary of a run of the performance benchmark."""

from pathlib import Path

import pytest
from benchmark.performance.summary import SCHEMA_VERSION, read_summary, summarize, write_summary


def _report(*names: str) -> dict[str, object]:
    return {
        "benchmarks": [
            {
                "name": f"bench_workload[{name}]",
                "group": name.split(".")[0],
                "param": name,
                "stats": {"rounds": 5, "median": 0.5, "iqr": 0.1, "min": 0.4, "mean": 0.55},
            }
            for name in names
        ],
    }


def test_the_summary_keeps_the_name_group_and_times_of_each_workload_in_order() -> None:
    summary = summarize(_report("graphs.render.titanic", "llm.sync.describe_ebm.titanic"), "abc123")
    assert [timing.name for timing in summary.workloads] == ["graphs.render.titanic", "llm.sync.describe_ebm.titanic"]
    first = summary.workloads[0]
    assert (first.group, first.rounds, first.median_seconds, first.iqr_seconds) == ("graphs", 5, 0.5, 0.1)
    assert summary.commit == "abc123"
    assert summary.schema_version == SCHEMA_VERSION


def test_a_summary_reads_back_as_it_was_written(tmp_path: Path) -> None:
    summary = summarize(_report("local.facts.titanic"), None)
    path = tmp_path / "summary.json"
    write_summary(summary, path)
    assert read_summary(path) == summary


def test_a_summary_of_another_version_fails_to_read(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_bytes(b'{"schema_version": 0, "workloads": []}')
    with pytest.raises(ValueError, match="version 0"):
        read_summary(path)


def test_the_timing_of_a_workload_that_the_run_did_not_measure_is_none() -> None:
    summary = summarize(_report("local.facts.titanic"), None)
    assert summary.timing("local.facts.titanic") is not None
    assert summary.timing("mcp.session") is None


def test_a_report_without_a_list_of_benchmarks_fails() -> None:
    with pytest.raises(TypeError, match="no list"):
        summarize({"benchmarks": None}, None)
