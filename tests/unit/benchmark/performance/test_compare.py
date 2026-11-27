"""Tests of the comparison of two runs of the performance benchmark."""

import pytest
from benchmark.performance.compare import compare, to_markdown
from benchmark.performance.summary import RunSummary, WorkloadTiming


def _run(**medians: float) -> RunSummary:
    timings = [WorkloadTiming(name, "group", 5, median, 0.0, median, median) for name, median in medians.items()]
    return RunSummary(1, "2026-11-27T00:00:00+00:00", None, "3.13", "x86_64 Linux", timings)


def test_a_workload_is_slower_only_above_the_threshold_and_the_floor() -> None:
    base = _run(fast=0.001, steady=1.0, slower=1.0)
    head = _run(fast=0.002, steady=1.2, slower=1.3)
    comparison = compare(base, head, threshold=0.25, floor_seconds=0.002)
    assert [change.name for change in comparison.slower] == ["slower"]


def test_a_faster_head_is_never_slower() -> None:
    comparison = compare(_run(call=2.0), _run(call=1.0))
    (change,) = comparison.changes
    assert not change.is_slower
    assert change.ratio == pytest.approx(0.5)


def test_a_workload_of_one_run_only_is_listed_and_not_compared() -> None:
    comparison = compare(_run(old=1.0, both=1.0), _run(both=1.0, new=1.0))
    assert [change.name for change in comparison.changes] == ["both"]
    assert comparison.only_base == ["old"]
    assert comparison.only_head == ["new"]


def test_the_markdown_names_each_slower_workload_and_lists_every_change() -> None:
    comparison = compare(_run(call=0.010, load=2.0, new=1.0), _run(call=0.020, load=2.0))
    text = to_markdown(comparison, threshold=0.25)
    assert "1 workloads are more than 25% slower: `call`." in text
    assert "| `call` | 10.0 ms | 20.0 ms | +100% (slower) |" in text
    assert "| `load` | 2.00 s | 2.00 s | +0% |" in text
    assert "Only the base measures: `new`." in text


def test_the_markdown_of_no_slower_workload_says_so() -> None:
    text = to_markdown(compare(_run(call=1.0), _run(call=1.0)))
    assert "No workload is more than 25% slower." in text
