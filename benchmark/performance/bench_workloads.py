"""Time each workload of the performance benchmark with pytest-benchmark.

pytest-benchmark runs one warm-up round and then the rounds of the workload, and reports the statistics of the
rounds. ``benchmark/run.py performance`` runs this file and writes the summary of its JSON report.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from benchmark.performance.workload_types import Context, Workload
from benchmark.performance.workloads import WORKLOADS


@pytest.mark.parametrize("workload", WORKLOADS, ids=[workload.name for workload in WORKLOADS])
def bench_workload(workload: Workload, benchmark: BenchmarkFixture, context: Context) -> None:
    """Time the rounds of one workload."""
    measured = workload.prepare(context)
    benchmark.group = workload.group
    benchmark.pedantic(measured.run, setup=measured.setup, rounds=workload.rounds, warmup_rounds=1)
