"""Tests of the table of workloads of the performance benchmark."""

from benchmark.performance.workloads import WORKLOADS

import ebaml


def test_each_workload_has_a_name_of_its_own() -> None:
    names = [workload.name for workload in WORKLOADS]
    assert len(names) == len(set(names))


def test_each_name_that_a_workload_requires_is_a_public_name_of_ebaml() -> None:
    required = {workload.requires for workload in WORKLOADS if workload.requires is not None}
    assert required <= set(ebaml.__all__)
    assert "RenderTiming" in required
