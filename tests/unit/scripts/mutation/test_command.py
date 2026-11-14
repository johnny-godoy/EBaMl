"""Tests for the two-stage test run of one mutant."""

import sys
from pathlib import Path

import orjson
import pytest
from scripts.mutation import command as command_script
from scripts.mutation import covering_tests

# a stand-in for pytest: it records its arguments, fails when a node id contains "fail", and fails the full run
# (the one with the test directory) when FULL_FAILS is set
FAKE_PYTEST = (
    "import os, sys; open('calls.txt', 'a').write(' '.join(sys.argv[1:]) + '\\n'); "
    "sys.exit(1 if any('fail' in word for word in sys.argv) or "
    "(os.environ.get('FULL_FAILS') and 'tests/unit' in sys.argv) else 0)"
)


def fake_command() -> list[str]:
    return [sys.executable, "-c", FAKE_PYTEST, "-x", command_script.TEST_PATH]


def calls(directory: Path) -> list[str]:
    return (directory / "calls.txt").read_text().splitlines()


def test_stage_one_replaces_the_test_directory_with_the_covering_node_ids() -> None:
    command = ["python", "-m", "pytest", "-x", "tests/unit"]
    assert command_script.stage_one_command(command, {"b.py::t2", "a.py::t1"}) == [
        "python",
        "-m",
        "pytest",
        "-x",
        "a.py::t1",
        "b.py::t2",
    ]


def test_a_failing_covering_test_kills_the_mutant_without_the_full_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert command_script.run_tests(fake_command(), {"a.py::test_fail"}) == 1
    assert calls(tmp_path) == ["-x a.py::test_fail"]


def test_a_mutant_survives_only_when_the_covering_tests_and_the_full_run_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert command_script.run_tests(fake_command(), {"a.py::test_ok"}) == 0
    assert calls(tmp_path) == ["-x a.py::test_ok", "-x tests/unit"]


def test_the_full_run_decides_when_the_covering_tests_pass_and_another_test_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FULL_FAILS", "1")  # a test outside the covering tests fails
    assert command_script.run_tests(fake_command(), {"a.py::test_ok"}) == 1
    assert len(calls(tmp_path)) == 2


def test_without_covering_tests_only_the_full_command_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert command_script.run_tests(fake_command(), set()) == 0
    assert calls(tmp_path) == ["-x tests/unit"]


def test_covering_from_environment_reads_the_tests_of_the_changed_statements(tmp_path: Path) -> None:
    original = tmp_path / "original"
    shard = tmp_path / "shard"
    for root, text in ((original, "x = 1\ny = 2\n"), (shard, "x = 1\ny = 3\n")):
        (root / "src").mkdir(parents=True)
        (root / "src" / "m.py").write_text(text)
    coverage = tmp_path / "coverage.json"
    coverage.write_bytes(orjson.dumps({"src/m.py": {"1": ["t.py::a"], "2": ["t.py::b"]}}))
    environ = {"MUTATION_MODULE": "src/m.py", "MUTATION_ORIGINAL": str(original), "MUTATION_COVERAGE": str(coverage)}
    assert covering_tests.covering_from_environment(shard, environ) == {"t.py::b"}
    assert covering_tests.covering_from_environment(shard, {}) == set()
