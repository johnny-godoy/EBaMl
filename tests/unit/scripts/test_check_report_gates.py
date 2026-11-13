"""Tests for the check that the CI report accounts for every job that runs on a pull request."""

from pathlib import Path
from typing import Any

import pytest
from scripts import check_report_gates

CI: dict[Any, Any] = {
    True: {"push": None, "pull_request": None},
    "jobs": {"prek": {}, "changes": {}, "report": {}},
}
PROOFS: dict[Any, Any] = {True: {"pull_request": {"paths": ["proofs/**"]}}, "jobs": {"lake": {"name": "lake build"}}}
RELEASE: dict[Any, Any] = {True: {"push": None}, "jobs": {"build": {}}}
GATES = [("prek", "prek", "always"), ("proofs", "lake build", "some")]
WORKFLOWS = {"ci.yml": CI, "proofs.yml": PROOFS, "release.yml": RELEASE}


NOT_GATES = ["changes", "report"]


def problems(
    gates: list[tuple[str, str, str]],
    required: list[str],
    workflows: dict[str, Any] | None = None,
) -> list[str]:
    return check_report_gates.report_problems(gates, required, workflows or WORKFLOWS, NOT_GATES)


def test_a_report_that_covers_every_job_passes() -> None:
    assert problems(GATES, ["prek"], WORKFLOWS) == []


def test_a_job_with_no_place_in_the_report_fails() -> None:
    extra: dict[Any, Any] = {True: {"pull_request": None}, "jobs": {"lint": {}}}
    [problem] = problems(GATES, ["prek"], {**WORKFLOWS, "lint.yml": extra})
    assert problem == "job 'lint' runs on pull_request but is no gate of ci_report.py and not in NOT_GATES"


def test_a_job_of_a_workflow_for_pushes_only_needs_no_place() -> None:
    assert problems(GATES, [], WORKFLOWS) == []


def test_a_gate_without_a_job_fails() -> None:
    [problem] = problems([*GATES, ("docs", "docs build", "always")], [], WORKFLOWS)
    assert problem == "gate 'docs build' is not the name of a job that runs on pull_request"


def test_a_not_gates_entry_without_a_job_fails() -> None:
    only_prek: dict[Any, Any] = {True: {"pull_request": None}, "jobs": {"prek": {}}}
    found = problems([("prek", "prek", "always")], [], {"ci.yml": only_prek})
    assert "NOT_GATES entry 'changes' is not the name of a job that runs on pull_request" in found


def test_a_job_that_is_a_gate_and_in_not_gates_fails() -> None:
    gates = [*GATES, ("changes", "changes", "always")]
    [problem] = problems(gates, [], WORKFLOWS)
    assert problem == "job 'changes' is a gate and is also in NOT_GATES"


@pytest.mark.parametrize(
    "gates",
    [
        [("prek", "prek", "some"), ("proofs", "lake build", "some")],
        [("prek", "prek", "always"), ("proofs", "lake build", "always")],
    ],
)
def test_a_gate_that_waits_for_a_path_filtered_job_fails(gates: list[tuple[str, str, str]]) -> None:
    [problem] = problems(gates, [])
    assert problem.endswith("must have runs='some' exactly when its workflow filters paths")


def test_a_required_check_that_is_no_gate_fails() -> None:
    [problem] = problems(GATES, ["prek", "docs build"], WORKFLOWS)
    assert problem == "required check 'docs build' is not a gate of ci_report.py"


def test_main_checks_the_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "main.json").write_text('{"rules": []}')
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "x.yml").write_text("on: pull_request\njobs:\n  mystery: {}\n")
    monkeypatch.setattr(check_report_gates, "RULESET", tmp_path / "main.json")
    monkeypatch.setattr(check_report_gates, "WORKFLOWS", tmp_path / "workflows")
    assert check_report_gates.main() == 1
    assert "job 'mystery' runs on pull_request" in capsys.readouterr().err


def test_the_report_of_the_repository_accounts_for_every_job() -> None:
    assert check_report_gates.main() == 0
