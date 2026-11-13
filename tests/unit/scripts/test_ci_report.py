"""Tests for the gates of the CI report: their states, their links and the final comment."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(name="ci_report")
def load_ci_report() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "ci_report.py"
    spec = importlib.util.spec_from_file_location("ci_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_run(name: str, status: str, conclusion: str | None = None) -> dict[str, str | None]:
    return {"name": name, "status": status, "conclusion": conclusion, "html_url": f"https://example.test/{name}"}


def test_every_gate_of_a_pull_request_waits_for_its_check_run(ci_report: ModuleType) -> None:
    states = ci_report.gate_states([], "pull_request")
    assert states["docs"] == states["commits"] == states["prek"] == "waiting"
    assert states["proofs"] == states["prompts"] == "absent"


def test_a_push_does_not_wait_for_the_checks_of_pull_requests(ci_report: ModuleType) -> None:
    states = ci_report.gate_states([], "push")
    assert states["docs"] == "waiting"
    assert states["title"] == states["commits"] == states["fragment"] == "absent"


@pytest.mark.parametrize(
    ("status", "conclusion", "state"),
    [
        ("queued", None, "waiting"),
        ("in_progress", None, "running"),
        ("completed", "success", "success"),
        ("completed", "timed_out", "failure"),
        ("completed", "neutral", "skipped"),
        ("completed", None, "cancelled"),
    ],
)
def test_the_state_of_a_gate_follows_its_check_run(
    ci_report: ModuleType,
    status: str,
    conclusion: str,
    state: str,
) -> None:
    states = ci_report.gate_states([check_run("docs build", status, conclusion)], "pull_request")
    assert states["docs"] == state


def test_a_gate_of_another_workflow_has_a_link_to_its_job_log(ci_report: ModuleType) -> None:
    links = ci_report.log_links([check_run("commits", "completed", "failure")])
    assert links == {"commits": "https://example.test/commits"}
    assert "https://example.test/commits" in ci_report.log_pointer(links["commits"])
    assert "Open the job log." in ci_report.log_pointer("")


def test_the_progress_comment_counts_the_gates_that_it_waits_for(
    ci_report: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    states = ci_report.gate_states([check_run("docs build", "completed", "failure")], "pull_request")
    text = ci_report.render_progress(states, {"docs": "the docs failed"})
    assert "CI in progress: 1 of 11 gates finished, 1 failed" in text
    assert "### ❌ docs build" in text
    assert "not started" in text


def test_the_final_report_names_a_failed_gate_of_another_workflow(
    ci_report: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs: list[Any] = [check_run(gate.check, "completed", "success") for gate in ci_report.GATES if gate.runs != "some"]
    runs[ci_report.GATES.index(next(gate for gate in ci_report.GATES if gate.id == "title"))]["conclusion"] = "failure"
    monkeypatch.setattr(ci_report, "check_runs", lambda: runs)
    monkeypatch.setattr(ci_report, "REPORT", tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    ci_report.assemble()
    text = (tmp_path / "comment.md").read_text()
    assert "CI failed: 1 of 11 gates" in text
    assert "[job log](https://example.test/title)" in text


def test_the_final_report_waits_until_the_deadline_for_a_gate_that_never_reports(
    ci_report: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ci_report, "check_runs", list)
    monkeypatch.setattr(ci_report, "SETTLE_SECONDS", 0)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    states, links = ci_report.settle()
    assert states["docs"] == "waiting"
    assert links == {}


LOG = (
    b"2026-01-01T00:00:00.1234567Z \x1b[36;1mstep one\x1b[0m\n"
    b"2026-01-01T00:00:01.0000000Z title: bad title\n"
    b"2026-01-01T00:00:02.0000000Z ##[error]Process completed with exit code 1.\n"
    b"2026-01-01T00:00:03.0000000Z Post job cleanup.\n"
)


def test_a_failed_gate_shows_the_log_around_its_first_error(
    ci_report: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        return LOG

    monkeypatch.setattr(ci_report, "fetch_archive", fetch)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    text = ci_report.failure_text("https://github.com/owner/repo/actions/runs/7/job/42")
    assert requested == ["https://api.github.com/repos/owner/repo/actions/jobs/42/logs"]
    assert "step one\ntitle: bad title\n##[error]Process completed with exit code 1.\nPost job cleanup." in text
    assert "\x1b" not in text
    assert "[job log](https://github.com/owner/repo/actions/runs/7/job/42)" in text


@pytest.mark.parametrize("url", ["", "https://example.test/not-a-job"])
def test_a_failed_gate_without_a_job_log_only_points_to_it(ci_report: ModuleType, url: str) -> None:
    assert ci_report.failure_text(url) == ci_report.log_pointer(url)
