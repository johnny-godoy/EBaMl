"""Tests for the check of the subject style and the size of commits."""

from collections.abc import Callable
from pathlib import Path

import pytest
from scripts import check_commits

from .conftest import GitRunner


@pytest.mark.parametrize(
    "subject",
    [
        "feat(llm): add ModelOptions",
        "fix!: drop the old flag",
        "docs: README links to the roadmap",
        "Merge branch 'main' into feature",
        "fixup! feat: Add a thing.",
        "not a conventional subject",
    ],
)
def test_subjects_without_style_problems(subject: str) -> None:
    assert check_commits.subject_problems(subject) == []


def test_a_capitalized_description_is_a_problem() -> None:
    [problem] = check_commits.subject_problems("feat(llm): Add ModelOptions")
    assert problem.endswith("start the description with a lower-case word")


def test_a_period_at_the_end_is_a_problem() -> None:
    [problem] = check_commits.subject_problems("fix: handle an empty task.")
    assert problem.endswith("remove the period at the end of the description")


def test_changed_lines_skip_generated_and_binary_files() -> None:
    numstat = "10\t5\tsrc/ebaml/a.py\n900\t900\tuv.lock\n3\t0\tchangelog.d/x.md\n-\t-\timage.png\n2\t2\tCHANGELOG.md\n"
    assert check_commits.changed_lines(numstat) == 15


def test_a_small_commit_has_no_size_problem() -> None:
    assert check_commits.size_problem("feat: x", "feat: x", check_commits.MAX_CHANGED_LINES) is None


def test_a_large_commit_without_a_trailer_is_a_problem() -> None:
    problem = check_commits.size_problem("feat: x", "feat: x", check_commits.MAX_CHANGED_LINES + 1)
    assert problem is not None
    assert problem.startswith("'feat: x' changes 401 lines (limit 400")


def test_a_large_commit_with_a_reasoned_trailer_passes() -> None:
    message = "refactor: move a module\n\nLarge-Commit: a file move with no change of content\n"
    assert check_commits.size_problem("refactor: move a module", message, 2000) is None


def test_a_thin_trailer_does_not_excuse_a_large_commit() -> None:
    message = "refactor: move a module\n\nLarge-Commit: big move\n"
    assert check_commits.size_problem("refactor: move a module", message, 2000) is not None


@pytest.fixture
def commit(git: GitRunner) -> Callable[[Path, str, int, str], None]:
    def make_commit(repo: Path, name: str, lines: int, message: str) -> None:
        (repo / name).write_text("x\n" * lines, encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", message)

    return make_commit


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_repo: Callable[[Path], Path], git: GitRunner) -> Path:
    make_repo(tmp_path)
    git(tmp_path, "checkout", "-q", "-b", "feature")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_main_passes_a_branch_of_small_well_written_commits(
    repo: Path,
    commit: Callable[[Path, str, int, str], None],
) -> None:
    commit(repo, "a.py", 10, "feat: add a")
    commit(repo, "uv.lock", 5000, "build: lock the dependencies")
    assert check_commits.main(["--base", "main"]) == 0


def test_main_reports_each_bad_commit_of_a_branch(
    repo: Path,
    commit: Callable[[Path, str, int, str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    commit(repo, "a.py", 10, "feat: Add a.")
    commit(repo, "b.py", 500, "feat: add b")
    commit(repo, "c.py", 500, "fixup! feat: add b")
    assert check_commits.main(["--base", "main"]) == 1
    err = capsys.readouterr().err
    assert "lower-case" in err
    assert "period" in err
    assert "'feat: add b' changes 500 lines" in err
    assert "fixup!" not in err


def test_main_checks_a_message_file(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    message = repo / "COMMIT_EDITMSG"
    message.write_text("feat: Add a thing\n\nBody.\n")
    assert check_commits.main(["--message-file", str(message)]) == 1
    assert "lower-case" in capsys.readouterr().err
    message.write_text("feat: add a thing\n")
    assert check_commits.main(["--message-file", str(message)]) == 0


def test_the_hook_rejects_a_large_staged_change_without_a_trailer(
    repo: Path,
    git: GitRunner,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (repo / "big.py").write_text("x\n" * 500, encoding="utf-8")
    git(repo, "add", "-A")
    message = repo / "COMMIT_EDITMSG"
    message.write_text("feat: add big\n")
    assert check_commits.main(["--message-file", str(message)]) == 1
    assert "'feat: add big' changes 500 lines" in capsys.readouterr().err
    message.write_text("feat: add big\n\nLarge-Commit: a generated fixture file\n")
    assert check_commits.main(["--message-file", str(message)]) == 0
    message.write_text("Merge branch 'main'\n")
    assert check_commits.main(["--message-file", str(message)]) == 0
