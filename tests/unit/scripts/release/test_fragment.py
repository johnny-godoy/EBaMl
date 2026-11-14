"""Tests for the check that a pull request with a change that users can see adds a changelog fragment."""

from collections.abc import Callable
from pathlib import Path

import pytest
from scripts.release import fragment

from tests.unit.scripts.conftest import GitRunner


@pytest.mark.parametrize(
    "path",
    ["src/ebaml/llm/descriptors.py", "baml_src/talk_to_ebm.baml", "benchmark/run.py"],
)
def test_package_prompt_and_benchmark_files_are_user_visible(path: str) -> None:
    assert fragment.is_user_visible(path)


@pytest.mark.parametrize(
    "path",
    [
        "tests/unit/ebaml/test_init.py",
        "baml_src/tests/support.baml",
        "baml_src/tests/model_spec_test.baml",
        "baml_src/tests/mcp_agent_test.baml",
        "baml_src/tests/live/prompts.baml",
        "README.md",
    ],
)
def test_tests_and_docs_are_not_user_visible(path: str) -> None:
    assert not fragment.is_user_visible(path)


@pytest.mark.parametrize("path", ["changelog.d/20261025_x.md", "changelog.d/a.md"])
def test_a_markdown_file_in_the_fragment_directory_is_a_fragment(path: str) -> None:
    assert fragment.is_fragment(path)


@pytest.mark.parametrize("path", ["changelog.d/README.md", "changelog.d/notes.txt", "docs/x.md"])
def test_other_files_are_not_fragments(path: str) -> None:
    assert not fragment.is_fragment(path)


def test_a_user_visible_change_without_a_fragment_is_a_problem() -> None:
    problem = fragment.fragment_problem(["src/ebaml/a.py", "tests/unit/x.py"], [])
    assert problem is not None
    assert "  src/ebaml/a.py\n" in problem
    assert "tests/unit/x.py" not in problem


def test_a_user_visible_change_with_a_fragment_passes() -> None:
    assert fragment.fragment_problem(["src/ebaml/a.py"], ["changelog.d/a.md"]) is None


def test_the_readme_of_the_fragments_is_not_a_fragment() -> None:
    assert fragment.fragment_problem(["src/ebaml/a.py"], ["changelog.d/README.md"]) is not None


def test_an_internal_change_needs_no_fragment() -> None:
    assert fragment.fragment_problem(["tests/unit/x.py", "README.md"], []) is None


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_repo: Callable[[Path], Path], git: GitRunner) -> Path:
    make_repo(tmp_path)
    git(tmp_path, "checkout", "-q", "-b", "feature")
    (tmp_path / "src" / "ebaml").mkdir(parents=True)
    (tmp_path / "src" / "ebaml" / "a.py").write_text("x")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_main_fails_without_a_fragment(repo: Path, git: GitRunner, capsys: pytest.CaptureFixture[str]) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    assert fragment.main(["main"]) == 1
    assert "src/ebaml/a.py" in capsys.readouterr().err


def test_main_passes_with_a_fragment(repo: Path, git: GitRunner) -> None:
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d" / "a.md").write_text("### Added\n\n- A.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    assert fragment.main(["main"]) == 0
