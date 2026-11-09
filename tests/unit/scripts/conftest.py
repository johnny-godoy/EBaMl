"""Fixtures that build a real git repository for the tests of the scripts."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest
from scripts.fixed_command import run_fixed

GIT = shutil.which("git") or "git"


class GitRunner(Protocol):
    """Run one git command in a repository."""

    def __call__(self, repo: Path, *args: str) -> None:
        """Run ``git <args>`` in ``repo``, and fail the test if git fails."""


@pytest.fixture
def git() -> GitRunner:
    """Give a function that runs a git command with a fixed identity.

    Returns
    -------
    GitRunner
        The function. It takes the repository and the git arguments.

    """

    def run_git(repo: Path, *args: str) -> None:
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
        run_fixed([GIT, *identity, "-C", str(repo), *args], check=True)

    return run_git


@pytest.fixture
def make_repo(git: GitRunner) -> Callable[[Path], Path]:
    """Give a function that builds a repository on ``main`` with one commit.

    Returns
    -------
    Callable[[Path], Path]
        The function. It takes the directory of the repository and returns it.

    """

    def build(root: Path) -> Path:
        root.mkdir(exist_ok=True)
        git(root, "init", "-q", "-b", "main")
        (root / "README.md").write_text("x", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "base")
        return root

    return build
