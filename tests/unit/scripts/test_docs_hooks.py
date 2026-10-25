"""Tests for the MkDocs hook that exports the notebook."""

import os
import sys
from pathlib import Path

import pytest
from scripts import docs_hooks


def touch(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ...]:
    notebooks = (touch(tmp_path / "first.py", 100), touch(tmp_path / "second.py", 100))
    package = (touch(tmp_path / "module.py", 100),)
    monkeypatch.setattr(docs_hooks, "NOTEBOOKS", notebooks)
    monkeypatch.setattr(docs_hooks, "PACKAGE", package)
    return (*notebooks, *package)


def test_missing_output_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    assert docs_hooks._is_stale(tmp_path / "out.html", sources[0])


def test_output_older_than_a_source_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    output = touch(tmp_path / "out.html", 200)
    assert not docs_hooks._is_stale(output, sources[0])
    os.utime(sources[2], (300, 300))
    assert docs_hooks._is_stale(output, sources[0])


def test_output_older_than_its_notebook_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    output = touch(tmp_path / "out.html", 200)
    os.utime(sources[0], (300, 300))
    assert docs_hooks._is_stale(output, sources[0])
    assert not docs_hooks._is_stale(output, sources[1])


def test_on_post_build_exports_every_notebook_into_the_site(
    tmp_path: Path,
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(docs_hooks.subprocess, "run", run)
    docs_hooks.on_post_build({"site_dir": str(tmp_path / "site")})

    assert (tmp_path / "site" / "notebooks").is_dir()
    assert [command for command, _ in calls] == [
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(notebook),
            "-o",
            str(tmp_path / "site" / "notebooks" / f"{notebook.stem}.html"),
            "--force",
        ]
        for notebook in sources[:2]
    ]
    assert all(kwargs == {"cwd": docs_hooks.ROOT, "check": True} for _, kwargs in calls)


@pytest.mark.usefixtures("sources")
def test_on_post_build_keeps_an_output_that_is_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    touch(tmp_path / "site" / "notebooks" / "first.html", 200)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(docs_hooks.subprocess, "run", run)
    docs_hooks.on_post_build({"site_dir": str(tmp_path / "site")})
    assert [command[5] for command, _ in calls] == [str(docs_hooks.NOTEBOOKS[1])]
