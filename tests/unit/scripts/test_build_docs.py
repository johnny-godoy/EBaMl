"""Tests for the script that exports the notebooks and builds the docs."""

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from scripts import build_docs


def touch(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ...]:
    notebooks = (touch(tmp_path / "first.py", 100), touch(tmp_path / "second.py", 100))
    package = (touch(tmp_path / "module.py", 100),)
    monkeypatch.setattr(build_docs, "NOTEBOOKS", notebooks)
    monkeypatch.setattr(build_docs, "PACKAGE", package)
    monkeypatch.setattr(build_docs, "EXPORT_DIR", tmp_path / "docs" / "notebooks")
    monkeypatch.setattr(build_docs, "PAPERS_ROOT", tmp_path / "papers")
    monkeypatch.setattr(build_docs, "PAPER_DIR", tmp_path / "docs" / "proofs")
    monkeypatch.setattr(build_docs, "SITE_DIR", tmp_path / "site")
    return (*notebooks, *package)


@pytest.fixture
def compiled(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path, Path]]:
    """Replace ``typst.compile``, so that no test compiles a real paper.

    Returns
    -------
    list[tuple[Path, Path, Path]]
        The list that collects the paper, the output and the root of each call.

    """
    calls: list[tuple[Path, Path, Path]] = []

    def compile_paper(paper: Path, *, output: Path, root: Path) -> None:
        calls.append((paper, output, root))

    monkeypatch.setattr(build_docs.typst, "compile", compile_paper)
    return calls


def record_runs(monkeypatch: pytest.MonkeyPatch, result: Callable[[list[str]], int] | None = None) -> list[list[str]]:
    """Replace ``subprocess.run``.

    Returns
    -------
    list[list[str]]
        The list that collects the commands.

    """
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> build_docs.subprocess.CompletedProcess[str]:
        commands.append(command)
        code = result(command) if result else 0
        return build_docs.subprocess.CompletedProcess(command, code, stdout="", stderr="")

    monkeypatch.setattr(build_docs.subprocess, "run", run)
    return commands


def test_package_covers_the_subpackages() -> None:
    assert any(path.parent.name == "graphs" for path in build_docs.PACKAGE)


def test_missing_output_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    assert build_docs._is_stale(tmp_path / "out.html", sources[0])


def test_output_older_than_a_source_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    output = touch(tmp_path / "out.html", 200)
    assert not build_docs._is_stale(output, sources[0])
    os.utime(sources[2], (300, 300))
    assert build_docs._is_stale(output, sources[0])


def test_output_older_than_its_notebook_is_stale(tmp_path: Path, sources: tuple[Path, ...]) -> None:
    output = touch(tmp_path / "out.html", 200)
    os.utime(sources[0], (300, 300))
    assert build_docs._is_stale(output, sources[0])
    assert not build_docs._is_stale(output, sources[1])


def test_export_writes_every_notebook_into_the_docs_folder(
    tmp_path: Path,
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = record_runs(monkeypatch)
    build_docs.export_notebooks()

    assert (tmp_path / "docs" / "notebooks").is_dir()
    assert commands == [
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(notebook),
            "-o",
            str(tmp_path / "docs" / "notebooks" / f"{notebook.stem}.html"),
            "--force",
        ]
        for notebook in sources[:2]
    ]


@pytest.mark.usefixtures("sources")
def test_export_keeps_an_output_that_is_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    touch(tmp_path / "docs" / "notebooks" / "first.html", 200)
    commands = record_runs(monkeypatch)
    build_docs.export_notebooks()
    assert [command[5] for command in commands] == [str(build_docs.NOTEBOOKS[1])]


def test_export_passes_the_arguments_that_make_a_failure_stop_the_build(
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs: list[dict[str, object]] = []

    def run(_command: list[str], **options: object) -> None:
        kwargs.append(options)

    monkeypatch.setattr(build_docs.subprocess, "run", run)
    build_docs.export_notebooks()
    assert kwargs == [{"cwd": build_docs.ROOT, "check": True}] * len(sources[:2])


def test_find_warnings_returns_the_griffe_lines() -> None:
    output = "Build started\ngriffe: a.py:1: Parameter 'x' is bad\nNo issues found\ngriffe: b.py:2: other\n"
    assert build_docs.find_warnings(output) == ["griffe: a.py:1: Parameter 'x' is bad", "griffe: b.py:2: other"]


def test_find_warnings_ignores_a_line_that_only_mentions_griffe() -> None:
    assert build_docs.find_warnings("see griffe: docs\n") == []


def test_build_fails_when_zensical_prints_a_griffe_warning_and_exits_zero(
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Zensical 0.0.71 exits with code 0 on a docstring warning; this is the bug that the gate closes.
    del sources

    def run(command: list[str], **_kwargs: object) -> build_docs.subprocess.CompletedProcess[str]:
        text = "griffe: src/x.py:1: Parameter 'bogus' does not appear in the function signature\n"
        return build_docs.subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr=text if "zensical" in command else "",
        )

    monkeypatch.setattr(build_docs.subprocess, "run", run)
    assert build_docs.build() == 1
    assert "Parameter 'bogus'" in capsys.readouterr().out


def test_build_passes_when_the_output_is_clean(sources: tuple[Path, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    del sources
    commands = record_runs(monkeypatch)
    assert build_docs.build() == 0
    assert commands[-1] == [sys.executable, "-m", "zensical", "build", "--strict"]


def test_build_returns_the_exit_code_of_zensical(sources: tuple[Path, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    del sources
    record_runs(monkeypatch, lambda command: 3 if "zensical" in command else 0)
    assert build_docs.build() == 3


def test_serve_exports_then_starts_the_server(sources: tuple[Path, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    del sources
    commands = record_runs(monkeypatch)
    assert build_docs.serve() == 0
    assert commands[-1] == [sys.executable, "-m", "zensical", "serve"]
    assert len(commands) == 3


def test_main_runs_build_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(build_docs.COMMANDS, "build", lambda: 7)
    assert build_docs.main([]) == 7


def test_main_runs_the_named_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(build_docs.COMMANDS, "serve", lambda: 5)
    assert build_docs.main(["serve"]) == 5


def test_failed_export_leaves_no_file_that_looks_fresh(
    tmp_path: Path,
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # marimo writes the HTML even when a cell fails, and a stale check would then skip the notebook on the next run.
    del sources

    def run(command: list[str], **_kwargs: object) -> None:
        Path(command[command.index("-o") + 1]).write_text("half", encoding="utf-8")
        raise build_docs.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(build_docs.subprocess, "run", run)
    with pytest.raises(build_docs.subprocess.CalledProcessError):
        build_docs.export_notebooks()
    assert not (tmp_path / "docs" / "notebooks" / "first.html").exists()


@pytest.mark.usefixtures("sources")
def test_export_papers_compiles_every_topic_into_the_docs_folder(
    tmp_path: Path,
    compiled: list[tuple[Path, Path, Path]],
) -> None:
    second = touch(tmp_path / "papers" / "second-topic" / "paper.typ", 100)
    first = touch(tmp_path / "papers" / "first-topic" / "paper.typ", 100)
    touch(tmp_path / "papers" / "style.typ", 100)
    build_docs.export_papers()

    pdfs = tmp_path / "docs" / "proofs"
    assert pdfs.is_dir()
    assert compiled == [
        (first, pdfs / "first-topic.pdf", tmp_path / "papers"),
        (second, pdfs / "second-topic.pdf", tmp_path / "papers"),
    ]


@pytest.mark.usefixtures("sources")
def test_export_papers_without_a_paper_creates_nothing(tmp_path: Path, compiled: list[tuple[Path, Path, Path]]) -> None:
    build_docs.export_papers()
    assert compiled == []
    assert not (tmp_path / "docs" / "proofs").exists()


@pytest.mark.parametrize("command", ["build", "serve"])
def test_build_and_serve_export_the_papers_before_the_site(
    command: str,
    sources: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del sources
    events = record_runs(monkeypatch)
    monkeypatch.setattr(build_docs, "export_papers", lambda: events.append(["papers"]))
    build_docs.COMMANDS[command]()
    assert [event[0] if event == ["papers"] else event[2] for event in events[-2:]] == ["papers", "zensical"]


def page(site: Path, path: str, body: str) -> None:
    target = site / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    page(site, "index.html", "home")
    page(site, "setup/index.html", "setup")
    page(site, "notebooks/titanic.html", "notebook")
    return site


def test_base_path_is_the_path_of_the_site_url() -> None:
    assert build_docs.base_path() == "/EBaMl/"


def test_raw_html_link_with_one_parent_too_many_is_broken(site: Path) -> None:
    # Zensical rewrote ``../notebooks/titanic.html`` of docs/notebook.md into this link, which works at ``/`` only.
    page(site, "notebook/index.html", '<iframe src="../../notebooks/titanic.html"></iframe>')
    assert build_docs.find_broken_links(site, "/EBaMl/") == ["notebook/index.html: ../../notebooks/titanic.html"]


def test_links_that_the_server_can_serve_are_not_broken(site: Path) -> None:
    links = (
        '<a href="../notebooks/titanic.html">notebook</a>'
        '<a href="../setup/">setup</a><a href="./..">home</a><a href="/EBaMl/setup/#install">anchor</a>'
        '<a href="#top">top</a><a href="https://example.org/x">external</a><a href="//cdn.example.org/x.js">cdn</a>'
        '<a href="mailto:a@example.org">mail</a><a>no link</a>'
    )
    page(site, "notebook/index.html", links)
    assert build_docs.find_broken_links(site, "/EBaMl/") == []


def test_link_to_a_file_that_the_site_does_not_have_is_broken(site: Path) -> None:
    page(site, "notebook/index.html", '<a href="../proofs/missing.pdf">paper</a>')
    assert build_docs.find_broken_links(site, "/EBaMl/") == ["notebook/index.html: ../proofs/missing.pdf"]


def test_find_broken_links_reads_only_the_pages_of_zensical(site: Path) -> None:
    # an exported notebook is not a page of Zensical, and its links are not the business of the docs build
    page(site, "notebooks/titanic.html", '<a href="missing.html">x</a>')
    assert build_docs.find_broken_links(site, "/EBaMl/") == []


def test_build_fails_when_the_site_has_a_broken_link(
    sources: tuple[Path, ...],
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del sources
    page(site, "notebook/index.html", '<iframe src="../../notebooks/titanic.html"></iframe>')
    record_runs(monkeypatch)
    assert build_docs.build() == 1
