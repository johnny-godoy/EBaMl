"""Build or serve the documentation with Zensical, after the marimo notebooks and the proof papers are exported.

Run it from the repo root: ``uv run --group docs python -m scripts.build_docs [build|serve]``. CI runs the same
command, so a local build fails exactly when CI fails.

The proof papers of ``proofs/papers/<topic>/paper.typ`` are compiled into ``docs/proofs/<topic>.pdf`` in the same way,
and the proof pages link to them.

The notebook pages embed ``notebooks/<name>.html``. The script exports these files into ``docs/notebooks/`` before
the build. Zensical copies a non-Markdown file of ``docs/`` to the site unchanged, so the public URL stays
``notebooks/<name>.html``. The folder is gitignored, and it survives the rebuilds of ``zensical serve``. The other
choice, an export into ``site/`` after the build, loses the files when a rebuild empties ``site/``. Zensical also
ignores a ``hooks`` setting without a warning, so a hook cannot run the export. A notebook is exported again only
when it, or the package that it runs, is newer than its output. The staleness check keeps repeated local builds fast.

Zensical prints a ``griffe:`` warning for a docstring that does not render, for example a parameter that the function
does not have, and still exits with code 0. The ``build`` command fails when the output has such a warning.
"""

import argparse
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- marimo and Zensical only offer their build as a command
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import structlog
import typst

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS: tuple[Path, ...] = (ROOT / "notebooks" / "titanic.py", ROOT / "notebooks" / "local_explanations.py")
EXPORT_DIR = ROOT / "docs" / "notebooks"
# the notebooks run the package, so a change to any module makes the output stale
PACKAGE: tuple[Path, ...] = tuple(sorted((ROOT / "src" / "ebaml").rglob("*.py")))
PAPERS_ROOT = ROOT / "proofs" / "papers"
PAPER_DIR = ROOT / "docs" / "proofs"
WARNING_PREFIX = "griffe:"


def _is_stale(output: Path, notebook: Path) -> bool:
    """Whether the exported notebook is missing, or older than the notebook or the package it runs.

    ``zensical serve`` rebuilds on every edit of the docs, and running the notebook takes a while.

    Parameters
    ----------
    output: Path
        The exported notebook.
    notebook: Path
        The source of the notebook.

    Returns
    -------
    bool
        True if the notebook has to be exported again.

    """
    sources = (notebook, *PACKAGE)
    return not output.exists() or any(source.stat().st_mtime > output.stat().st_mtime for source in sources)


def export_notebooks() -> None:
    """Export the marimo notebooks, with the output of their cells, into ``docs/notebooks/``.

    The LLM cells of the notebooks only run on a button press, so the export needs no API key.

    Raises
    ------
    subprocess.CalledProcessError
        If marimo fails, for example when a cell raises an error. The incomplete file is deleted first.

    """
    for notebook in NOTEBOOKS:
        output = EXPORT_DIR / f"{notebook.stem}.html"
        if not _is_stale(output, notebook):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "marimo", "export", "html", str(notebook), "-o", str(output), "--force"]
        try:
            subprocess.run(command, cwd=ROOT, check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
        except subprocess.CalledProcessError:
            # marimo writes the file even when a cell fails, and the next run would take it as fresh
            output.unlink(missing_ok=True)
            raise


def export_papers() -> None:
    """Compile each proof paper of ``proofs/papers/`` into ``docs/proofs/<topic>.pdf``.

    The paper of a topic imports the shared style of ``proofs/papers/``, so that folder is the root of the compilation.
    Compiling takes a second, so the function compiles every paper on every call. A paper that does not compile fails
    the build.

    """
    for paper in sorted(PAPERS_ROOT.glob("*/paper.typ")):
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        typst.compile(paper, output=PAPER_DIR / f"{paper.parent.name}.pdf", root=PAPERS_ROOT)


def find_warnings(output: str) -> list[str]:
    """Find the docstring warnings in the output of a Zensical build.

    Parameters
    ----------
    output: str
        The text that the build printed.

    Returns
    -------
    list[str]
        The lines that start with ``griffe:``, in order.

    """
    return [line for line in output.splitlines() if line.startswith(WARNING_PREFIX)]


def build() -> int:
    """Export the notebooks and the papers, build the site in strict mode, and fail on a docstring warning.

    Returns
    -------
    int
        The exit code: the code of Zensical if it fails, 1 if the output has a warning, and 0 otherwise.

    """
    export_notebooks()
    export_papers()
    command = [sys.executable, "-m", "zensical", "build", "--strict"]
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    # one text for both streams, so the gate sees the warnings wherever Zensical writes them
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode:
        return result.returncode
    warnings = find_warnings(output)
    for warning in warnings:
        logger.error("docstring warning", warning=warning)
    return int(bool(warnings))


def serve() -> int:
    """Export the notebooks and the papers, and start the preview server of Zensical.

    Returns
    -------
    int
        The exit code of the server.

    """
    export_notebooks()
    export_papers()
    command = [sys.executable, "-m", "zensical", "serve"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input


COMMANDS: dict[str, Callable[[], int]] = {"build": build, "serve": serve}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command of the script.

    Parameters
    ----------
    argv: Sequence[str] | None
        The arguments of the command line, without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit code of the command.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("command", nargs="?", default="build", choices=sorted(COMMANDS))
    return COMMANDS[parser.parse_args(argv).command]()


if __name__ == "__main__":
    sys.exit(main())
