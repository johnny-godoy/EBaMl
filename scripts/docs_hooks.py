"""MkDocs hooks: run the marimo notebooks and publish their output next to the docs.

Each notebook page embeds ``notebooks/<name>.html``. It is exported into the built site, so it always shows the output
of the current code, and it is never committed.
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import] -- marimo only exposes notebook export as a command
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS: tuple[Path, ...] = (ROOT / "notebooks" / "titanic.py", ROOT / "notebooks" / "local_explanations.py")
# the notebooks run the package, so a change to either makes the output stale
PACKAGE: tuple[Path, ...] = tuple(sorted((ROOT / "src" / "ebaml").glob("*.py")))


def _is_stale(output: Path, notebook: Path) -> bool:
    """Whether the exported notebook is missing, or older than the notebook or the package it runs.

    ``mkdocs serve`` rebuilds on every edit of the docs, and running the notebook takes a while.

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


def on_post_build(config: dict[str, Any], **_kwargs: object) -> None:
    """Export the marimo notebooks, with the output of their cells, into the built site.

    The LLM cells of the notebooks only run on a button press, so the export needs no API key.

    Parameters
    ----------
    config: dict[str, Any]
        The MkDocs configuration.
    **_kwargs: object
        Unused arguments that MkDocs passes to its hooks.

    """
    for notebook in NOTEBOOKS:
        output = Path(config["site_dir"]) / "notebooks" / f"{notebook.stem}.html"
        if not _is_stale(output, notebook):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "marimo", "export", "html", str(notebook), "-o", str(output), "--force"]
        subprocess.run(command, cwd=ROOT, check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
