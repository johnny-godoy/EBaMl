"""Compile every workaround in the project into one Markdown report to audit.

Each workaround keeps its reason next to the code (see ``scripts/ignore_sites.py``); this script gathers them, grouped
by tool, so they can be reviewed in one place. Run it as a module from the repo root::

    uv run python -m scripts.compile_ignores                      # print to stdout
    uv run python -m scripts.compile_ignores -o ignores-report.md  # write to a file

The report is generated, so do not commit it or edit it by hand.
"""

import itertools
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import cyclopts

from scripts.ignore_sites import Site, find_sites

MISSING = "**MISSING**"


def _cell(text: str) -> str:
    """Escape text for a Markdown table cell.

    Parameters
    ----------
    text: str
        The raw text.

    Returns
    -------
    str
        The text with pipes escaped, safe inside a table row.

    """
    return text.replace("|", "\\|")


def render(sites: Sequence[Site]) -> str:
    """Render workarounds as Markdown: a summary, then one table per tool.

    Parameters
    ----------
    sites: Sequence[Site]
        The workarounds to render.

    Returns
    -------
    str
        The Markdown report.

    """
    by_program = {
        program: list(group)
        for program, group in itertools.groupby(sorted(sites, key=lambda s: s.program), key=lambda s: s.program)
    }
    unjustified = [site for site in sites if site.reason_problem]
    lines = ["# Workaround audit", "", f"{len(sites)} workaround(s), {len(unjustified)} without a proper reason.", ""]
    if by_program:
        lines += ["| Tool | Count |", "| --- | --- |", *(f"| {p} | {len(g)} |" for p, g in by_program.items()), ""]
    for program, group in by_program.items():
        lines += [f"## {program}", "", "| Location | Silences | Reason |", "| --- | --- | --- |"]
        lines += [
            f"| `{site.location}` | `{_cell(site.what)}` | {_cell(site.reason) if site.reason else MISSING} |"
            for site in group
        ]
        lines.append("")
    return "\n".join(lines)


app: cyclopts.App = cyclopts.App(
    name="compile_ignores",
    help=__doc__.splitlines()[0] if __doc__ else None,
    result_action="return_value",
)


@app.default
def compile_report(output: Annotated[Path | None, cyclopts.Parameter(name=["--output", "-o"])] = None) -> int:
    """Write the report to stdout or to ``--output``.

    Parameters
    ----------
    output: Path, optional
        Write the report here instead of stdout.

    Returns
    -------
    int
        The process exit code, always 0: use ``check_ignores`` to enforce.

    """
    report = render(find_sites())
    if output is None:
        sys.stdout.write(report)
    else:
        output.write_text(report, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv: Sequence[str] | None
        The command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        The process exit code.

    """
    return int(app(None if argv is None else list(argv)))


if __name__ == "__main__":
    sys.exit(main())
