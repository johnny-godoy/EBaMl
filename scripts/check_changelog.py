"""Fail when ``CHANGELOG.md`` and the version in ``pyproject.toml`` do not agree.

Each pull request with a user-visible change is a release: it raises ``version`` in ``pyproject.toml`` and writes its
entries under a new heading ``## [X.Y.Z] - YYYY-MM-DD``. This check fails when a change waits under an
``[Unreleased]`` heading, when the newest heading is not the version in ``pyproject.toml``, or when the headings are not
in strictly decreasing version order. Run it from the repo root: ``uv run python -m scripts.check_changelog``.
"""

import re
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING: re.Pattern[str] = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\S+))?\s*$", re.MULTILINE)
VERSION: re.Pattern[str] = re.compile(r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)")


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Read a ``MAJOR.MINOR.PATCH`` version.

    Parameters
    ----------
    text: str
        The version as written.

    Returns
    -------
    tuple[int, int, int] | None
        The three numbers, or ``None`` if the text is not a version of that form.

    """
    match = VERSION.fullmatch(text)
    if match is None:
        return None
    return int(match["major"]), int(match["minor"]), int(match["patch"])


def _is_iso_date(text: str | None) -> bool:
    """Tell whether a heading date is a valid ``YYYY-MM-DD`` date.

    Parameters
    ----------
    text: str | None
        The date of the heading, or ``None`` if the heading has no date.

    Returns
    -------
    bool
        True for a valid date in the ``YYYY-MM-DD`` form.

    """
    if text is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def changelog_problems(changelog: str, version: str) -> list[str]:
    """List the ways in which the changelog breaks the release rule.

    Parameters
    ----------
    changelog: str
        The content of ``CHANGELOG.md``.
    version: str
        The ``version`` in ``pyproject.toml``.

    Returns
    -------
    list[str]
        One message for each problem. The list is empty when the changelog follows the rule.

    """
    problems: list[str] = []
    released: list[tuple[int, int, int]] = []
    for heading in HEADING.finditer(changelog):
        name = heading["name"]
        if name.lower() == "unreleased":
            problems.append(
                "CHANGELOG.md has an [Unreleased] section: raise the version in pyproject.toml and move the entries "
                "under a heading '## [X.Y.Z] - YYYY-MM-DD'",
            )
            continue
        parsed = _parse_version(name)
        if parsed is None:
            problems.append(f"heading [{name}] is not a MAJOR.MINOR.PATCH version")
            continue
        if not _is_iso_date(heading["date"]):
            problems.append(f"heading [{name}] needs a release date in the form ' - YYYY-MM-DD'")
        if released and parsed >= released[-1]:
            problems.append(f"heading [{name}] must be lower than the heading above it")
        released.append(parsed)
    if not released:
        problems.append("CHANGELOG.md has no release heading")
    elif _parse_version(version) != released[0]:
        newest = ".".join(map(str, released[0]))
        problems.append(f"pyproject.toml has version {version}, but the newest CHANGELOG.md heading is [{newest}]")
    return problems


def main() -> int:
    """Report every problem of the changelog.

    Returns
    -------
    int
        The process exit code: 0 if the changelog follows the release rule, 1 otherwise.

    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    problems = changelog_problems(changelog, pyproject["project"]["version"])
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
