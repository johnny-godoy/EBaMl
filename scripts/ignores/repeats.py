"""Fail when one ignore rule appears at more sites than the limit, so that a repeated cause gets one shared fix.

A copied ignore hides its cause, and the copies grow with each feature. ``scripts/ignores/sites.py`` finds every
workaround. This script groups them by tool and rule, for example ``ruff: ignore[banned-api]``, and counts the sites of
each group. A group with more than ``LIMIT`` sites fails. The sites in a file of ``SHARED_HOMES`` do not count, because
a shared fix keeps its one ignore there. A tool in ``UNLIMITED_TOOLS`` has no limit, with the reason next to it.

Run it as a module from the repo root: ``uv run python -m scripts.ignores.repeats``.
``.claude/rules/prek-and-ignores.md`` lists the shared fix for each repeated cause.
"""

import itertools
import re
import sys
from collections.abc import Iterable, Mapping

import attrs

from scripts.ignores.sites import Site, find_sites

# The most sites that one rule may have outside the shared homes. A rule with more sites has a cause to fix once.
LIMIT = 5

# The files that hold the shared fix of a repeated cause. Their ignores are the one copy that the fix needs.
SHARED_HOMES = frozenset(
    {
        "scripts/fixed_command.py",
        "src/ebaml/core/ebm_classes.py",
        "tests/docs/conftest.py",
        "tests/unit/ebaml/conftest.py",
        "tests/unit/ebaml/mcp/conftest.py",
    },
)

# A tool whose ignores have no shared fix, with the reason. Each one belongs to the line that carries it.
UNLIMITED_TOOLS: Mapping[str, str] = {
    "cosmic-ray": "each pragma marks one equivalent mutant of its own line, so no shared fix exists",
}

WHITESPACE: re.Pattern[str] = re.compile(r"\s+")


@attrs.frozen
class Repeat:
    """A rule that has more sites than the limit."""

    program: str
    rule: str
    sites: tuple[Site, ...]

    @property
    def message(self) -> str:
        """Describe the repeat and the way to fix it.

        Returns
        -------
        str
            One line for the user: the rule, the count, the places and the fix.

        """
        places = ", ".join(site.location for site in self.sites)
        return (
            f"{len(self.sites)} sites ignore `{self.rule}` ({self.program}), and the limit is {LIMIT}: {places}. "
            "Fix the cause once (a shared helper, import or fixture, or one config entry with a `reason:` block). "
            "See .claude/rules/prek-and-ignores.md."
        )


def _rule(site: Site) -> str:
    return WHITESPACE.sub(" ", site.what).strip()


def find_repeats(sites: Iterable[Site], limit: int = LIMIT) -> list[Repeat]:
    """Group the sites by tool and rule, and keep each group with more sites than the limit.

    Parameters
    ----------
    sites: Iterable[Site]
        Every workaround of the repo.
    limit: int
        The most sites that one rule may have.

    Returns
    -------
    list[Repeat]
        The groups over the limit, the largest first. A site in a shared home, or of an unlimited tool, is not counted.

    """
    counted = [site for site in sites if site.file not in SHARED_HOMES and site.program not in UNLIMITED_TOOLS]
    ordered = sorted(counted, key=lambda site: (site.program, _rule(site), site.file, site.line))
    repeats = [
        Repeat(program=program, rule=rule, sites=tuple(group))
        for (program, rule), group in itertools.groupby(ordered, key=lambda site: (site.program, _rule(site)))
    ]
    return sorted((repeat for repeat in repeats if len(repeat.sites) > limit), key=lambda repeat: -len(repeat.sites))


def main() -> int:
    """Report every rule that has more sites than the limit.

    Returns
    -------
    int
        The process exit code: 0 if no rule is over the limit, 1 otherwise.

    """
    repeats = find_repeats(find_sites())
    for repeat in repeats:
        sys.stderr.write(f"{repeat.message}\n")
    return 1 if repeats else 0


if __name__ == "__main__":
    sys.exit(main())
