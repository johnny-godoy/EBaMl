"""Guard: the docs, the README and the notebooks import public names from ``ebaml``, not from a submodule."""

import re
from pathlib import Path

import pytest

import ebaml

ROOT: Path = Path(__file__).resolve().parents[3]
# ``docs/plans/`` holds handoff prompts for the agent, so it may name modules.
FILES: list[Path] = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "notebooks").glob("*.py")),
]
DEEP_IMPORT: re.Pattern[str] = re.compile(r"^\s*from (ebaml\.[\w.]+) import (\([^)]*\)|[^\n(]+)", re.MULTILINE)


def deep_imports_of_public_names(text: str) -> list[str]:
    """Find the imports from an ``ebaml`` submodule of a name that ``ebaml`` exports.

    Parameters
    ----------
    text
        The text of a Markdown file or a notebook.

    Returns
    -------
    list[str]
        The full path of each such name.

    """
    problems: list[str] = []
    for match in DEEP_IMPORT.finditer(text):
        names = [name.strip() for name in match.group(2).strip("()").replace("\n", " ").split(",") if name.strip()]
        problems.extend(f"{match.group(1)}.{name}" for name in names if name in ebaml.__all__)
    return problems


def test_guard_finds_a_deep_import_of_a_public_name() -> None:
    text = "from ebaml.llm.descriptors import describe_ebm, JSONValue\nfrom ebaml import EBMGraph\n"
    assert deep_imports_of_public_names(text) == ["ebaml.llm.descriptors.describe_ebm"]


def test_guard_finds_a_name_in_a_parenthesized_import() -> None:
    text = "    from ebaml.local.audience import (\n        Layperson,\n        other_name,\n    )\n"
    assert deep_imports_of_public_names(text) == ["ebaml.local.audience.Layperson"]


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_file_imports_public_names_from_ebaml(path: Path) -> None:
    assert deep_imports_of_public_names(path.read_text(encoding="utf-8")) == []
