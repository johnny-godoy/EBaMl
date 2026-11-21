"""What the README promises as the page of the package on PyPI, and as the source of the parts that the site includes.

PyPI shows ``README.md`` without the repository, so a relative link or image there leads nowhere. The site includes the
parts of the README that it shows too, with ``--8<-- "README.md:<part>"``, so each such part must exist in the README.
"""

import re
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[2]
README: str = (ROOT / "README.md").read_text(encoding="utf-8")
# The README is the first page that a new reader sees: a short introduction, a quick start, the table of supported
# models and the links to the site. The guides live on the site. A README grows one section at a time, and nobody
# notices, so this limit makes the growth a decision. The limit leaves room for about 50 more lines.
MAX_README_LINES = 150
# the target of a Markdown link or image, and the address of an HTML link or image
MARKDOWN_TARGET: re.Pattern[str] = re.compile(r"\]\(([^)\s]+)")
HTML_TARGET: re.Pattern[str] = re.compile(r"\b(?:href|src)=\"([^\"]+)\"")
README_PART: re.Pattern[str] = re.compile(r'--8<-- "README\.md:([\w-]+)"')


def link_targets(text: str) -> list[str]:
    """Find the target of each link and image of a Markdown text.

    Parameters
    ----------
    text
        The Markdown text.

    Returns
    -------
    list[str]
        The targets, in the order of the text: Markdown links and images first, then HTML ones.

    """
    return [*MARKDOWN_TARGET.findall(text), *HTML_TARGET.findall(text)]


def test_the_guard_finds_the_targets_of_markdown_and_html_links() -> None:
    text = '[Setup](docs/setup.md) and ![graph](https://example.org/g.svg) and <img src="a.png">'
    assert link_targets(text) == ["docs/setup.md", "https://example.org/g.svg", "a.png"]


def test_every_link_and_image_of_the_readme_is_an_absolute_https_address() -> None:
    relative = [target for target in link_targets(README) if not target.startswith("https://")]
    assert relative == []


def test_the_readme_is_at_most_the_length_of_a_short_introduction() -> None:
    assert len(README.splitlines()) <= MAX_README_LINES


@pytest.mark.parametrize("page", sorted((ROOT / "docs").glob("*.md")), ids=lambda page: page.name)
def test_each_part_of_the_readme_that_a_page_includes_exists(page: Path) -> None:
    for part in README_PART.findall(page.read_text(encoding="utf-8")):
        assert f"<!-- --8<-- [start:{part}] -->" in README
        assert f"<!-- --8<-- [end:{part}] -->" in README
