"""Check the structure of each page where axe has no rule: headings, images, frames, the title and the description.

A screen reader lists the headings of a page to move through it, so each page has one ``h1`` and no skipped level. A
reader whose browser does not show a frame needs a link to the file next to it. Search engines show the title and the
description of a page, so each page has its own.
"""

import collections
import itertools
import posixpath
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit

import attrs
import pytest
from playwright.sync_api import Browser

from tests.site.conftest import NAV_PAGES, Site

# Reads the facts of the page in one call: the order of the headings, the images, the frames and the head.
PAGE_FACTS_SCRIPT = """() => {
  // Markdown wraps a frame alone on its lines in a paragraph, so the siblings of that paragraph are next to the frame.
  const wrapper = (frame) => (frame.parentElement.tagName === "P" && frame.parentElement.children.length === 1
    ? frame.parentElement : frame);
  const linkedNearby = (frame) => [wrapper(frame).previousElementSibling, wrapper(frame).nextElementSibling].some(
    (sibling) => sibling !== null && [sibling, ...sibling.querySelectorAll("a[href]")].some(
      (element) => element.href === frame.src));
  const description = document.querySelector('meta[name="description"]');
  return {
    title: document.title,
    description: description === null ? "" : description.content,
    headings: [...document.querySelectorAll("h1, h2, h3, h4, h5, h6")].map((heading) => Number(heading.tagName[1])),
    images: [...document.querySelectorAll("img")].map((image) => ({source: image.src, alt: image.getAttribute("alt")})),
    frames: [...document.querySelectorAll("iframe")].map((frame) => ({
      source: frame.src, title: frame.getAttribute("title"), linked_nearby: linkedNearby(frame)})),
  };
}"""
SITE_NAME_SEPARATOR = " - "


@attrs.frozen
class Image:
    """One image of a page: its address and its alt text, or ``None`` if it has no ``alt`` attribute."""

    source: str
    alt: str | None


@attrs.frozen
class Frame:
    """One ``iframe`` of a page: its address, its title, and whether a sibling element links to the same address."""

    source: str
    title: str | None
    linked_nearby: bool


@attrs.frozen
class PageFacts:
    """The facts of one page that the tests of this module check."""

    title: str
    description: str
    headings: list[int]
    images: list[Image]
    frames: list[Frame]


@pytest.fixture(scope="module")
def page_facts(site: Site, browser: Browser) -> dict[str, PageFacts]:
    """Load each page of the nav once, and read its facts.

    Returns
    -------
    dict[str, PageFacts]
        The facts of each page, by its path.

    """
    context = browser.new_context()
    tab = context.new_page()
    facts: dict[str, PageFacts] = {}
    for path in NAV_PAGES:
        tab.goto(site.url(path))
        raw = tab.evaluate(PAGE_FACTS_SCRIPT)
        facts[path] = PageFacts(
            title=raw["title"],
            description=raw["description"],
            headings=raw["headings"],
            images=[Image(**image) for image in raw["images"]],
            frames=[Frame(**frame) for frame in raw["frames"]],
        )
    context.close()
    return facts


def test_the_nav_lists_every_page_of_the_sitemap(site: Site) -> None:
    assert sorted(site.sitemap_pages()) == sorted(NAV_PAGES)


def test_each_page_has_one_h1(page_facts: Mapping[str, PageFacts], page: str) -> None:
    assert page_facts[page].headings.count(1) == 1


def test_the_headings_of_each_page_skip_no_level(page_facts: Mapping[str, PageFacts], page: str) -> None:
    levels = page_facts[page].headings
    skips = [(level, next_level) for level, next_level in itertools.pairwise(levels) if next_level > level + 1]
    assert skips == []


def _file_name(address: str) -> str:
    return unquote(posixpath.basename(urlsplit(address).path))


def test_each_image_has_alt_text_that_is_not_its_file_name(page_facts: Mapping[str, PageFacts], page: str) -> None:
    bad = [
        image.source
        for image in page_facts[page].images
        if (image.alt or "").strip() in {"", _file_name(image.source), image.source}
    ]
    assert bad == []


def test_each_frame_has_a_title(page_facts: Mapping[str, PageFacts], page: str) -> None:
    untitled = [frame.source for frame in page_facts[page].frames if not (frame.title or "").strip()]
    assert untitled == []


def test_each_embedded_pdf_has_a_link_to_the_file_next_to_it(page_facts: Mapping[str, PageFacts], page: str) -> None:
    unlinked = [
        frame.source
        for frame in page_facts[page].frames
        if urlsplit(frame.source).path.endswith(".pdf") and not frame.linked_nearby
    ]
    assert unlinked == []


def test_the_title_of_each_page_names_the_page(page_facts: Mapping[str, PageFacts], page: str) -> None:
    name, _, _ = page_facts[page].title.rpartition(SITE_NAME_SEPARATOR)
    assert name.strip()


def test_no_two_pages_have_the_same_title(page_facts: Mapping[str, PageFacts]) -> None:
    counts = collections.Counter(facts.title for facts in page_facts.values())
    assert [title for title, count in counts.items() if count > 1] == []


def test_each_page_has_a_description(page_facts: Mapping[str, PageFacts], page: str) -> None:
    assert page_facts[page].description.strip()


def test_no_two_pages_have_the_same_description(page_facts: Mapping[str, PageFacts]) -> None:
    counts = collections.Counter(facts.description for facts in page_facts.values())
    assert [description for description, count in counts.items() if count > 1] == []
