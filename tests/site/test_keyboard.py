"""Walk through each page with the Tab key, as a person who uses no mouse does.

axe checks the markup of a page, but not what the keyboard reaches. The tests press Tab until the focus leaves the
page, and record each element that gets the focus. Then they check three things:

1. The keyboard reaches the search, the theme toggle, each link of the nav that the page shows, and each copy button.
2. The focus moves through the parts of the page in the order that a reader sees them: the header, the nav, the table
   of contents, the content and the footer. No element has a positive ``tabindex``, which breaks that order.
3. Each element shows a focus indicator when it has the focus: an outline, or a change of its shadow, background or
   text color.

The selectors in ``TARGETS`` and ``REGIONS`` name the elements of the theme. If a change of the theme renames one, the
walk finds no such element, and the first test fails.
"""

from collections.abc import Callable, Iterator

import attrs
import pytest
from playwright.sync_api import Browser, Page

from tests.site.conftest import Site

# The controls that the keyboard must reach, by the name in a failure message. The theme toggle is a group of radio
# buttons with no size. The Tab key enters the group once, so the group counts as one control.
TARGETS = {
    "search": "button.md-search__button",
    "theme toggle": "input.md-option",
    "nav link": ".md-sidebar--primary .md-nav--primary a.md-nav__link[href]",
    "copy button": "button.md-code__button",
}
# The parts of a page in the order that a reader sees them on a wide screen. The "Back to top" button of the theme
# floats over the content, and its place in the markup is after the content.
REGIONS = (
    ".md-skip",
    ".md-header",
    ".md-sidebar--primary",
    ".md-sidebar--secondary",
    ".md-content",
    ".md-top",
    ".md-footer",
)
VIEWPORT = {"width": 1280, "height": 800}
# The JavaScript functions that both scripts below use. A radio button of the theme toggle has no size, and the theme
# draws its focus ring on the label that follows it. A focus indicator is an outline, or a change of the shadow, the
# background or the text color against the element without the focus.
STYLE_FUNCTIONS = """
  const ringOwner = (element) => {
    const box = element.getBoundingClientRect();
    const next = element.nextElementSibling;
    return box.width === 0 && box.height === 0 && next !== null && next.tagName === "LABEL" ? next : element;
  };
  const indicatorStyle = (element) => {
    const style = getComputedStyle(ringOwner(element));
    return JSON.stringify([style.boxShadow, style.backgroundColor, style.color]);
  };
"""
# Marks each visible target, keeps the style of each element without the focus, and empties each frame. The content of
# a frame is a marimo export or a PDF, which the project does not write, and an empty frame takes one stop of the walk.
# Returns the number of elements that can get the focus, so that the walk has an upper limit.
MARK_SCRIPT = f"""(targets) => {{
  {STYLE_FUNCTIONS}
  for (const [name, selector] of Object.entries(targets)) {{
    document.querySelectorAll(selector).forEach((element, index) => {{
      const radio = element.type === "radio";
      const number = radio ? element.name : index;
      if (radio || element.checkVisibility()) element.dataset.ebamlTarget = `${{name}} ${{number}}`;
    }});
  }}
  const focusable = document.querySelectorAll(
    "a[href], button, input, select, textarea, iframe, [tabindex], code, pre");
  focusable.forEach((element) => {{ element.dataset.ebamlStyle = indicatorStyle(element); }});
  document.querySelectorAll("iframe").forEach((frame) => {{ frame.srcdoc = "<p>An empty frame of the test.</p>"; }});
  return focusable.length;
}}"""
# Describes the element that has the focus. It waits one turn of the event loop first, because the script that marks a
# frame with the focus runs after the blur event of the page.
FOCUS_SCRIPT = f"""async (regions) => {{
  {STYLE_FUNCTIONS}
  await new Promise((resolve) => setTimeout(resolve));
  const element = document.activeElement;
  if (element === null || element === document.body) return null;
  const style = getComputedStyle(ringOwner(element));
  const outline = style.outlineStyle !== "none" && parseFloat(style.outlineWidth) > 0;
  return {{
    key: element.dataset.ebamlKey || (element.dataset.ebamlKey = String(Math.random())),
    target: element.dataset.ebamlTarget || "",
    description: `${{element.tagName.toLowerCase()}} "${{(element.getAttribute("aria-label") || element.title
      || element.textContent || "").trim().slice(0, 40)}}"`,
    region: regions.findIndex((selector) => element.closest(selector) !== null),
    ring: outline || indicatorStyle(element) !== element.dataset.ebamlStyle,
  }};
}}"""
POSITIVE_TABINDEX_SCRIPT = """() => [...document.querySelectorAll("[tabindex]")]
  .filter((element) => element.tabIndex > 0).map((element) => element.outerHTML.slice(0, 80))"""
TARGETS_SCRIPT = (
    """() => [...document.querySelectorAll("[data-ebaml-target]")].map((element) => element.dataset.ebamlTarget)"""
)


@attrs.frozen
class Stop:
    """One element that got the focus in the walk.

    Attributes
    ----------
    target: str
        The name and the number of the target, for example ``copy button 2``, or an empty text for another element.
    description: str
        The tag and the text of the element, for a failure message.
    region: int
        The position of the part of the page in ``REGIONS``, or -1 for an element outside them.
    ring: bool
        Whether the element, or its label, shows a focus ring.

    """

    target: str
    description: str
    region: int
    ring: bool


@attrs.frozen
class Walk:
    """The result of a walk through one page with the Tab key.

    Attributes
    ----------
    stops: list[Stop]
        Each element that got the focus, in order.
    targets: list[str]
        Each visible target of the page, which the walk must reach.
    positive_tabindex: list[str]
        The start of the markup of each element with a positive ``tabindex``.

    """

    stops: list[Stop]
    targets: list[str]
    positive_tabindex: list[str]


def _walk(tab: Page) -> Walk:
    limit = tab.evaluate(MARK_SCRIPT, TARGETS) + 2
    tab.wait_for_load_state("load")
    stops: list[Stop] = []
    seen: set[str] = set()
    for _ in range(limit):
        tab.keyboard.press("Tab")
        focus = tab.evaluate(FOCUS_SCRIPT, list(REGIONS))
        if focus is None or focus["key"] in seen:
            break
        seen.add(focus.pop("key"))
        stops.append(Stop(**focus))
    return Walk(
        stops=stops,
        targets=tab.evaluate(TARGETS_SCRIPT),
        positive_tabindex=tab.evaluate(POSITIVE_TABINDEX_SCRIPT),
    )


@pytest.fixture(scope="module")
def walk_page(site: Site, browser: Browser) -> Iterator[Callable[[str], Walk]]:
    """Give a function that walks through a page once, and keeps the result for the other tests of the page.

    Yields
    ------
    Callable[[str], Walk]
        The function. It takes the path of the page.

    """
    context = browser.new_context(viewport=VIEWPORT)
    walks: dict[str, Walk] = {}

    def walk(path: str) -> Walk:
        if path not in walks:
            tab = context.new_page()
            tab.goto(site.url(path))
            walks[path] = _walk(tab)
            tab.close()
        return walks[path]

    yield walk
    context.close()


def test_the_keyboard_reaches_the_search_the_toggle_the_nav_and_each_copy_button(
    walk_page: Callable[[str], Walk],
    page: str,
) -> None:
    walk = walk_page(page)
    reached = {stop.target for stop in walk.stops}
    assert {target.rpartition(" ")[0] for target in walk.targets} >= {"search", "theme toggle", "nav link"}
    assert [target for target in walk.targets if target not in reached] == []


def test_the_focus_moves_through_the_parts_of_the_page_in_reading_order(
    walk_page: Callable[[str], Walk],
    page: str,
) -> None:
    walk = walk_page(page)
    regions = [stop.region for stop in walk.stops]
    assert walk.positive_tabindex == []
    assert [stop.description for stop in walk.stops if stop.region < 0] == []
    assert regions == sorted(regions)


def test_each_element_shows_a_focus_ring(walk_page: Callable[[str], Walk], page: str) -> None:
    walk = walk_page(page)
    assert [stop.description for stop in walk.stops if not stop.ring] == []
