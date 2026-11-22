"""Check that no page scrolls sideways at the width of a small phone.

A page that is wider than the screen makes a phone user scroll in two directions to read each line (WCAG 2.2, success
criterion 1.4.10, reflow). A code block or a table may be wider, if it scrolls by itself and the page does not.
"""

from playwright.sync_api import Browser

from tests.site.conftest import Site

PHONE_VIEWPORT = {"width": 360, "height": 740}
# Lists each element that ends right of the screen, unless an ancestor that fits the screen scrolls it. An ancestor
# with ``overflow-x: hidden`` cuts the content off, so it does not count: the reader cannot reach the hidden part.
OVERFLOW_SCRIPT = """() => {
  const width = document.documentElement.clientWidth;
  const scrolledInside = (element) => {
    for (let parent = element.parentElement; parent !== null; parent = parent.parentElement) {
      const scrolls = ["auto", "scroll"].includes(getComputedStyle(parent).overflowX);
      if (scrolls && parent.getBoundingClientRect().right <= width + 1) return true;
    }
    return false;
  };
  const wide = [...document.querySelectorAll("body *")].filter((element) => {
    const box = element.getBoundingClientRect();
    return box.width > 0 && box.right > width + 1 && !scrolledInside(element);
  });
  return {
    page_width: document.documentElement.scrollWidth,
    screen_width: width,
    too_wide: wide.slice(0, 5).map((element) => element.outerHTML.slice(0, 100)),
  };
}"""


def test_no_page_scrolls_sideways_at_the_width_of_a_phone(site: Site, browser: Browser, page: str) -> None:
    context = browser.new_context(viewport=PHONE_VIEWPORT)
    tab = context.new_page()
    tab.goto(site.url(page))
    layout = tab.evaluate(OVERFLOW_SCRIPT)
    context.close()
    assert layout["too_wide"] == []
    assert layout["page_width"] <= layout["screen_width"]
