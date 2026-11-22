"""Run axe-core on each page of the site, in the light and the dark color scheme, with the WCAG 2.2 A and AA rules.

The two schemes have different colors, so a contrast that passes in one can fail in the other. ``emulate_media`` sets
the color scheme of the system, and the palette of ``zensical.toml`` follows it. Each test first checks that the page
shows the expected scheme, so a dark run that shows the light scheme fails and does not pass in silence.

``tests/site/package.json`` pins the version of axe-core. ``CONTRIBUTING.md`` tells how to read a failure.
"""

from typing import Literal

import attrs
import pytest
from playwright.sync_api import Browser

from tests.site.conftest import Site, node_tool

# The tags of axe-core for the success criteria of WCAG 2.0, 2.1 and 2.2 at the levels A and AA.
WCAG_TAGS = ("wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa")
# The value of ``data-md-color-scheme`` that the theme sets for each color scheme of the system.
THEME_SCHEMES = {"light": "default", "dark": "slate"}
AXE_RUN_SCRIPT = """(tags) => axe.run(document, {runOnly: {type: "tag", values: tags}}).then((results) =>
  results.violations.map((violation) => ({
    rule: violation.id,
    impact: violation.impact || "",
    help: violation.help,
    help_url: violation.helpUrl,
    targets: violation.nodes.map((node) => node.target.join(" ")),
    summary: violation.nodes[0].failureSummary || "",
  })))"""


@attrs.frozen
class Violation:
    """One rule of axe that a page breaks, with the elements that break it.

    Attributes
    ----------
    rule: str
        The ID of the rule, for example ``color-contrast``.
    impact: str
        How much the problem hurts a user: ``minor``, ``moderate``, ``serious`` or ``critical``.
    help: str
        One sentence that says what the rule asks for.
    help_url: str
        The page of Deque University that explains the rule and its fixes.
    targets: list[str]
        A CSS selector for each element that breaks the rule.
    summary: str
        What is wrong with the first element, for example the contrast ratio and the two colors.

    """

    rule: str
    impact: str
    help: str
    help_url: str
    targets: list[str]
    summary: str

    def report(self) -> str:
        """Write the violation for the failure message of a test.

        Returns
        -------
        str
            The rule, its impact, its help, the elements and the summary, on several lines.

        """
        elements = "\n".join(f"    {target}" for target in self.targets)
        return f"{self.rule} ({self.impact}): {self.help}\n  {self.help_url}\n{elements}\n  {self.summary}"


@pytest.fixture(scope="module")
def axe_script() -> str:
    """Read the pinned axe-core script.

    Returns
    -------
    str
        The minified script, ready to inject into a page.

    """
    return node_tool("axe-core", "axe.min.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("color_scheme", ["light", "dark"])
def test_each_page_passes_the_wcag_rules_of_axe(
    site: Site,
    browser: Browser,
    axe_script: str,
    page: str,
    color_scheme: Literal["light", "dark"],
) -> None:
    context = browser.new_context(color_scheme=color_scheme)
    tab = context.new_page()
    tab.goto(site.url(page))
    assert tab.locator("body").get_attribute("data-md-color-scheme") == THEME_SCHEMES[color_scheme]
    tab.add_script_tag(content=axe_script)
    violations = [Violation(**violation) for violation in tab.evaluate(AXE_RUN_SCRIPT, list(WCAG_TAGS))]
    context.close()
    assert not violations, "\n\n".join(violation.report() for violation in violations)
