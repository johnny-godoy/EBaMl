"""Validate the HTML of the pages that the project writes with html-validate.

The pages are each ``index.html`` of the site and the 404 page. The marimo exports in ``notebooks/`` and the files of
the theme in ``assets/`` are not checked, because the project does not write them. ``tests/site/package.json`` pins
the version of html-validate.

``RULES`` turns off the rules that the markup of the theme breaks on every page, each with the reason. The project
cannot change that markup, so a page of the project that breaks such a rule is not found. Each other rule of the presets
applies.
"""

from pathlib import Path

import orjson
from scripts.fixed_command import run_fixed

from tests.site.conftest import Site, node_tool

PRESETS = ("html-validate:standard", "html-validate:a11y", "html-validate:document")
RULES = {
    # reason: an integrity hash makes sense only for a file of another origin; the site loads its own files only.
    "require-sri": ["error", {"target": "crossorigin"}],
    # reason: the theme writes the table of contents twice on each page, once in the nav for a small screen and once
    # beside the content, as two nav elements with the same label "On this page".
    "unique-landmark": "off",
    # reason: the theme writes autocomplete="off" on the hidden checkboxes that open the nav and the search.
    "valid-autocomplete": "off",
    # reason: the theme writes tabindex="" on the label of each section of the nav.
    "attribute-allowed-values": "off",
    # reason: the search form of the theme sends no request; its script shows the results while the reader types.
    "wcag/h32": "off",
}


def test_the_pages_of_the_project_are_valid_html(site: Site, tmp_path: Path) -> None:
    config = tmp_path / "htmlvalidate.json"
    config.write_bytes(orjson.dumps({"root": True, "extends": list(PRESETS), "rules": RULES}))
    command = [node_tool(".bin", "html-validate"), "--config", config, "--formatter", "text", *site.project_pages()]
    result = run_fixed(command)
    assert result.returncode == 0, result.stdout + result.stderr
