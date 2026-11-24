"""Measure four kinds of page with Lighthouse CI, and hold each score to its budget.

The pages are the home page, a guide, a notebook page and the API reference: the API reference is the longest page, and
the notebook page embeds a marimo export. The budgets of accessibility, best practices and SEO are 100. The budget of
performance of each page is its first measured score minus a margin of 0.15, because the score changes with the load of
the machine. Lighthouse simulates a phone on a slow network, and the local server sends no compression, so the
performance scores are lower than on GitHub Pages.

``tests/site/package.json`` pins the version of Lighthouse CI. A failure prints each assertion that failed, with the
expected and the found score, and each audit that lost points in a category with a budget of 100. The reports stay in
``.lighthouseci/`` under the temporary directory of the test.
"""

import re
from pathlib import Path

import orjson
from scripts.fixed_command import run_fixed

from tests.site.conftest import Site, node_tool

# The budget of performance of each page, by its path. The first measured scores were 0.78, 0.88, 0.50 and 0.53.
PERFORMANCE_BUDGETS = {
    "": 0.6,
    "usage/": 0.7,
    "global-notebook/": 0.35,
    "api/": 0.35,
}
FULL_SCORE_CATEGORIES = ("accessibility", "best-practices", "seo")
# Chromium does not start its sandbox as root, and a container of a contributor often runs as root. The pages are ours.
CHROME_FLAGS = "--headless=new --no-sandbox"
# The script of the theme asks the GitHub API for the stars and the release of the repository on each page. GitHub
# limits the requests without a token to 60 an hour for each address, and the runners of CI share addresses, so the
# answer can be an error that Lighthouse counts against best practices. The answer is not part of the page. Lighthouse
# ignores the console message of a request that it blocks.
BLOCKED_URL_PATTERNS = ("*api.github.com*",)


def _configuration(site: Site, chromium_path: str) -> dict[str, dict[str, object]]:
    """Write the configuration of Lighthouse CI: the pages to measure and the budget of each score.

    Returns
    -------
    dict[str, dict[str, object]]
        The configuration, as ``lighthouserc.json`` holds it.

    """
    matrix = [
        {
            "matchingUrlPattern": f"^{re.escape(site.url(path))}$",
            "assertions": {
                "categories:performance": ["error", {"minScore": budget}],
                **{f"categories:{category}": ["error", {"minScore": 1}] for category in FULL_SCORE_CATEGORIES},
            },
        }
        for path, budget in PERFORMANCE_BUDGETS.items()
    ]
    return {
        "ci": {
            "collect": {
                "url": [site.url(path) for path in PERFORMANCE_BUDGETS],
                "numberOfRuns": 1,
                "chromePath": chromium_path,
                "settings": {"chromeFlags": CHROME_FLAGS, "blockedUrlPatterns": list(BLOCKED_URL_PATTERNS)},
            },
            "assert": {"assertMatrix": matrix},
        },
    }


def _audits_below_full_score(directory: Path) -> list[str]:
    """List the audits that lost points in a category with a budget of 100, for the failure message.

    Returns
    -------
    list[str]
        One line for each audit and page: the page, the category, the audit, its score and its title.

    """
    lines: list[str] = []
    for report_path in sorted((directory / ".lighthouseci").glob("lhr-*.json")):
        report = orjson.loads(report_path.read_bytes())
        for category in FULL_SCORE_CATEGORIES:
            for reference in report["categories"][category]["auditRefs"]:
                audit = report["audits"][reference["id"]]
                if reference["weight"] and audit["score"] is not None and audit["score"] < 1:
                    lines.append(f"{report['finalUrl']} {category}: {audit['id']} = {audit['score']}, {audit['title']}")
    return lines


def test_the_scores_of_lighthouse_meet_the_budgets(site: Site, chromium_path: str, tmp_path: Path) -> None:
    config = tmp_path / "lighthouserc.json"
    config.write_bytes(orjson.dumps(_configuration(site, chromium_path)))
    lhci = node_tool(".bin", "lhci")
    run_fixed([lhci, "collect", f"--config={config}"], cwd=tmp_path, check=True)
    result = run_fixed([lhci, "assert", f"--config={config}"], cwd=tmp_path)
    assert result.returncode == 0, "\n".join([result.stdout + result.stderr, *_audits_below_full_score(tmp_path)])
