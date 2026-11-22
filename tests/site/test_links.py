"""Check every link of the pages that the project writes: internal links offline, external links on a schedule.

The strict build of Zensical checks the links of the Markdown, but not the links of raw HTML, the anchors of the API
reference, or the links that the theme writes. This module reads each built page and checks each ``href`` and ``src``:

- An internal link must name a file of the site. Its anchor, if any, must name an ``id`` of the target page. A link to
  the public address of the site (``site_url``) is internal too, because the README uses absolute addresses.
- An external link must answer with a status below 400. A remote site can fail for a day, so this check runs only with
  ``--external-links``, in a weekly workflow, and never blocks a pull request.

The marimo exports in ``notebooks/`` are not pages of the project, so their links are not checked. The links to them
are.
"""

import html.parser
import posixpath
from collections.abc import Iterable, Mapping
from typing import override
from urllib.parse import unquote, urljoin, urlsplit

import attrs
import pytest
from playwright.sync_api import APIRequestContext, Playwright
from playwright.sync_api import Error as PlaywrightError

from tests.site.conftest import Site

LINK_ATTRIBUTES = frozenset({"href", "src"})
# The hosts whose answer to a script is not a sign of a broken link, each with its reason. Add a host only after the
# weekly check fails on it for a link that works in a browser.
FLAKY_HOSTS: dict[str, str] = {}
REQUEST_TIMEOUT_MILLISECONDS = 30_000
FIRST_ERROR_STATUS = 400
USER_AGENT = "Mozilla/5.0 (compatible; link check of a documentation site)"


@attrs.frozen
class Link:
    """One ``href`` or ``src`` of a page.

    Attributes
    ----------
    page: str
        The path of the page in the site, for example ``usage/index.html``.
    address: str
        The absolute address of the link on the public site.

    """

    page: str
    address: str


class _LinkParser(html.parser.HTMLParser):
    """Collect the ``href`` and ``src`` values and the ``id`` values of one page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in LINK_ATTRIBUTES:
                self.links.append(value)
            elif name == "id" or (tag == "a" and name == "name"):
                self.anchors.add(value)


@attrs.frozen
class SiteLinks:
    """The links and the anchors of every page of the project.

    Attributes
    ----------
    links: list[Link]
        Each link of each page, with the address resolved against the public address of the page.
    anchors: Mapping[str, set[str]]
        The ``id`` values of each page, by the path of the page in the site.

    """

    links: list[Link]
    anchors: Mapping[str, set[str]]


@pytest.fixture(scope="module")
def site_links(site: Site) -> SiteLinks:
    """Read the links and the anchors of every page of the project.

    Returns
    -------
    SiteLinks
        The links and the anchors.

    """
    links: list[Link] = []
    anchors: dict[str, set[str]] = {}
    for path in site.project_pages():
        relative = path.relative_to(site.directory).as_posix()
        page_address = site.site_url + relative.removesuffix("index.html")
        parser = _LinkParser()
        parser.feed(path.read_text(encoding="utf-8"))
        anchors[relative] = parser.anchors
        links.extend(Link(page=relative, address=urljoin(page_address, value)) for value in parser.links)
    return SiteLinks(links=links, anchors=anchors)


def _target_file(site: Site, address: str) -> str | None:
    """Find the file of the site that an internal address names.

    Returns
    -------
    str | None
        The path of the file in the site, or ``None`` if the site has no such file.

    """
    path = unquote(urlsplit(address).path.removeprefix(urlsplit(site.site_url).path))
    normal = posixpath.normpath(path) if path else ""
    if normal.startswith(".."):
        return None
    candidates = [f"{normal}/index.html".removeprefix("./"), normal] if normal not in {"", "."} else ["index.html"]
    return next((candidate for candidate in candidates if (site.directory / candidate).is_file()), None)


def _internal_problems(site: Site, site_links: SiteLinks) -> Iterable[str]:
    for link in site_links.links:
        if not link.address.startswith(site.site_url):
            continue
        target = _target_file(site, link.address)
        if target is None:
            yield f"{link.page}: {link.address} names no file of the site"
            continue
        anchor = unquote(urlsplit(link.address).fragment)
        if anchor and target in site_links.anchors and anchor not in site_links.anchors[target]:
            yield f"{link.page}: {link.address} names no id of {target}"


def test_every_internal_link_and_anchor_names_a_part_of_the_site(site: Site, site_links: SiteLinks) -> None:
    problems = list(_internal_problems(site, site_links))
    assert not problems, "\n".join(problems)


def _external_addresses(site: Site, site_links: SiteLinks) -> list[str]:
    addresses = {
        link.address.partition("#")[0]
        for link in site_links.links
        if urlsplit(link.address).scheme in {"http", "https"} and not link.address.startswith(site.site_url)
    }
    return sorted(address for address in addresses if urlsplit(address).hostname not in FLAKY_HOSTS)


def _failure(client: APIRequestContext, address: str) -> str | None:
    """Request an external address, and describe the failure.

    Returns
    -------
    str | None
        A description of the failure, or ``None`` if the address answers with a status below 400.

    """
    try:
        response = client.get(address, timeout=REQUEST_TIMEOUT_MILLISECONDS, fail_on_status_code=False)
    except PlaywrightError as error:
        return f"{address}: {error.message.splitlines()[0]}"
    return None if response.status < FIRST_ERROR_STATUS else f"{address}: HTTP {response.status}"


@pytest.mark.external_links
def test_every_external_link_answers(site: Site, site_links: SiteLinks, playwright: Playwright) -> None:
    client = playwright.request.new_context(user_agent=USER_AGENT)
    failures = [
        failure
        for address in _external_addresses(site, site_links)
        if (failure := _failure(client, address)) is not None
    ]
    client.dispose()
    assert not failures, "\n".join(failures)
