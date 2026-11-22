"""Fixtures of the site suite: the built site, a local server, the browser and the list of pages.

The suite builds the site once for each session with the command of CI (``scripts.docs.build``), and serves the
directory on a free port of the loopback interface under the path of ``site_url`` (``/EBaMl/``), as GitHub Pages does.
With ``--prebuilt-site``, the suite serves the ``site/`` directory as it is: CI builds the site in one job and checks it
in another.
Each test gets the address of the server, so a link that works locally and fails on GitHub Pages fails here too.

The pages come from the nav of ``zensical.toml``, so a new page gets the checks with no edit of a test.
``test_the_nav_lists_every_page_of_the_sitemap`` makes sure that no built page escapes the nav.

The Node tools (axe-core, html-validate and Lighthouse CI) are pinned in ``tests/site/package.json``. Install them
with ``npm ci --prefix tests/site``. Playwright starts its own Chromium. To use another build of Chromium, set
``EBAML_CHROMIUM`` to its executable.
"""

import functools
import http.server
import os
import re
import threading
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import override
from urllib.parse import urlsplit

import attrs
import pytest
from playwright.sync_api import Browser, Playwright, sync_playwright
from scripts.docs.build import CONFIG, SITE_DIR, build

SUITE_DIR = Path(__file__).resolve().parent
NODE_MODULES = SUITE_DIR / "node_modules"
CHROMIUM_VARIABLE = "EBAML_CHROMIUM"
EXTERNAL_LINKS_OPTION = "--external-links"
PREBUILT_SITE_OPTION = "--prebuilt-site"
SITEMAP_LOCATION: re.Pattern[str] = re.compile(r"<loc>([^<]+)</loc>")


type NavEntry = Mapping[str, str | list[NavEntry]]


def _nav_files(entries: Sequence[NavEntry]) -> list[str]:
    files: list[str] = []
    for entry in entries:
        for target in entry.values():
            files.extend(_nav_files(target) if isinstance(target, list) else [target])
    return files


def page_path(markdown_file: str) -> str:
    """Give the path of the page of a Markdown file, relative to the root of the site.

    Parameters
    ----------
    markdown_file: str
        The file, relative to ``docs/``, for example ``usage.md``.

    Returns
    -------
    str
        The path, for example ``usage/``. The home page has the empty path.

    """
    stem = markdown_file.removesuffix(".md")
    if stem == "index":
        return ""
    return f"{stem.removesuffix('/index')}/"


CONFIGURATION = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["project"]
NAV_PAGES: tuple[str, ...] = tuple(page_path(name) for name in _nav_files(CONFIGURATION["nav"]))


@attrs.frozen
class Site:
    """The built site and the local server that serves it.

    Attributes
    ----------
    directory: Path
        The built site.
    base_url: str
        The address of the root of the site on the local server, with a slash at the end.
    site_url: str
        The public address of the site, from ``site_url`` of ``zensical.toml``.

    """

    directory: Path
    base_url: str
    site_url: str

    def url(self, path: str) -> str:
        """Give the local address of a page.

        Parameters
        ----------
        path: str
            The path of the page, relative to the root of the site.

        Returns
        -------
        str
            The address on the local server.

        """
        return self.base_url + path

    def project_pages(self) -> list[Path]:
        """List the pages that the project writes: each ``index.html`` and the 404 page.

        The marimo exports in ``notebooks/`` and the files of the theme in ``assets/`` are not pages of the project.

        Returns
        -------
        list[Path]
            The pages, in a fixed order.

        """
        return [*sorted(self.directory.rglob("index.html")), self.directory / "404.html"]

    def sitemap_pages(self) -> list[str]:
        """List the pages of the sitemap of the build.

        Returns
        -------
        list[str]
            The path of each page relative to the root of the site, in the order of the sitemap.

        """
        sitemap = (self.directory / "sitemap.xml").read_text(encoding="utf-8")
        return [location.removeprefix(self.site_url) for location in SITEMAP_LOCATION.findall(sitemap)]


def _handler_class(directory: Path, base_path: str) -> type[http.server.SimpleHTTPRequestHandler]:
    class SiteHandler(http.server.SimpleHTTPRequestHandler):
        @override
        def translate_path(self, path: str) -> str:
            """Serve the site under the base path only, as GitHub Pages does, so that a link that leaves it fails.

            Returns
            -------
            str
                The file to serve. A path outside the base path gives a file that does not exist.

            """
            if not path.startswith(base_path):
                return str(directory / "outside-the-base-path")
            return super().translate_path("/" + path.removeprefix(base_path))

        @override
        def log_message(self, format: str, *args: str | int) -> None:
            """Stay silent, so that the test output shows no access log."""

    return SiteHandler


@pytest.fixture(scope="session")
def site(request: pytest.FixtureRequest) -> Iterator[Site]:
    """Build the site with the command of CI, and serve it on a free port under the path of ``site_url``.

    Yields
    ------
    Site
        The built site and the address of the server.

    """
    if request.config.getoption(PREBUILT_SITE_OPTION):
        if not (SITE_DIR / "index.html").is_file():
            pytest.fail(f"{PREBUILT_SITE_OPTION}: {SITE_DIR} holds no built site")
    elif build():
        pytest.fail("the strict docs build failed, see its output above")
    site_url: str = CONFIGURATION["site_url"]
    base_path = urlsplit(site_url).path
    handler = functools.partial(_handler_class(SITE_DIR, base_path), directory=str(SITE_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Site(directory=SITE_DIR, base_url=f"http://127.0.0.1:{server.server_port}{base_path}", site_url=site_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="session")
def playwright() -> Iterator[Playwright]:
    """Start Playwright once for the session.

    Yields
    ------
    Playwright
        The running Playwright.

    """
    with sync_playwright() as running:
        yield running


@pytest.fixture(scope="session")
def chromium_path(playwright: Playwright) -> str:
    """Find the Chromium executable: ``EBAML_CHROMIUM`` if it is set, else the build that Playwright installed.

    Returns
    -------
    str
        The path of the executable.

    """
    return os.environ.get(CHROMIUM_VARIABLE) or playwright.chromium.executable_path


@pytest.fixture(scope="session")
def browser(playwright: Playwright, chromium_path: str) -> Iterator[Browser]:
    """Start one headless Chromium for the session.

    Yields
    ------
    Browser
        The browser. Each test opens its own context, with its own viewport and color scheme.

    """
    started = playwright.chromium.launch(executable_path=chromium_path)
    yield started
    started.close()


def node_tool(*parts: str) -> Path:
    """Find a file of the Node tools of the suite, and fail with the install command if it is missing.

    Parameters
    ----------
    *parts: str
        The path of the file inside ``tests/site/node_modules``.

    Returns
    -------
    Path
        The file.

    """
    path = NODE_MODULES.joinpath(*parts)
    if not path.exists():
        pytest.fail(f"{path} is missing: install the Node tools of the suite with `npm ci --prefix tests/site`")
    return path


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the options that check the external links, and that check a site that is already built."""
    parser.addoption(
        PREBUILT_SITE_OPTION,
        action="store_true",
        help="check the site in site/ as it is, with no build first",
    )
    parser.addoption(
        EXTERNAL_LINKS_OPTION,
        action="store_true",
        help="also check the external links of the site; a weekly workflow runs it",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Leave out the check of the external links, unless the command asks for it.

    A remote site can fail for a day, so the check runs on a weekly schedule only, and never blocks a pull request.
    """
    if config.getoption(EXTERNAL_LINKS_OPTION):
        return
    selected = [item for item in items if item.get_closest_marker("external_links") is None]
    deselected = [item for item in items if item.get_closest_marker("external_links") is not None]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run each test that takes ``page`` once for each page of the nav."""
    if "page" in metafunc.fixturenames:
        metafunc.parametrize("page", NAV_PAGES, ids=[path.rstrip("/") or "home" for path in NAV_PAGES])
