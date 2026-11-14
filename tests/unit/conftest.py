"""Pytest hooks and Hypothesis settings for all unit tests."""

import io
import os

import progressbar
import pytest
from hypothesis import settings
from scripts.mutation import covering_tests

# A random failure in CI is a flaky build that blocks other pull requests, so CI and the mutation runs draw the same
# examples every time and keep no example database. A timing limit fails at random on a slow runner, so it is off.
settings.register_profile("ci", derandomize=True, database=None, deadline=None)
settings.load_profile("ci" if os.environ.get("CI") or os.environ.get("MUTATION_FIRST") else "default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run the tests that can kill the mutant first.

    ``scripts/mutation/run.py`` sets ``MUTATION_FIRST`` to the name of the mutated module, ``MUTATION_MODULE`` to
    its path, ``MUTATION_ORIGINAL`` to the project root with the unmutated source, and
    ``MUTATION_COVERAGE`` to the coverage map of the baseline. The run stops at the first failure, so the tests
    that execute the mutated statements come first, then the other tests of the mutated module, then the rest.
    Without the variables, the order does not change.
    """
    module = os.environ.get("MUTATION_FIRST")
    if not module:
        return
    covering = covering_tests.covering_from_environment(config.rootpath, os.environ)
    items.sort(key=lambda item: (item.nodeid not in covering, item.path.stem != f"test_{module}"))


@pytest.fixture
def error_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Give the test the stream that its progress bars write to, so that the test reads their lines.

    progressbar2 replaces ``sys.stderr`` with the error stream that it saw at import, so this fixture replaces that
    stream, and the output stream too.

    Returns
    -------
    io.StringIO
        The replacement for the error stream.

    """
    monkeypatch.setattr(progressbar.utils.streams, "original_stdout", io.StringIO())
    stream = io.StringIO()
    monkeypatch.setattr(progressbar.utils.streams, "original_stderr", stream)
    return stream
