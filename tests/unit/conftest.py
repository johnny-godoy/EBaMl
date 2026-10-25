"""Pytest hooks and Hypothesis settings for all unit tests."""

import os

import pytest
from hypothesis import settings

# A random failure in CI is a flaky build that blocks other pull requests, so CI and the mutation runs draw the same
# examples every time and keep no example database. A timing limit fails at random on a slow runner, so it is off.
settings.register_profile("ci", derandomize=True, database=None, deadline=None)
settings.load_profile("ci" if os.environ.get("CI") or os.environ.get("MUTATION_FIRST") else "default")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the tests of the module under mutation first.

    ``scripts/mutation.py`` sets ``MUTATION_FIRST`` to the name of the mutated module. A mutant of that module is
    killed by its own tests in most cases. The run stops at the first failure, so the tests of that module come first.
    Without the variable, the order does not change.
    """
    module = os.environ.get("MUTATION_FIRST")
    if module:
        items.sort(key=lambda item: item.path.stem != f"test_{module}")
