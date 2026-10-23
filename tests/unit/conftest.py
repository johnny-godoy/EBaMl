"""Pytest hooks for all unit tests."""

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the tests of the module under mutation first.

    ``scripts/mutation.py`` sets ``MUTATION_FIRST`` to the name of the mutated module. A mutant of that module is
    killed by its own tests in most cases. The run stops at the first failure, so the tests of that module come first.
    Without the variable, the order does not change.
    """
    module = os.environ.get("MUTATION_FIRST")
    if module:
        items.sort(key=lambda item: item.path.stem != f"test_{module}")
