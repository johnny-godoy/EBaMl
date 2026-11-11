"""Tests for the status of the claim that a set is the smallest."""

import pytest

from ebaml.local.minimality import MinimalityStatus


@pytest.mark.parametrize(
    ("status", "is_smallest"),
    [
        (MinimalityStatus.SORTING_PROOF, True),
        (MinimalityStatus.SOLVER_PROOF, True),
        (MinimalityStatus.AT_LOWER_BOUND, True),
        (MinimalityStatus.UNPROVED, False),
    ],
)
def test_only_an_unproved_set_is_not_called_the_smallest(status: MinimalityStatus, *, is_smallest: bool) -> None:
    assert status.is_smallest is is_smallest


def test_the_values_of_the_statuses_are_their_names_in_lower_case() -> None:
    # the values reach the JSON of the MCP tool, so they are part of the interface
    assert {status.name.lower() == status.value for status in MinimalityStatus} == {True}
    assert len(MinimalityStatus) == 4
