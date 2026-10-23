"""Tests for the exceptions."""

import pytest

from ebaml.custom_types import FeatureType
from ebaml.exceptions import (
    EBMGraphError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    InvalidOptionError,
)


def test_invalid_option_message() -> None:
    err = InvalidOptionError("x", ["a", "b"])
    assert str(err) == "Invalid option: x. Valid options are ['a', 'b']."
    assert isinstance(err, ValueError)


def test_from_enum_lists_member_names() -> None:
    err = InvalidOptionError.from_enum("bad", FeatureType)
    assert isinstance(err, InvalidOptionError)
    assert "CONTINUOUS" in str(err)
    assert "bad" in str(err)


@pytest.mark.parametrize("exc", [EBMGraphSimplificationError, EBMGraphTokenLimitError])
def test_graph_error_hierarchy(exc: type[Exception]) -> None:
    assert issubclass(exc, EBMGraphError)
    assert issubclass(EBMGraphError, Exception)
