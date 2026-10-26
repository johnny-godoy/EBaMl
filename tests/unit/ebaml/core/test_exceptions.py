"""Tests for the exceptions."""

import inspect

import pytest

import ebaml
from ebaml.core import exceptions
from ebaml.core.custom_types import FeatureType
from ebaml.core.exceptions import (
    EBaMlError,
    EBMGraphError,
    EBMGraphShapeError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    GraphTextError,
    InvalidInstanceError,
    InvalidOptionError,
    OptionRangeError,
    SDKUnavailableError,
    UnsupportedModelError,
)

# the errors for a bad value; code that caught ValueError before they existed must still catch them
VALUE_ERRORS = [
    EBMGraphShapeError,
    GraphTextError,
    InvalidInstanceError,
    InvalidOptionError,
    OptionRangeError,
    UnsupportedModelError,
]


def test_invalid_option_message() -> None:
    err = InvalidOptionError("x", ["a", "b"])
    assert str(err) == "Invalid option: x. Valid options are ['a', 'b']."
    assert isinstance(err, ValueError)


def test_from_enum_lists_member_names() -> None:
    err = InvalidOptionError.from_enum("bad", FeatureType)
    assert isinstance(err, InvalidOptionError)
    assert "CONTINUOUS" in str(err)
    assert "bad" in str(err)


@pytest.mark.parametrize("exc", [EBMGraphSimplificationError, EBMGraphTokenLimitError, EBMGraphShapeError])
def test_graph_error_hierarchy(exc: type[Exception]) -> None:
    assert issubclass(exc, EBMGraphError)
    assert issubclass(EBMGraphError, Exception)


def test_every_error_of_the_module_derives_from_the_base_class() -> None:
    errors = [value for _, value in inspect.getmembers(exceptions, inspect.isclass) if issubclass(value, Exception)]
    assert len(errors) == 11
    assert all(issubclass(error, EBaMlError) for error in errors)


@pytest.mark.parametrize("exc", VALUE_ERRORS)
def test_an_error_for_a_bad_value_is_a_value_error(exc: type[Exception]) -> None:
    assert issubclass(exc, ValueError)


@pytest.mark.parametrize("exc", [EBaMlError, EBMGraphError, EBMGraphSimplificationError, EBMGraphTokenLimitError])
def test_an_error_that_was_not_a_value_error_stays_so(exc: type[Exception]) -> None:
    assert not issubclass(exc, ValueError)


def test_sdk_unavailable_error_is_an_import_error() -> None:
    assert issubclass(SDKUnavailableError, ImportError)
    assert not issubclass(SDKUnavailableError, ValueError)


def test_the_package_exports_every_error() -> None:
    errors = {name for name, value in inspect.getmembers(exceptions, inspect.isclass) if issubclass(value, Exception)}
    assert errors <= set(ebaml.__all__)
