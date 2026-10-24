"""Tests for the package."""

import pytest
from beartype.roar import BeartypeCallHintViolation

import ebaml
from ebaml.core.custom_types import coerce_enum
from ebaml.local import facts as local


def test_package_docstring() -> None:
    assert ebaml.__doc__


def test_functions_check_their_argument_types() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        coerce_enum("not an enum class", "none")  # type: ignore[arg-type] -- the call must break the hint to prove the check runs


def test_int_satisfies_a_float_hint() -> None:
    assert local.LocalOptions(confidence_level=0).confidence_level == 0
