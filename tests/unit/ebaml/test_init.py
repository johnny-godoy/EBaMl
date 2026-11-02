"""Tests for the package."""

import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the test starts a fresh interpreter with a fixed argv
import sys

import pytest
from beartype.roar import BeartypeCallHintViolation

import ebaml
from ebaml.core.custom_types import coerce_enum
from ebaml.llm import descriptors
from ebaml.local import facts as local


def test_package_docstring() -> None:
    assert ebaml.__doc__


def test_functions_check_their_argument_types() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        coerce_enum("not an enum class", "none")  # type: ignore[arg-type] -- the call must break the hint to prove the check runs


def test_int_satisfies_a_float_hint() -> None:
    assert local.LocalOptions(confidence_level=0).confidence_level == 0


LAZY_NAMES = ("LocalExplanation", "describe_ebm", "describe_graph", "explain_instance")


def test_every_exported_name_resolves() -> None:
    for name in ebaml.__all__:
        assert getattr(ebaml, name) is not None


def test_each_lazy_name_is_the_object_of_the_descriptors() -> None:
    for name in LAZY_NAMES:
        assert getattr(ebaml, name) is getattr(descriptors, name)
        assert name in ebaml.__all__


def test_lazy_name_is_stored_in_the_namespace() -> None:
    ebaml.__dict__.pop("describe_graph", None)
    loaded = ebaml.__getattr__("describe_graph")  # ruff: ignore[unnecessary-dunder-call] -- the hook runs only for a missing name, so only a direct call proves it
    assert ebaml.__dict__["describe_graph"] is loaded


def test_dir_lists_the_lazy_names() -> None:
    assert set(LAZY_NAMES) <= set(dir(ebaml))
    assert set(dir(ebaml)) == set(ebaml.__all__)
    assert ebaml.__dir__() == sorted(ebaml.__all__)  # ruff: ignore[unnecessary-dunder-call] -- the module function has no other caller, and vulture needs one


def test_unknown_name_raises_the_standard_message() -> None:
    with pytest.raises(AttributeError, match="module 'ebaml' has no attribute 'missing_name'"):
        _ = ebaml.missing_name  # type: ignore[attr-defined] -- the access must fail to prove the message


def test_import_loads_neither_the_sdk_nor_the_descriptors() -> None:
    code = "import sys, ebaml; print('baml_sdk' in sys.modules, 'ebaml.llm.descriptors' in sys.modules)"
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv, never untrusted input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["False", "False"]
