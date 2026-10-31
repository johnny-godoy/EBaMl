"""Tests for the package."""

import subprocess  # ruff: ignore[suspicious-subprocess-import] -- only a new interpreter shows what a plain import loads
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


def test_every_public_name_is_an_attribute_of_the_package() -> None:
    assert [name for name in ebaml.__all__ if not hasattr(ebaml, name)] == []


@pytest.mark.parametrize("name", ["LocalExplanation", "describe_ebm", "describe_graph", "explain_instance"])
def test_a_name_of_the_descriptors_loads_on_first_use(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.delitem(vars(ebaml), name, raising=False)
    assert getattr(ebaml, name) is getattr(descriptors, name)
    # the first use stores the name, so the next use does not call __getattr__
    assert vars(ebaml)[name] is getattr(descriptors, name)


def test_getattr_returns_the_name_of_the_descriptors() -> None:
    # the attribute access calls the hook only once, so the test calls it itself
    assert ebaml.__getattr__("describe_graph") is descriptors.describe_graph  # ruff: ignore[unnecessary-dunder-call] -- the test checks the module hook itself


def test_an_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match=r"^module 'ebaml' has no attribute 'missing'$"):
        _ = ebaml.missing  # type: ignore[attr-defined] -- the access must fail to prove the error


def test_import_of_the_package_does_not_load_the_sdk() -> None:
    # the descriptors import baml_sdk and can run `baml generate`, so a plain import must not load them
    code = "import sys, ebaml; print('baml_sdk' in sys.modules, 'ebaml.llm.descriptors' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv, never untrusted input
    assert result.stdout.split() == ["False", "False"]
