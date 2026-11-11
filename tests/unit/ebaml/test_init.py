"""Tests for the package."""

import sys

import pytest
from beartype.roar import BeartypeCallHintViolation
from scripts.fixed_command import run_fixed

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


LAZY_NAMES = ("EBMDescription", "LocalExplanation", "describe_ebm", "describe_graph", "explain_instance")


def test_every_exported_name_resolves() -> None:
    for name in ebaml.__all__:
        assert getattr(ebaml, name) is not None


def test_each_lazy_name_is_the_object_of_the_descriptors() -> None:
    for name in LAZY_NAMES:
        assert getattr(ebaml, name) is getattr(descriptors, name)
        assert name in ebaml.__all__


def test_lazy_name_is_stored_in_the_namespace() -> None:
    ebaml.__dict__.pop("describe_graph", None)
    loaded = ebaml.describe_graph  # the name is missing from the namespace, so the module hook loads it
    assert ebaml.__dict__["describe_graph"] is loaded


def test_dir_lists_the_lazy_names() -> None:
    assert set(LAZY_NAMES) <= set(dir(ebaml))
    assert set(dir(ebaml)) == set(ebaml.__all__)
    assert dir(ebaml) == sorted(ebaml.__all__)


def test_unknown_name_raises_the_standard_message() -> None:
    with pytest.raises(AttributeError, match="module 'ebaml' has no attribute 'missing_name'"):
        _ = ebaml.missing_name  # type: ignore[attr-defined] -- the access must fail to prove the message


def test_import_loads_neither_the_sdk_nor_the_descriptors() -> None:
    code = "import sys, ebaml; print('baml_sdk' in sys.modules, 'ebaml.llm.descriptors' in sys.modules)"
    result = run_fixed([sys.executable, "-c", code], check=True)
    assert result.stdout.split() == ["False", "False"]


def test_import_loads_ortools_and_not_highspy() -> None:
    # highspy and ortools each ship a HiGHS library, and a process that loads both fails with an undefined symbol.
    # ebaml reaches HiGHS through the MathOpt API of ortools only.
    code = "import sys, ebaml; print('ortools.math_opt.python.mathopt' in sys.modules, 'highspy' in sys.modules)"
    result = run_fixed([sys.executable, "-c", code], check=True)
    assert result.stdout.split() == ["True", "False"]
