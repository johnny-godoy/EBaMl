"""The errors of ``ebaml``: the one base class, the built-in classes that a caller may catch, and the export of each."""

import inspect

import pytest

import ebaml
from ebaml.core import exceptions

# Each exception with the built-in class that it also derives from, if any. A new exception needs a row, so that its
# author decides what the callers of the package can catch.
BUILTIN_BASE = {
    "APIKeyRejectedError": None,
    "DataFileError": ValueError,
    "EBMGraphError": None,
    "EBMGraphShapeError": ValueError,
    "EBMGraphSimplificationError": None,
    "EBMGraphTokenLimitError": None,
    "EditError": ValueError,
    "GraphTextError": ValueError,
    "HistoryError": ValueError,
    "InvalidInstanceError": ValueError,
    "InvalidOptionError": ValueError,
    "LLMRequestError": None,
    "MissingAPIKeyError": None,
    "OptionRangeError": ValueError,
    "ProviderUnreachableError": None,
    "RateLimitedError": None,
    "SDKUnavailableError": ImportError,
    "SearchLimitError": None,
    "SelectionError": ValueError,
    "StaleRevisionError": None,
    "TokenFitError": ValueError,
    "UnreviewedEditError": None,
    "UnsupportedModelError": ValueError,
}


def _defined_errors() -> dict[str, type[BaseException]]:
    return {
        name: member
        for name, member in vars(exceptions).items()
        if inspect.isclass(member) and issubclass(member, BaseException) and member.__module__ == exceptions.__name__
    }


def test_the_table_lists_each_exception_of_the_module() -> None:
    assert set(_defined_errors()) - {"EBaMlError"} == set(BUILTIN_BASE)


@pytest.mark.parametrize("name", sorted(BUILTIN_BASE))
def test_each_exception_derives_from_the_base_class_and_from_its_builtin(name: str) -> None:
    error = _defined_errors()[name]
    assert issubclass(error, ebaml.EBaMlError)
    builtin = BUILTIN_BASE[name]
    if builtin is not None:
        assert issubclass(error, builtin)


def test_every_exception_of_the_module_is_exported() -> None:
    assert set(_defined_errors()) <= set(ebaml.__all__)
    for name in _defined_errors():
        assert getattr(ebaml, name) is _defined_errors()[name]


def test_the_base_class_is_the_only_one_that_derives_from_exception_directly() -> None:
    direct = [name for name, error in _defined_errors().items() if error.__bases__ == (Exception,)]
    assert direct == ["EBaMlError"]


def test_each_exception_has_a_docstring() -> None:
    assert [name for name, error in _defined_errors().items() if not inspect.getdoc(error)] == []


def test_the_errors_of_the_provider_share_one_base_class() -> None:
    names = {name for name, error in _defined_errors().items() if issubclass(error, ebaml.LLMRequestError)}
    assert names == {
        "LLMRequestError",
        "MissingAPIKeyError",
        "APIKeyRejectedError",
        "RateLimitedError",
        "ProviderUnreachableError",
    }


def test_an_error_of_the_package_is_caught_as_the_base_class() -> None:
    with pytest.raises(ebaml.EBaMlError):
        ebaml.ModelOptions(provider="openai", model="model", temperature=-1)


def test_a_bad_value_is_also_a_value_error() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ebaml.ModelOptions(provider="openai", model="model", temperature=-1)
