"""The names that ``ebaml`` exports, and the signature of each, against a snapshot that the maintainer reviews.

A change to the public API changes ``public_api.json``. The reviewer sees the change in the diff of that file, and
``scripts/release/api_breaks.py`` asks for a ``**BREAKING:**`` entry in the changelog when the change breaks a user.
Update the snapshot with ``EBAML_UPDATE_SNAPSHOTS=1 uv run pytest tests/api --no-cov``.
"""

import enum
import inspect
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import attrs
import orjson
import pytest

import ebaml

SNAPSHOT = Path(__file__).parent / "public_api.json"

# the repr of a default value that is an object or a function holds its address, which changes in each process
_ADDRESS = re.compile(r" at 0x[0-9a-f]+")


def _signature(target: Callable[..., object]) -> str | None:
    try:
        return _ADDRESS.sub("", str(inspect.signature(target)))
    except (TypeError, ValueError):
        return None


def _bases(target: type) -> list[str]:
    return [base.__name__ for base in target.__mro__[1:] if base.__module__.startswith(("ebaml", "builtins"))]


def describe(name: str, target: object) -> dict[str, object]:
    """Describe one public name: its kind, and the facts that a user of the name depends on.

    Parameters
    ----------
    name: str
        The exported name.
    target: object
        The object that the name stands for.

    Returns
    -------
    dict[str, object]
        The kind, and for a class its bases, its fields or members, and for a callable its signature.

    """
    if inspect.isclass(target):
        entry: dict[str, object] = {"kind": "class", "bases": _bases(target)}
        if issubclass(target, enum.Enum):
            entry["kind"] = "enum"
            entry["members"] = [member.name for member in target]
        if attrs.has(target):
            entry["fields"] = [f"{field.name}: {field.type}" for field in attrs.fields(target)]
        if issubclass(target, BaseException):
            entry["kind"] = "exception"
        entry["signature"] = _signature(target)
        return entry
    if callable(target):
        return {"kind": "function", "signature": _signature(target)}
    return {"kind": type(target).__name__, "name": name}


def public_api() -> Mapping[str, dict[str, object]]:
    """Describe every name in ``ebaml.__all__``, sorted by name.

    Returns
    -------
    Mapping[str, dict[str, object]]
        The description of each name.

    """
    return {name: describe(name, getattr(ebaml, name)) for name in sorted(ebaml.__all__)}


def _current() -> bytes:
    return orjson.dumps(public_api(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"


def test_the_public_surface_matches_the_snapshot() -> None:
    if os.environ.get("EBAML_UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_bytes(_current())
    expected = orjson.loads(SNAPSHOT.read_bytes())
    current = orjson.loads(_current())
    assert sorted(current) == sorted(expected), "an exported name was added or removed"
    changed = [name for name in current if current[name] != expected[name]]
    assert not changed, f"the API of {changed} changed: review it, then update the snapshot (see the module docstring)"


def test_all_has_no_duplicate() -> None:
    assert len(set(ebaml.__all__)) == len(ebaml.__all__)


def test_no_exported_name_is_private_except_the_version() -> None:
    assert [name for name in ebaml.__all__ if name.startswith("_")] == ["__version__"]


def test_the_version_follows_semantic_versioning() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?", ebaml.__version__)


def test_every_public_function_and_class_has_a_docstring() -> None:
    missing = [
        name
        for name in ebaml.__all__
        if (inspect.isclass(getattr(ebaml, name)) or inspect.isfunction(getattr(ebaml, name)))
        and not inspect.getdoc(getattr(ebaml, name))
    ]
    assert not missing, f"public names without a docstring: {missing}"


def test_every_public_function_has_a_numpy_style_parameters_section() -> None:
    functions = [
        name
        for name in ebaml.__all__
        if inspect.isfunction(getattr(ebaml, name)) and inspect.signature(getattr(ebaml, name)).parameters
    ]
    missing = [name for name in functions if "Parameters\n" not in (inspect.getdoc(getattr(ebaml, name)) or "")]
    assert not missing, f"public functions with parameters and no Parameters section: {missing}"


def test_every_function_of_the_public_api_is_fully_annotated() -> None:
    bare: list[str] = []
    for name in ebaml.__all__:
        target = getattr(ebaml, name)
        if not inspect.isfunction(target):
            continue
        signature = inspect.signature(target)
        if signature.return_annotation is inspect.Signature.empty or any(
            parameter.annotation is inspect.Parameter.empty for parameter in signature.parameters.values()
        ):
            bare.append(name)
    assert not bare, f"public functions with a missing annotation: {bare}"


def test_the_exported_names_come_from_modules_of_ebaml() -> None:
    foreign = [
        name
        for name in ebaml.__all__
        if (module := getattr(getattr(ebaml, name), "__module__", "ebaml")) and not module.startswith("ebaml")
    ]
    assert not foreign, f"exports that another package defines: {foreign}"


@pytest.mark.parametrize(
    "module",
    ["ebaml.mcp", "ebaml.llm", "ebaml.local", "ebaml.edit", "ebaml.graphs", "ebaml.core"],
)
def test_each_layer_package_has_a_docstring(module: str) -> None:
    assert inspect.getdoc(__import__(module, fromlist=["_"]))
