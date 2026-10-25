"""Fixtures and helpers for the tests of the ebaml package."""

import ast
import builtins
import functools
import hashlib
import inspect
import os
import types
import typing
from collections.abc import Callable, MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

import interpret
import joblib
import numpy as np
import pandas as pd
import pytest
from beartype.door import is_bearable
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

import baml_sdk

# `baml generate` writes the signatures of the SDK functions only to this stub. At run time each function takes
# (*args, **kwargs), so inspect.signature cannot see the parameters.
SDK_STUB: Path = Path(baml_sdk.__file__).with_suffix(".pyi")


@pytest.fixture(scope="session")
def fit_ebm() -> Callable[..., ExplainableBoostingClassifier]:
    """Return a function that fits an EBM, or reads the same fit from the cache directory.

    ``scripts/mutation.py`` sets ``EBAML_FIT_CACHE`` to a directory. Every mutant runs the tests in a new process, and
    a fit takes seconds, so the processes share the fitted models through that directory. Without the variable, the
    function fits the model every time. A mutant changes ``ebaml`` only, so a fit does not depend on the mutant.

    Returns
    -------
    Callable[..., ExplainableBoostingClassifier]
        The function. It takes the data, the target, and the ``interactions`` and ``random_state`` of the classifier.

    """
    cache = os.environ.get("EBAML_FIT_CACHE")

    def fit(
        data: pd.DataFrame,
        target: object,
        *,
        interactions: int,
        random_state: int,
    ) -> ExplainableBoostingClassifier:
        params = {"interactions": interactions, "random_state": random_state}
        ebm = ExplainableBoostingClassifier(interactions=interactions, random_state=random_state)
        if not cache:
            return ebm.fit(data, target)
        digest = hashlib.sha256(interpret.__version__.encode())
        for part in (repr(sorted(params.items())), repr(list(data.columns)), repr(list(data.dtypes))):
            digest.update(part.encode())
        digest.update(pd.util.hash_pandas_object(data).to_numpy().tobytes())
        digest.update(np.asarray(target).tobytes())
        path = Path(cache) / f"{digest.hexdigest()}.joblib"
        if path.exists():
            cached = joblib.load(path)
            assert isinstance(cached, ExplainableBoostingClassifier)
            return cached
        ebm.fit(data, target)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        joblib.dump(ebm, partial)
        partial.replace(path)  # one step, so another process never reads a half-written file
        return ebm

    return fit


@pytest.fixture(scope="module")
def continuous_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> tuple[ExplainableBoostingClassifier, pd.DataFrame]:
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600), "c": rng.choice(["x", "y"], 600)})
    noise = rng.normal(size=600)
    target = ((2 * data["a"] + 0.3 * data["b"] + (data["c"] == "x") + noise) > 0).astype(int)
    return fit_ebm(data, target, interactions=0, random_state=0), data


# fake_model builds EBM instances with preset attributes, so tests can pass them where an EBM type is required.
def fake_model(**attributes: Any) -> ExplainableBoostingClassifier:  # ruff: ignore[any-type] -- attribute values are arbitrary test data
    """Build an EBM without fitting it, and set the given attributes on it.

    Returns
    -------
    ExplainableBoostingClassifier
        An instance that passes the runtime type check but has only the given attributes.

    """
    model = ExplainableBoostingClassifier.__new__(ExplainableBoostingClassifier)
    vars(model).update(attributes)
    return model


def instance_of(data: pd.DataFrame, index: int) -> dict[str, object]:
    """Read one row of ``data`` as the instance of a local explanation, with the column names as keys.

    Returns
    -------
    dict[str, object]
        The value of every feature, by feature name.

    """
    return {str(name): value for name, value in data.iloc[index].items()}


def returning[T](value: T) -> Callable[..., T]:
    """Build a fake function that takes any arguments and returns ``value``.

    Returns
    -------
    Callable[..., T]
        The fake.

    """

    def fake(*_args: object, **_kwargs: object) -> T:
        return value

    return fake


def recording[T](calls: MutableMapping[str, Any], value: T) -> Callable[..., T]:
    """Build a fake function that stores its keyword arguments in ``calls`` and returns ``value``.

    Returns
    -------
    Callable[..., T]
        The fake.

    """

    def fake(**kwargs: object) -> T:
        calls.update(kwargs)
        return value

    return fake


def appending[T](calls: MutableSequence[dict[str, Any]], value: T) -> Callable[..., T]:
    """Build a fake function that appends its keyword arguments to ``calls`` and returns ``value``.

    Returns
    -------
    Callable[..., T]
        The fake.

    """

    def fake(**kwargs: object) -> T:
        calls.append(kwargs)
        return value

    return fake


@functools.cache
def _sdk_stub() -> ast.Module:
    return ast.parse(SDK_STUB.read_text(encoding="utf-8"))


@functools.cache
def _sdk_functions() -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in _sdk_stub().body if isinstance(node, ast.FunctionDef)}


class _UnknownNameError(LookupError):
    """The annotation names a type that exists only in the stub, such as ``UNSET``."""


def _lookup(name: str) -> object:
    for namespace in (baml_sdk, builtins):
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise _UnknownNameError(name)


def _subscript(origin: object, item: object) -> object:
    if not hasattr(origin, "__getitem__"):
        raise _UnknownNameError(repr(origin))
    # a special form of typing, such as typing.Union, has no static type that takes arguments in square brackets
    generic: Any = origin
    built: object = generic[item]
    return built


def _build(node: ast.expr) -> object:
    """Build the type that an annotation of the stub names, by name lookup only, so no code of the stub runs.

    Returns
    -------
    object
        The type.

    Raises
    ------
    _UnknownNameError
        If a name is neither in ``baml_sdk``, nor in ``typing``, nor a builtin.

    """
    match node:
        case ast.Constant(value=value):
            built = value
        case ast.Name(id=name):
            built = _lookup(name)
        case ast.Attribute(value=ast.Name(id="typing"), attr=name):
            built = getattr(typing, name)
        case ast.Subscript(value=origin, slice=ast.Tuple(elts=items)):
            built = _subscript(_build(origin), tuple(_build(item) for item in items))
        case ast.Subscript(value=origin, slice=item):
            built = _subscript(_build(origin), _build(item))
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            built = _subscript(typing.Union, (_build(left), _build(right)))
        case _:
            raise _UnknownNameError(ast.unparse(node))
    return built


def _resolve(annotation: ast.expr | None) -> object:
    """Build the type of an annotation of the stub, or keep its text when it names a type that exists only in the stub.

    Returns
    -------
    object
        The type, or the text of the annotation.

    """
    if annotation is None:
        return inspect.Parameter.empty
    try:
        return _build(annotation)
    except _UnknownNameError:
        return ast.unparse(annotation)


def sdk_signature(name: str) -> inspect.Signature:
    """Read the signature of an SDK function from the generated stub.

    Returns
    -------
    inspect.Signature
        The parameters, with their types, and the return type.

    """
    node = _sdk_functions()[name]
    arguments = node.args
    required = len(arguments.args) - len(arguments.defaults)
    parameters = [
        inspect.Parameter(
            arg.arg,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if index < required else None,
            annotation=_resolve(arg.annotation),
        )
        for index, arg in enumerate(arguments.args)
    ]
    parameters += [
        inspect.Parameter(
            arg.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else None,
            annotation=_resolve(arg.annotation),
        )
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    ]
    return inspect.Signature(parameters, return_annotation=_resolve(node.returns))


def fake_sdk[T](name: str, fake: Callable[..., T]) -> Callable[..., T]:
    """Wrap a fake of an SDK function, so that it obeys the signature that ``baml generate`` wrote for the function.

    A named parameter of the fake must be a parameter of the function. On each call, the arguments must bind to the
    parameters and fit their types, and the result must fit the return type. A ``baml generate`` that renames a
    parameter or changes a type then fails the tests that use the fake.

    Returns
    -------
    Callable[..., T]
        The checked fake.

    """
    signature = sdk_signature(name)
    named = {
        parameter.name
        for parameter in inspect.signature(fake).parameters.values()
        if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    unknown = named - set(signature.parameters)
    assert not unknown, f"the fake of {name} has parameters that the SDK function does not have: {sorted(unknown)}"

    def checked(*args: object, **kwargs: object) -> T:
        bound = signature.bind(*args, **kwargs)
        for argument, value in bound.arguments.items():
            hint = signature.parameters[argument].annotation
            assert is_bearable(value, hint), f"{name}: {argument}={value!r} does not fit {hint}"
        result = fake(*args, **kwargs)
        assert is_bearable(result, signature.return_annotation), f"{name} returned {result!r}"
        return result

    return checked


def sdk_fields(name: str) -> dict[str, object]:
    """Read the fields of an SDK class from the generated stub.

    Returns
    -------
    dict[str, object]
        The type of each field, by name.

    """
    (node,) = (node for node in _sdk_stub().body if isinstance(node, ast.ClassDef) and node.name == name)
    return {
        item.target.id: _resolve(item.annotation)
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }


def _atoms(hint: object) -> frozenset[object]:
    origin = typing.get_origin(hint)
    if origin in {typing.Union, types.UnionType}:
        return frozenset().union(*(_atoms(arg) for arg in typing.get_args(hint)))
    if origin is typing.Literal:
        return frozenset(("literal", value) for value in typing.get_args(hint))
    return frozenset({hint})


def same_type(ours: object, theirs: object) -> bool:
    """Tell whether two type hints accept the same values.

    For example, ``Literal["a", "b"]`` and ``Literal["a"] | Literal["b"]`` are the same type.

    Returns
    -------
    bool
        True if both hints accept the same values.

    """
    return _atoms(ours) == _atoms(theirs)


def _protocol_properties(protocol: type) -> dict[str, object]:
    return {
        name: typing.get_type_hints(member.fget)["return"]
        for name, member in vars(protocol).items()
        if isinstance(member, property) and member.fget is not None
    }


def protocol_mismatches(protocol: type, sdk_class: str, nested: typing.Mapping[type, str]) -> list[str]:
    """Compare the properties of a protocol in ``ebaml`` with the fields of the SDK class that it stands for.

    ``nested`` maps a protocol that a property returns to the SDK class of the field, such as ``Usage`` to
    ``TokenUsage``. The SDK class may have more fields than the protocol reads.

    Returns
    -------
    list[str]
        One line for each property that the SDK class lacks or types differently. The list is empty when they match.

    """
    fields = sdk_fields(sdk_class)
    problems: list[str] = []
    for name, hint in _protocol_properties(protocol).items():
        if name not in fields:
            problems.append(f"{sdk_class} has no field {name}")
        elif isinstance(hint, type) and hint in nested:
            problems += protocol_mismatches(hint, nested[hint], nested)
        elif not same_type(hint, fields[name]):
            problems.append(f"{sdk_class}.{name} is {fields[name]}, the protocol reads {hint}")
    return problems


def call_mismatches(protocol: type, sdk_function: str) -> list[str]:
    """Compare the ``__call__`` of a protocol in ``ebaml`` with the parameters of the SDK function that it stands for.

    The protocol must name each required parameter of the function, with the same type, and no other parameter.

    Returns
    -------
    list[str]
        One line for each parameter that differs. The list is empty when they match.

    """
    hints = typing.get_type_hints(protocol.__call__)
    ours = [name for name in inspect.signature(protocol.__call__).parameters if name != "self"]
    theirs = {
        name: parameter
        for name, parameter in sdk_signature(sdk_function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    problems = [f"{sdk_function} has no parameter {name}" for name in ours if name not in theirs]
    problems += [f"the protocol lacks the parameter {name}" for name in theirs if name not in ours]
    problems += [
        f"{sdk_function}({name}) is {theirs[name].annotation}, the protocol passes {hints[name]}"
        for name in ours
        if name in theirs and not same_type(hints[name], theirs[name].annotation)
    ]
    return problems
