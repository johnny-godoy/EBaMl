"""Fixtures and helpers for the tests of the ebaml package."""

import hashlib
import os
from collections.abc import Callable, MutableMapping, MutableSequence
from pathlib import Path
from typing import Any

import interpret
import joblib
import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports


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
