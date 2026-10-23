"""Fixtures and helpers for the tests of the ebaml package."""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import interpret
import joblib
import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier


@pytest.fixture(scope="session")
def fit_ebm() -> Callable[..., ExplainableBoostingClassifier]:
    """Return a function that fits an EBM, or reads the same fit from the cache directory.

    ``scripts/mutation.py`` sets ``EBAML_FIT_CACHE`` to a directory. Every mutant runs the tests in a new process, and
    a fit takes seconds, so the processes share the fitted models through that directory. Without the variable, the
    function fits the model every time. A mutant changes ``ebaml`` only, so a fit does not depend on the mutant.

    Returns
    -------
    Callable[..., ExplainableBoostingClassifier]
        The function. It takes the data, the target and the keyword arguments of the classifier.

    """
    cache = os.environ.get("EBAML_FIT_CACHE")

    def fit(data: pd.DataFrame, target: object, **params: object) -> ExplainableBoostingClassifier:
        ebm = ExplainableBoostingClassifier(**params)
        if not cache:
            return ebm.fit(data, target)
        digest = hashlib.sha256(interpret.__version__.encode())
        for part in (repr(sorted(params.items())), repr(list(data.columns)), repr(list(data.dtypes))):
            digest.update(part.encode())
        digest.update(pd.util.hash_pandas_object(data).to_numpy().tobytes())
        digest.update(np.asarray(target).tobytes())
        path = Path(cache) / f"{digest.hexdigest()}.joblib"
        if path.exists():
            return joblib.load(path)
        ebm.fit(data, target)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        joblib.dump(ebm, partial)
        partial.replace(path)  # one step, so another process never reads a half-written file
        return ebm

    return fit


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
