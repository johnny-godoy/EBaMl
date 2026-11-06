"""Fixtures of the MCP tests."""

from collections.abc import Callable

import pandas as pd  # ruff: ignore[banned-api] -- the fixture builds the pandas frame that interpret fits on
import pytest
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests of the MCP SDK on asyncio only.

    Returns
    -------
    str
        The name of the backend.

    """
    return "asyncio"


@pytest.fixture(scope="module")
def interaction_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier | ExplainableBoostingRegressor],
) -> ExplainableBoostingClassifier:
    """Fit a binary classifier with one interaction term, ``p x q``, and the main effects ``p`` and ``q``.

    Returns
    -------
    ExplainableBoostingClassifier
        The fitted model.

    """
    data = pd.DataFrame({"p": range(300), "q": [value % 7 for value in range(300)]})
    target = ((data["p"] % 5) * (data["q"] % 3) > 2).astype(int)
    ebm = fit_ebm(data, target, interactions=1, random_state=0)
    assert isinstance(ebm, ExplainableBoostingClassifier)
    return ebm
