"""Fixtures of the MCP tests."""

from collections.abc import Callable, Iterator

import pandas as pd  # ruff: ignore[banned-api] -- the fixture builds the pandas frame that interpret fits on
import pytest
import structlog

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor


@pytest.fixture
def reset_structlog() -> Iterator[None]:
    """Undo the logging setup of ``serve`` after a test that asks for this fixture.

    ``serve`` points structlog at ``sys.stderr``. Under pytest that is a capture stream that closes after the test, so
    a later test that logs would write to a closed file.

    Yields
    ------
    None
        Nothing. The test runs.

    """
    yield
    structlog.reset_defaults()


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
