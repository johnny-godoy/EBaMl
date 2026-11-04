"""Fixtures of the MCP tests."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests of the MCP SDK on asyncio only.

    Returns
    -------
    str
        The name of the backend.

    """
    return "asyncio"
