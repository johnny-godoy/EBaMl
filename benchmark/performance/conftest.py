"""The options and the fixture of the performance benchmark: the saved models, the fake LLM and the workload filter.

``benchmark/run.py performance`` passes the options. The fixture ``context`` starts the local fake LLM server of the
integration tests once for the run, and removes the keys of the real providers from the environment, so that no round
reaches a real provider.
"""

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import FAKE_KEY, KEY_VARIABLE, PROVIDER_KEYS, serve_fake_llm

from benchmark.performance.models import DEFAULT_MODEL_DIR
from benchmark.performance.workload_types import Context
from ebaml import ModelOptions


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the options of the performance benchmark."""
    group = parser.getgroup("performance", "the performance benchmark of EBaMl")
    group.addoption("--models-dir", type=Path, default=DEFAULT_MODEL_DIR, help="the directory of the saved models")
    group.addoption(
        "--workload",
        action="append",
        default=[],
        help="run only the workloads whose name matches this shell pattern; repeat it for more patterns",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep only the workloads that match a pattern of ``--workload``, if the run names a pattern."""
    patterns: list[str] = config.getoption("workload")
    if not patterns:
        return
    selected = [item for item in items if any(fnmatch.fnmatchcase(_name(item), pattern) for pattern in patterns)]
    deselected = [item for item in items if item not in selected]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def _name(item: pytest.Item) -> str:
    """Read the name of the workload of a test.

    Returns
    -------
    str
        The id of the parameter of the test, which is the name of the workload.

    """
    return item.callspec.id if isinstance(item, pytest.Function) and hasattr(item, "callspec") else item.name


@pytest.fixture(scope="session")
def context(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Context]:
    """Start the fake LLM server, and point a model at it, for every workload of the run.

    Yields
    ------
    Context
        The saved models, the fake LLM, its model and a directory for files.

    """
    saved = {name: os.environ.get(name) for name in (*PROVIDER_KEYS, KEY_VARIABLE)}
    for name in PROVIDER_KEYS:
        os.environ.pop(name, None)
    os.environ[KEY_VARIABLE] = FAKE_KEY
    try:
        with serve_fake_llm() as fake_llm:
            llm = ModelOptions(
                provider="openai-generic",
                model="fake-model",
                api_key_env=KEY_VARIABLE,
                base_url=fake_llm.base_url,
            )
            yield Context(request.config.getoption("models_dir"), fake_llm, llm, tmp_path_factory.mktemp("work"))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
