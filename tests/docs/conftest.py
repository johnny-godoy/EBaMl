"""Run the code examples of the documentation as tests, with Sybil.

The examples are in two places, and the site shows both. The ``Examples`` sections of the docstrings in ``src/ebaml``
are doctests, and the API reference page shows them. The ``python`` code blocks of ``README.md`` and ``docs/*.md`` run
in order, and the code blocks of one file share their variables. The root ``conftest.py`` gives the collection hook of
this file to pytest, because a collection hook works only for the files below its ``conftest.py``.

The Markdown examples use ``X`` and ``y`` with no definition, and they call the LLM. ``setup`` gives each Markdown file
a small binary data set, and it replaces the LLM calls of ``baml_sdk`` with fakes that send no request. ``fake_sdk``
checks each fake against the signature that ``baml generate`` wrote. An example therefore fails when it calls a
function with arguments that the real function does not accept.

Run the examples with ``uv run pytest README.md docs src --no-cov``. Put setup that the reader does not need in an
``<!-- invisible-code-block: python -->`` comment, and skip a block with ``<!-- skip: next -- <reason> -->``.
"""

import doctest
from collections.abc import Callable, MutableMapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser
from sybil.parsers.rest import DocTestParser

import baml_sdk
from tests.unit.ebaml.conftest import fake_sdk

ROOT: Path = Path(__file__).resolve().parents[2]
# the doctests compare the text of the output, so a reflowed long repr must still match
OPTIONS: int = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
FAKE_TEXT = "A fake reply, which stands in for the text of the LLM."
USAGE: baml_sdk.TokenUsage = baml_sdk.TokenUsage(input_tokens=1, output_tokens=1)
PATCHER = "_offline_llm"


def _counted(*_args: object, **_kwargs: object) -> baml_sdk.Counted:
    return baml_sdk.Counted(text=FAKE_TEXT, usage=USAGE)


def _local_text(*_args: object, **_kwargs: object) -> baml_sdk.LocalExplanationText:
    return baml_sdk.LocalExplanationText(
        counterfactual=FAKE_TEXT,
        sufficient_reason=FAKE_TEXT,
        completions=FAKE_TEXT,
        usage=USAGE,
    )


# the BAML functions that send a request to the LLM. The other functions of baml_sdk run locally, so they stay real.
LLM_FAKES = {"describe_graph_cot": _counted, "summarize_ebm": _counted, "explain_local": _local_text}


def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Make the data that the Markdown examples call ``X`` and ``y``.

    Three continuous features with few values and a binary target keep the fit to a few seconds. ``explain_instance``
    supports a binary classifier only.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The features and the target.

    """
    rng = np.random.default_rng(0)
    # few distinct values give short graphs, so the graph text stays inside the token budget
    features = pd.DataFrame({
        "age": rng.integers(1, 9, 300) * 10.0,
        "fare": rng.integers(1, 11, 300) * 50.0,
        "siblings": rng.integers(0, 5, 300).astype(float),
    })
    logits = 0.05 * (40 - features["age"]) + 0.004 * features["fare"] - 0.3 * features["siblings"]
    target = pd.Series((logits + rng.normal(0, 0.5, 300) > 0).astype(int), name="survived")
    return features, target


def setup(namespace: MutableMapping[str, object]) -> None:
    """Prepare one Markdown file: give it ``X`` and ``y``, and replace the LLM calls with fakes.

    Parameters
    ----------
    namespace: MutableMapping[str, object]
        The variables that the code blocks of the file share.

    """
    patcher = pytest.MonkeyPatch()
    for name, fake in LLM_FAKES.items():
        patcher.setattr(baml_sdk, name, fake_sdk(name, fake))
    namespace[PATCHER] = patcher
    namespace["X"], namespace["y"] = training_data()


def teardown(namespace: MutableMapping[str, object]) -> None:
    """Put the real LLM calls back after one Markdown file.

    Parameters
    ----------
    namespace: MutableMapping[str, object]
        The variables that the code blocks of the file share.

    """
    patcher = namespace[PATCHER]
    assert isinstance(patcher, pytest.MonkeyPatch)
    patcher.undo()


MARKDOWN: Sybil = Sybil(
    parsers=[PythonCodeBlockParser(doctest_optionflags=OPTIONS), SkipParser()],
    path=str(ROOT),
    patterns=["README.md", "docs/*.md"],
    setup=setup,
    teardown=teardown,
    name="markdown",
)
DOCSTRINGS: Sybil = Sybil(
    parsers=[DocTestParser(optionflags=OPTIONS)],
    path=str(ROOT),
    # "**/" needs one directory at least, so the modules at the top of the package need a pattern of their own
    patterns=["src/ebaml/*.py", "src/ebaml/**/*.py"],
    name="docstrings",
)

pytest_collect_file: Callable[[Path, pytest.Collector], pytest.Collector | None] = (MARKDOWN + DOCSTRINGS).pytest()
