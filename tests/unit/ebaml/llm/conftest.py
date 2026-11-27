"""Fixtures for the tests of ebaml.llm: the sync and the async form of each function that calls an LLM."""

import asyncio
from collections.abc import Callable
from typing import Any

import attrs
import pytest

import baml_sdk
from ebaml.llm import async_descriptors, descriptors
from ebaml.llm.difficulties import DifficultyProposal
from ebaml.llm.ebm_description import EBMDescription
from ebaml.llm.pipelines import LocalExplanation
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from tests.unit.ebaml.conftest import fake_sdk


@attrs.frozen
class Form:
    """The four functions that call an LLM, in one form. Each one takes its request and returns its result."""

    name: str
    describe_graph: Callable[[DescribeGraphRequest], str]
    describe_ebm: Callable[[DescribeEBMRequest], EBMDescription]
    explain_instance: Callable[[ExplainInstanceRequest], LocalExplanation]
    propose_difficulties: Callable[[ProposeDifficultiesRequest], DifficultyProposal]


SYNC: Form = Form(
    "sync",
    descriptors.describe_graph,
    descriptors.describe_ebm,
    descriptors.explain_instance,
    descriptors.propose_difficulties,
)
ASYNC: Form = Form(
    "async",
    lambda request: asyncio.run(async_descriptors.describe_graph_async(request)),
    lambda request: asyncio.run(async_descriptors.describe_ebm_async(request)),
    lambda request: asyncio.run(async_descriptors.explain_instance_async(request)),
    lambda request: asyncio.run(async_descriptors.propose_difficulties_async(request)),
)
FORMS: tuple[Form, Form] = (SYNC, ASYNC)


@pytest.fixture(params=FORMS, ids=[form.name for form in FORMS])
def form(request: pytest.FixtureRequest) -> Form:
    """Give the test each form in turn.

    Returns
    -------
    Form
        The sync form, then the async form.

    """
    chosen: Form = request.param
    return chosen


def fake_both_forms(monkeypatch: pytest.MonkeyPatch, name: str, fake: Callable[..., Any]) -> None:
    """Replace the sync function ``<name>_sync`` and the async function ``<name>`` of ``baml_sdk`` with one fake.

    ``fake_sdk`` checks each call against the generated signature, and gives the async function a coroutine.

    """
    monkeypatch.setattr(baml_sdk, f"{name}_sync", fake_sdk(f"{name}_sync", fake))
    monkeypatch.setattr(baml_sdk, name, fake_sdk(name, fake))
