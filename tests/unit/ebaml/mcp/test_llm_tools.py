"""Tests of the tools that ask the LLM, with the calls into ``ebaml.llm`` faked."""

import pytest

from ebaml.edit.edits import SetScores
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins
from ebaml.llm.descriptors import EBMDescription
from ebaml.llm.options import ModelOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest
from ebaml.mcp import llm_tools
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import FittedClassifier

LLM: ModelOptions = ModelOptions(provider="ollama", model="llama3")


def test_describe_feature_passes_the_feature_index_and_the_llm(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    calls: list[tuple[int, ModelOptions]] = []

    def fake_describe_graph(request: DescribeGraphRequest) -> str:
        assert isinstance(request.model, ModelOptions)
        assert request.ebm.term_scores_[0].tolist() == ebm.term_scores_[0].tolist()
        calls.append((request.feature_index, request.model))
        return "graph text"

    monkeypatch.setattr(llm_tools, "describe_graph", fake_describe_graph)
    assert llm_tools.describe_feature(ModelStore(ebm=ebm), LLM, "b") == "graph text"
    assert calls == [(1, LLM)]


def test_describe_model_computes_the_text_once(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    calls: list[ModelOptions] = []

    def fake_describe_ebm(request: DescribeEBMRequest) -> EBMDescription:
        assert request.ebm.term_scores_[0].tolist() == ebm.term_scores_[0].tolist()
        assert isinstance(request.model, ModelOptions)
        calls.append(request.model)
        return EBMDescription("model text", [], [])

    monkeypatch.setattr(llm_tools, "describe_ebm", fake_describe_ebm)
    store = ModelStore(ebm=ebm)
    assert llm_tools.describe_model(store, LLM) == "model text"
    assert llm_tools.describe_model(store, LLM) == "model text"
    assert calls == [LLM]
    assert store.described("original") == "model text"


def test_describe_model_never_returns_the_description_of_the_model_before_an_edit(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    scores_seen: list[float] = []

    def fake_describe_ebm(request: DescribeEBMRequest) -> EBMDescription:
        scores_seen.append(float(term_view(request.ebm, "a").scores[1]))
        return EBMDescription(f"description {len(scores_seen)}", [], [])

    monkeypatch.setattr(llm_tools, "describe_ebm", fake_describe_ebm)
    assert llm_tools.describe_model(store, LLM) == "description 1"
    store.editing().apply(SetScores("a", Bins(0, 1), 5.0), "Raise the lowest bins")
    assert llm_tools.describe_model(store, LLM) == "description 2"
    assert llm_tools.describe_model(store, LLM) == "description 2"
    assert scores_seen[1] == pytest.approx(5.0)
    assert scores_seen[0] != pytest.approx(5.0)


def test_an_edit_during_the_llm_call_leaves_no_stale_description(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LLM call holds no lock, so an edit can land during it. The text then describes a model that is gone."""
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    calls: list[str] = []

    def fake_describe_ebm(_request: DescribeEBMRequest) -> EBMDescription:
        calls.append("call")
        if len(calls) == 1:
            store.editing().apply(SetScores("a", Bins(0, 1), 5.0), "An edit while the LLM works")
        return EBMDescription(f"description {len(calls)}", [], [])

    monkeypatch.setattr(llm_tools, "describe_ebm", fake_describe_ebm)
    assert llm_tools.describe_model(store, LLM) == "description 1"
    assert store.described("original") is None
    assert store.described(store.editing().revision) is None
    assert llm_tools.describe_model(store, LLM) == "description 2"
    assert llm_tools.describe_model(store, LLM) == "description 2"
    assert len(calls) == 2
