"""Tests of the tools that ask the LLM, with the calls into ``ebaml.llm`` faked."""

import asyncio

import pytest

from ebaml.core.custom_types import GraphMode, VisionFallback
from ebaml.core.exceptions import RateLimitedError
from ebaml.edit.edits import SetScores
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins
from ebaml.llm.difficulties import DifficultyProposal, FeatureDifficulty
from ebaml.llm.options import GraphOptions, ModelOptions
from ebaml.llm.pipelines import EBMDescription, FeatureFailure
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ProposeDifficultiesRequest
from ebaml.local.costs import DataCost, Difficulty, ScaledCost
from ebaml.mcp import llm_tools
from ebaml.mcp.explain_tools import CostStrategyName
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import FittedClassifier, as_frame, awaiting, instance_of

LLM: ModelOptions = ModelOptions(provider="ollama", model="llama3")
TEXT: GraphOptions = GraphOptions()


def test_describe_feature_passes_the_graph_options(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[GraphOptions] = []

    def fake_describe_graph(request: DescribeGraphRequest) -> str:
        seen.append(request.graph)
        return "graph text"

    monkeypatch.setattr(llm_tools, "describe_graph_async", awaiting(fake_describe_graph))
    vision = GraphOptions(mode=GraphMode.VISION, vision_fallback=VisionFallback.TEXT)
    asyncio.run(llm_tools.describe_feature(ModelStore(ebm=continuous_ebm[0]), LLM, "b", vision))
    assert seen == [vision]


def test_describe_model_keeps_one_text_for_each_mode(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[GraphMode] = []

    def fake_describe_ebm(request: DescribeEBMRequest) -> EBMDescription:
        seen.append(request.graph.mode)
        return EBMDescription(f"described in {request.graph.mode}", [], [])

    monkeypatch.setattr(llm_tools, "describe_ebm_async", awaiting(fake_describe_ebm))
    store = ModelStore(ebm=continuous_ebm[0])
    vision = GraphOptions(mode=GraphMode.VISION)
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "described in text"
    assert asyncio.run(llm_tools.describe_model(store, LLM, vision)) == "described in vision"
    assert asyncio.run(llm_tools.describe_model(store, LLM, vision)) == "described in vision"
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "described in text"
    assert seen == [GraphMode.TEXT, GraphMode.VISION]


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

    monkeypatch.setattr(llm_tools, "describe_graph_async", awaiting(fake_describe_graph))
    assert asyncio.run(llm_tools.describe_feature(ModelStore(ebm=ebm), LLM, "b", TEXT)) == "graph text"
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

    monkeypatch.setattr(llm_tools, "describe_ebm_async", awaiting(fake_describe_ebm))
    store = ModelStore(ebm=ebm)
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "model text"
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "model text"
    assert calls == [LLM]
    assert store.described("original", GraphMode.TEXT) == "model text"


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

    monkeypatch.setattr(llm_tools, "describe_ebm_async", awaiting(fake_describe_ebm))
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 1"
    store.editing().apply(SetScores("a", Bins(0, 1), 5.0), "Raise the lowest bins")
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 2"
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 2"
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

    monkeypatch.setattr(llm_tools, "describe_ebm_async", awaiting(fake_describe_ebm))
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 1"
    assert store.described("original", GraphMode.TEXT) is None
    assert store.described(store.editing().revision, GraphMode.TEXT) is None
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 2"
    assert asyncio.run(llm_tools.describe_model(store, LLM, TEXT)) == "description 2"
    assert len(calls) == 2


def test_propose_feature_difficulties_reads_the_row_and_returns_the_arguments_of_explain_prediction(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, frame = continuous_ebm
    seen: list[ProposeDifficultiesRequest] = []
    item = FeatureDifficulty("a", Difficulty.HARD, 2.0, "years")

    def fake_propose(request: ProposeDifficultiesRequest) -> DifficultyProposal:
        seen.append(request)
        return DifficultyProposal([item], ScaledCost(request.base, {"a": 2.0}, proposed_by_model=True))

    monkeypatch.setattr(llm_tools, "propose_difficulties_async", awaiting(fake_propose))
    store = ModelStore(ebm=ebm, data=as_frame(frame))
    advice = asyncio.run(llm_tools.propose_feature_difficulties(store, LLM, None, 3, CostStrategyName.DATA))
    (request,) = seen
    assert (request.instance, request.model, request.base) == (instance_of(frame, 3), LLM, DataCost())
    assert advice == llm_tools.DifficultyAdvice([item], {"a": 2.0}, CostStrategyName.DATA)
    assert advice.difficulty_from_model


def test_describe_model_names_each_failed_feature_and_keeps_no_partial_text(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_describe_ebm(_request: DescribeEBMRequest) -> EBMDescription:
        calls.append(len(calls))
        failure = FeatureFailure("b", RateLimitedError("The provider limited the rate.\nTo fix it:\n  1. Wait."))
        return EBMDescription("partial text", [], [], failed_features=[failure])

    monkeypatch.setattr(llm_tools, "describe_ebm_async", awaiting(fake_describe_ebm))
    store = ModelStore(ebm=continuous_ebm[0])
    text = asyncio.run(llm_tools.describe_model(store, LLM, TEXT))
    assert text == "partial text\n\nfailed_features: b (The provider limited the rate.)"
    asyncio.run(llm_tools.describe_model(store, LLM, TEXT))
    # the store keeps no partial text, so the second call asks again
    assert calls == [0, 1]
