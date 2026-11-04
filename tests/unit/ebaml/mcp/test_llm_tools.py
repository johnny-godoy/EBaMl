"""Tests of the tools that ask the LLM, with the calls into ``ebaml.llm`` faked."""

import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.llm.options import ModelOptions
from ebaml.mcp import llm_tools
from ebaml.mcp.model_store import ModelStore

LLM: ModelOptions = ModelOptions(provider="ollama", model="llama3")


def test_describe_feature_passes_the_feature_index_and_the_llm(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    calls: list[tuple[ExplainableBoostingModel, int, ModelOptions]] = []

    def fake_describe_graph(fitted: ExplainableBoostingModel, feature_index: int, *, model: ModelOptions) -> str:
        calls.append((fitted, feature_index, model))
        return "graph text"

    monkeypatch.setattr(llm_tools, "describe_graph", fake_describe_graph)
    assert llm_tools.describe_feature(ModelStore(ebm=ebm), LLM, "b") == "graph text"
    assert calls == [(ebm, 1, LLM)]


def test_describe_model_computes_the_text_once(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ebm, _ = continuous_ebm
    calls: list[ModelOptions] = []

    def fake_describe_ebm(fitted: ExplainableBoostingModel, *, model: ModelOptions) -> str:
        assert fitted is ebm
        calls.append(model)
        return "model text"

    monkeypatch.setattr(llm_tools, "describe_ebm", fake_describe_ebm)
    store = ModelStore(ebm=ebm)
    assert llm_tools.describe_model(store, LLM) == "model text"
    assert llm_tools.describe_model(store, LLM) == "model text"
    assert calls == [LLM]
    assert store.model_description == "model text"
