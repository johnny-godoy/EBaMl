"""Tests of the tool that explains one prediction."""

import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.local.audience import DataScientist
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.facts import compute_local_facts
from ebaml.mcp.explain_tools import explain_prediction
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import instance_of


def test_instance_and_row_give_the_same_facts(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, data = continuous_ebm
    store = ModelStore(ebm=ebm, data=data)
    by_row = explain_prediction(store, row=3)
    assert by_row == explain_prediction(store, instance=instance_of(data, 3))
    assert by_row.facts == compute_local_facts(ebm, instance_of(data, 3))


def test_texts_follow_the_audience(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, data = continuous_ebm
    explanation = explain_prediction(ModelStore(ebm=ebm, data=data), row=0, audience="data_scientist")
    assert explanation.texts == facts_to_texts(explanation.facts, DataScientist())
    assert explanation.texts != explain_prediction(ModelStore(ebm=ebm, data=data), row=0).texts


def test_confidence_level_reaches_the_facts(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, data = continuous_ebm
    explanation = explain_prediction(ModelStore(ebm=ebm, data=data), row=0, confidence_level=0.5)
    assert explanation.facts.confidence_level == pytest.approx(0.5)


@pytest.mark.parametrize("arguments", [{}, {"row": 0, "instance": {"a": 1.0}}])
def test_exactly_one_of_instance_and_row_is_needed(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    arguments: dict[str, object],
) -> None:
    ebm, data = continuous_ebm
    with pytest.raises(InvalidInstanceError, match="exactly one"):
        explain_prediction(ModelStore(ebm=ebm, data=data), **arguments)  # pyrefly: ignore[bad-argument-type] -- the test passes bad argument combinations on purpose


def test_a_regressor_has_no_local_facts_yet(regression_ebm: ExplainableBoostingRegressor) -> None:
    with pytest.raises(UnsupportedModelError, match="binary classifier"):
        explain_prediction(ModelStore(ebm=regression_ebm), instance={"a": 0.5, "c": "x"})
