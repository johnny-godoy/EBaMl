"""Tests of the tool that explains one prediction."""

import pandas as pd  # ruff: ignore[banned-api] -- the test builds the pandas frame that interpret fits on
import pytest
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.local.audience import DataScientist
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.facts import compute_local_facts
from ebaml.mcp.explain_tools import explain_prediction
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import as_frame, instance_of


def test_instance_and_row_give_the_same_facts(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    store = ModelStore(ebm=ebm, data=data)
    by_row = explain_prediction(store, row=3)
    assert by_row == explain_prediction(store, instance=instance_of(frame, 3))
    assert by_row.facts == compute_local_facts(ebm, instance_of(frame, 3))


def test_texts_follow_the_audience(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    explanation = explain_prediction(ModelStore(ebm=ebm, data=data), row=0, audience="data_scientist")
    assert explanation.texts == facts_to_texts(explanation.facts, DataScientist())
    assert explanation.texts != explain_prediction(ModelStore(ebm=ebm, data=data), row=0).texts


def test_confidence_level_reaches_the_facts(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    explanation = explain_prediction(ModelStore(ebm=ebm, data=data), row=0, confidence_level=0.5)
    assert explanation.facts.confidence_level == pytest.approx(0.5)


@pytest.mark.parametrize("arguments", [{}, {"row": 0, "instance": {"a": 1.0}}])
def test_exactly_one_of_instance_and_row_is_needed(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    arguments: dict[str, object],
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    with pytest.raises(InvalidInstanceError, match="exactly one"):
        explain_prediction(ModelStore(ebm=ebm, data=data), **arguments)  # pyrefly: ignore[bad-argument-type] -- the test passes bad argument combinations on purpose


def test_a_regressor_keeps_the_answer_to_the_average_prediction(regression_ebm: ExplainableBoostingRegressor) -> None:
    explanation = explain_prediction(ModelStore(ebm=regression_ebm), instance={"a": 0.5, "c": "x"})
    assert explanation.facts.prediction.decision in {
        f"above {regression_ebm.intercept_:.4g}",
        f"at most {regression_ebm.intercept_:.4g}",
    }
    assert "predicted_value" in explanation.texts.prediction


def test_a_multiclass_classifier_has_no_local_facts_yet(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(UnsupportedModelError, match="binary classifier or a regressor"):
        explain_prediction(ModelStore(ebm=multiclass_ebm), instance={})
