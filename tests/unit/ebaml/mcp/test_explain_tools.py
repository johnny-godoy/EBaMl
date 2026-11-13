"""Tests of the tool that explains one prediction."""

from typing import Any

import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EditError, InvalidInstanceError
from ebaml.edit.edit_session import RevisionInfo
from ebaml.edit.edits import SetScores
from ebaml.edit.selections import WholeGraph
from ebaml.local.audience import DataScientist
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.facts import compute_local_facts
from ebaml.local.local_kinds import OutputKind
from ebaml.mcp.explain_tools import PredictionExplanation, PredictionRequest, explain_prediction
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import FittedClassifier, as_frame, instance_of


def _explain(store: ModelStore, **fields: Any) -> PredictionExplanation:  # ruff: ignore[any-type] -- the tests pass the fields of PredictionRequest, also bad ones
    return explain_prediction(store, PredictionRequest(**fields))


def test_instance_and_row_give_the_same_facts(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    store = ModelStore(ebm=ebm, data=data)
    by_row = _explain(store, row=3)
    assert by_row == _explain(store, instance=instance_of(frame, 3))
    assert by_row.facts == compute_local_facts(ebm, instance_of(frame, 3))


def test_texts_follow_the_audience(continuous_ebm: FittedClassifier) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    explanation = _explain(ModelStore(ebm=ebm, data=data), row=0, audience="data_scientist")
    assert explanation.texts == facts_to_texts(explanation.facts, DataScientist())
    assert explanation.texts != _explain(ModelStore(ebm=ebm, data=data), row=0).texts


def test_confidence_level_reaches_the_facts(continuous_ebm: FittedClassifier) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    explanation = _explain(ModelStore(ebm=ebm, data=data), row=0, confidence_level=0.5)
    assert explanation.facts.confidence_level == pytest.approx(0.5)


@pytest.mark.parametrize("arguments", [{}, {"row": 0, "instance": {"a": 1.0}}])
def test_exactly_one_of_instance_and_row_is_needed(
    continuous_ebm: FittedClassifier,
    arguments: dict[str, object],
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    with pytest.raises(InvalidInstanceError, match="exactly one"):
        _explain(ModelStore(ebm=ebm, data=data), **arguments)


def test_a_regressor_keeps_the_answer_to_the_average_prediction(regression_ebm: ExplainableBoostingRegressor) -> None:
    explanation = _explain(ModelStore(ebm=regression_ebm), instance={"a": 0.5, "c": "x"})
    assert explanation.facts.prediction.decision in {
        f"above {regression_ebm.intercept_:.4g}",
        f"at most {regression_ebm.intercept_:.4g}",
    }
    assert "predicted_value" in explanation.texts.prediction


def test_a_multiclass_classifier_has_a_predicted_class_and_a_class_score(
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    explanation = _explain(
        ModelStore(ebm=multiclass_ebm),
        instance={"a": 2.0, "c": "x"},
        audience="data_scientist",
    )
    assert explanation.facts.prediction.decision == "high"
    assert explanation.facts.prediction.output_kind is OutputKind.CLASS_PROBABILITY
    assert "class_score" in explanation.texts.prediction
    assert "predicted_class" in explanation.texts.prediction


def test_the_explanation_uses_the_edited_copy_and_a_revision_explains_an_earlier_model(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, frame = continuous_ebm
    store = ModelStore(ebm=ebm)
    instance = instance_of(frame, 0)
    original = _explain(store, instance=instance)
    store.editing().apply(SetScores("a", WholeGraph(), 9.0), "Test the effect of a")
    head = store.editing().revision
    edited = _explain(store, instance=instance)
    assert edited.facts != original.facts
    assert edited.session == RevisionInfo(head, head, 1)
    earlier = _explain(store, revision="original", instance=instance)
    assert earlier.facts == original.facts
    assert earlier.session == RevisionInfo("original", head, 1)
    assert _explain(store, revision=head, instance=instance) == edited
    with pytest.raises(EditError, match="no revision 'abc'"):
        _explain(store, revision="abc", instance=instance)
