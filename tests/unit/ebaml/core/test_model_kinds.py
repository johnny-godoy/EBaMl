"""Tests for the model kinds."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from interpret.glassbox import (
    ExplainableBoostingClassifier,  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
    ExplainableBoostingRegressor,  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
)

import ebaml
from ebaml.core.exceptions import UnsupportedModelError
from ebaml.core.model_kinds import BinaryClassifier, MulticlassClassifier, Regressor, model_kind
from tests.unit.ebaml.conftest import fake_model


def test_binary_kind_of_a_fitted_classifier(continuous_ebm: tuple[ExplainableBoostingClassifier, object]) -> None:
    assert model_kind(continuous_ebm[0]) == BinaryClassifier((0, 1))


def test_regressor_kind_of_a_fitted_regressor(regression_ebm: ExplainableBoostingRegressor) -> None:
    assert model_kind(regression_ebm) == Regressor()


def test_multiclass_kind_of_a_fitted_classifier(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    assert model_kind(multiclass_ebm) == MulticlassClassifier(("high", "low", "mid"))


def test_kinds_are_public() -> None:
    assert ebaml.model_kind is model_kind
    assert {"BinaryClassifier", "ModelKind", "MulticlassClassifier", "Regressor", "model_kind"} <= set(ebaml.__all__)


def test_binary_outputs_name_the_positive_class() -> None:
    kind = BinaryClassifier(("no", "yes"))
    assert kind.outputs == ["yes"]
    assert kind.score_meaning == "A score is the change of the log-odds of the class 'yes'."


def test_binary_predicts_the_sigmoid() -> None:
    assert BinaryClassifier((0, 1)).predict([0.0, np.log(3)]) == pytest.approx([0.5, 0.75])


def test_multiclass_outputs_name_every_class() -> None:
    kind = MulticlassClassifier((1, 2, 3))
    assert kind.outputs == ["1", "2", "3"]
    assert "centred over the classes" in kind.score_meaning


def test_multiclass_predicts_the_softmax_of_the_last_axis() -> None:
    probabilities = MulticlassClassifier(("a", "b", "c")).predict([[0.0, np.log(2), 0.0], [0.0, 0.0, 0.0]])
    assert probabilities == pytest.approx(np.array([[0.25, 0.5, 0.25], [1 / 3, 1 / 3, 1 / 3]]))


@given(
    scores=arrays(np.float64, (4, 3), elements=st.floats(-20, 20)),
    shift=arrays(np.float64, (4, 1), elements=st.floats(-20, 20)),
)
def test_softmax_ignores_a_constant_over_the_classes(scores: np.ndarray, shift: np.ndarray) -> None:
    kind = MulticlassClassifier(("a", "b", "c"))
    assert kind.predict(scores + shift) == pytest.approx(kind.predict(scores), abs=1e-9)


def test_regressor_predicts_the_scores() -> None:
    kind = Regressor()
    assert kind.outputs == ["prediction"]
    assert kind.score_meaning == "A score is the change of the predicted value, in the units of the target."
    assert kind.predict([1.5, -2.0]).tolist() == [1.5, -2.0]


@pytest.mark.parametrize("link", [None, "log"])
def test_unknown_link_is_unsupported(link: str | None) -> None:
    model = fake_model() if link is None else fake_model(link_=link)
    with pytest.raises(UnsupportedModelError, match=r"EBaMl reads the links \['identity', 'logit', 'mlogit'\]"):
        model_kind(model)


def test_classification_link_without_classes_is_unsupported() -> None:
    with pytest.raises(UnsupportedModelError, match="no classes_"):
        model_kind(fake_model(link_="logit"))


def test_logit_link_needs_two_classes() -> None:
    with pytest.raises(UnsupportedModelError, match=r"logit link needs two classes\. Got 3"):
        model_kind(fake_model(link_="logit", classes_=np.array([0, 1, 2])))


@pytest.mark.parametrize("n_classes", [1, 2])
def test_mlogit_link_needs_three_classes(n_classes: int) -> None:
    with pytest.raises(UnsupportedModelError, match=rf"three or more classes\. Got {n_classes}"):
        model_kind(fake_model(link_="mlogit", classes_=np.arange(n_classes)))


def test_class_labels_are_python_values() -> None:
    kind = model_kind(fake_model(link_="logit", classes_=np.array([0, 1])))
    assert isinstance(kind, BinaryClassifier)
    assert all(type(label) is int for label in kind.classes)
