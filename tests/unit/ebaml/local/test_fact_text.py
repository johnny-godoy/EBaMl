"""Tests for the texts of the local facts."""

from typing import override

import attrs
import orjson
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.custom_types import JSONValue
from ebaml.local import fact_text as local_text
from ebaml.local import facts as local
from ebaml.local.audience import DataScientist, Layperson
from tests.unit.ebaml.conftest import instance_of


def test_the_default_audience_is_a_layperson(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, data = continuous_ebm
    facts = local.compute_local_facts(ebm, instance_of(data, 0))
    assert local_text.facts_to_texts(facts) == local_text.facts_to_texts(facts, Layperson())


def test_texts_hide_numbers_from_layperson(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, data = continuous_ebm
    facts = local.compute_local_facts(ebm, instance_of(data, 0))
    layperson = local_text.facts_to_texts(facts)
    expert = local_text.facts_to_texts(facts, DataScientist())
    for text in attrs.astuple(layperson):
        for key in ("score", "std", "probability", "logit", "confidence"):
            assert key not in text
    assert "score" in expert.prediction
    assert "flip_holds_at_confidence" in expert.counterfactual
    assert "confidence_interval" in expert.prediction
    assert orjson.loads(expert.completions)["steps"][0]["share_with_same_prediction"] <= 1


def test_texts_without_counterfactual_or_sufficient_reason() -> None:
    prediction = local.Prediction(decision=0, output=0.7, lower=0.6, upper=0.8, score=-0.8, score_std=0.1, margin=0.8)
    facts = local.LocalFacts(prediction, [], None, None, [], 0.95)
    texts = local_text.facts_to_texts(facts, DataScientist())
    assert orjson.loads(texts.counterfactual) == {"flip_possible": False}
    assert orjson.loads(texts.sufficient_reason) == {"exists": False}


def _text_facts(logit: float, scores: list[float], confidence_level: float = 0.95) -> local.LocalFacts:
    contributions = [local.Contribution(f"f{i}", str(i), score, 0.1) for i, score in enumerate(scores)]
    prediction = local.Prediction(
        decision="y",
        output=0.7,
        lower=0.6,
        upper=0.8,
        score=logit,
        score_std=0.1,
        margin=abs(logit),
    )
    return local.LocalFacts(prediction, contributions, None, None, [], confidence_level)


@pytest.mark.parametrize(
    ("logit", "scores", "effects"),
    [
        (1.4, [0.6, -0.3, 0.0], ["strongly towards", "moderately against", "slightly against"]),
        (-0.8, [-0.6, 0.3, 0.0], ["strongly towards", "moderately against", "slightly against"]),
        (0.0, [-0.3], ["moderately towards"]),  # a logit of 0 belongs to the negative class
        (0.5, [0.3, -0.3], ["moderately towards", "moderately against"]),
    ],
)
def test_layperson_effects_point_towards_or_against_the_prediction(
    logit: float,
    scores: list[float],
    effects: list[str],
) -> None:
    texts = local_text.facts_to_texts(_text_facts(logit, scores))
    found = [entry["effect"] for entry in orjson.loads(texts.prediction)["contributions_largest_first"]]
    assert found == [f"{effect} this prediction" for effect in effects]


def test_layperson_text_of_a_prediction() -> None:
    texts = local_text.facts_to_texts(_text_facts(1.4, [0.6]))
    assert orjson.loads(texts.prediction) == {
        "predicted_class": "y",
        "contributions_largest_first": [{"feature": "f0", "value": "0", "effect": "strongly towards this prediction"}],
        "likelihood": "likely",
    }


def test_a_custom_audience_decides_the_keys_of_every_fact() -> None:
    class Marker(Layperson):
        @override
        def contribution(self, contribution: local.Contribution, prediction: local.Prediction) -> dict[str, JSONValue]:
            return {"c": contribution.feature, "of": prediction.decision}

        @override
        def prediction(self, prediction: local.Prediction, confidence_level: float) -> dict[str, JSONValue]:
            return {"p": prediction.decision, "level": confidence_level}

        @override
        def change(self, change: local.Change) -> dict[str, JSONValue]:
            return {"ch": change.feature}

        @override
        def flip(self, counterfactual: local.Counterfactual, confidence_level: float) -> dict[str, JSONValue]:
            return {"fl": counterfactual.new_decision, "level": confidence_level}

        @override
        def margin(self, reason: local.SufficientReason) -> dict[str, JSONValue]:
            return {"m": reason.features}

        @override
        def share(self, share: float) -> str:
            return f"s{share}"

    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95), Marker())
    assert orjson.loads(texts.prediction) == {
        "predicted_class": "y",
        "contributions_largest_first": [
            {"feature": "f0", "value": "0", "c": "f0", "of": "y"},
            {"feature": "f1", "value": "1", "c": "f1", "of": "y"},
        ],
        "p": "y",
        "level": 0.95,
    }
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)", "ch": "f0"}],
        "minimum_number_of_changes": 1,
        "fl": 1,
        "level": 0.95,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "features_not_needed": ["f1"],
        "m": ["f0"],
    }
    assert [step["share_with_same_prediction"] for step in orjson.loads(texts.completions)["steps"]] == [
        "s0.5",
        "s0.12351",
    ]


def test_data_scientist_text_of_a_prediction() -> None:
    texts = local_text.facts_to_texts(_text_facts(1.4, [0.6123]), DataScientist())
    assert orjson.loads(texts.prediction) == {
        "predicted_class": "y",
        "contributions_largest_first": [{"feature": "f0", "value": "0", "score": 0.612, "std": 0.1}],
        "probability": 0.7,
        "confidence_interval": [0.6, 0.8],
        "confidence_level": 0.95,
        "logit": 1.4,
    }


def _text_facts_with_everything(confidence_level: float) -> local.LocalFacts:
    change = local.Change("f0", "1", "[2, 3)", -0.1234)
    steps = [local.CompletionStep([], 0.5), local.CompletionStep(["f0"], 0.12351)]
    return attrs.evolve(
        _text_facts(1.4, [0.6, 0.1], confidence_level),
        counterfactual=local.Counterfactual([change], new_decision=1, new_output=0.3),
        sufficient_reason=local.SufficientReason(["f0"], 0.9876, minimum=True),
        completions=steps,
    )


def test_layperson_text_of_the_explanations() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95))
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)"}],
        "minimum_number_of_changes": 1,
        "flip_holds_even_if_the_model_is_somewhat_off": True,
        "likelihood_of_new_prediction": "unlikely",
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "features_not_needed": ["f1"],
    }
    assert orjson.loads(texts.completions) == {
        "steps": [
            {"known_features": [], "share_with_same_prediction": "about as likely as not"},
            {"known_features": ["f0"], "share_with_same_prediction": "very unlikely"},
        ],
    }


def test_data_scientist_text_of_the_explanations() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95), DataScientist())
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)", "score_change": -0.123}],
        "minimum_number_of_changes": 1,
        "flip_holds_at_confidence": 0.95,
        "probability_of_new_prediction": 0.3,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "features_not_needed": ["f1"],
        "guaranteed_margin_logit": 0.988,
    }
    assert orjson.loads(texts.completions) == {
        "steps": [
            {"known_features": [], "share_with_same_prediction": 0.5},
            {"known_features": ["f0"], "share_with_same_prediction": 0.124},
        ],
    }


@pytest.mark.parametrize(("confidence_level", "holds"), [(-0.5, False), (0.0, False), (0.5, True), (2.0, True)])
def test_layperson_is_told_whether_the_flip_survives_model_error(confidence_level: float, *, holds: bool) -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(confidence_level))
    assert orjson.loads(texts.counterfactual)["flip_holds_even_if_the_model_is_somewhat_off"] is holds
