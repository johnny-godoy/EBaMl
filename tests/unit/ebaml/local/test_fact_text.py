"""Tests for the texts of the local facts."""

import attrs
import orjson
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.custom_types import Audience
from ebaml.local import fact_text as local_text
from ebaml.local import facts as local
from tests.unit.ebaml.conftest import instance_of, returning


def test_texts_hide_numbers_from_layperson(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, data = continuous_ebm
    facts = local.compute_local_facts(ebm, instance_of(data, 0))
    layperson = local_text.facts_to_texts(facts)
    expert = local_text.facts_to_texts(facts, Audience.DATA_SCIENTIST)
    for text in attrs.astuple(layperson):
        for key in ("score", "std", "probability", "logit", "confidence"):
            assert key not in text
    assert "score" in expert.prediction
    assert "flip_holds_at_confidence" in expert.counterfactual
    assert "confidence_interval" in expert.prediction
    assert orjson.loads(expert.completions)["steps"][0]["share_with_same_prediction"] <= 1


@pytest.mark.parametrize(
    ("probability", "word"),
    [
        (0.99, "almost certain"),
        (0.85, "very likely"),
        (0.7, "likely"),
        (0.5, "about as likely as not"),
        (0.3, "unlikely"),
        (0.1, "very unlikely"),
        (0.01, "almost impossible"),
    ],
)
def test_likelihood_words(probability: float, word: str) -> None:
    assert local_text._likelihood(probability) == word


@pytest.mark.parametrize(("score", "word"), [(-0.6, "strongly"), (0.3, "moderately"), (0.1, "slightly")])
def test_strength_words(score: float, word: str) -> None:
    assert local_text._strength(score) == word


def test_texts_without_counterfactual_or_sufficient_reason() -> None:
    prediction = local.Prediction(decision=0, output=0.7, lower=0.6, upper=0.8, score=-0.8, score_std=0.1, margin=0.8)
    facts = local.LocalFacts(prediction, [], None, None, [], 0.95)
    texts = local_text.facts_to_texts(facts, Audience.DATA_SCIENTIST)
    assert orjson.loads(texts.counterfactual) == {"flip_possible": False}
    assert orjson.loads(texts.sufficient_reason) == {"exists": False}


@pytest.mark.parametrize(
    ("probability", "word"),
    [
        (0.95, "almost certain"),
        (0.8, "very likely"),
        (0.6, "likely"),
        (0.4, "about as likely as not"),
        (0.2, "unlikely"),
        (0.05, "very unlikely"),
    ],
)
def test_likelihood_words_start_at_their_threshold(probability: float, word: str) -> None:
    assert local_text._likelihood(probability) == word


@pytest.mark.parametrize(
    ("score", "word"),
    [(0.5, "strongly"), (-0.5, "strongly"), (0.2, "moderately"), (-0.2, "moderately")],
)
def test_strength_words_start_at_their_threshold(score: float, word: str) -> None:
    assert local_text._strength(score) == word


def test_round_keeps_three_decimals() -> None:
    assert local_text._round(0.12351) == pytest.approx(0.124, abs=1e-12)
    assert local_text._round(0.12349) == pytest.approx(0.123, abs=1e-12)


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


def test_data_scientist_text_of_a_prediction() -> None:
    texts = local_text.facts_to_texts(_text_facts(1.4, [0.6123]), Audience.DATA_SCIENTIST)
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
        counterfactual=local.Counterfactual([change], new_decision=1, new_output=0.7),
        sufficient_reason=local.SufficientReason(["f0"], 0.9876, minimum=True),
        completions=steps,
    )


def test_layperson_text_of_the_explanations() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95))
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)"}],
        "minimum_number_of_changes": 1,
        "after_the_changes": {"predicted_class": 1, "likelihood": "likely"},
        "flip_holds_even_if_the_model_is_somewhat_off": True,
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
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95), Audience.DATA_SCIENTIST)
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)", "score_change": -0.123}],
        "minimum_number_of_changes": 1,
        "after_the_changes": {"predicted_class": 1, "probability": 0.7},
        "flip_holds_at_confidence": 0.95,
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


@pytest.mark.parametrize("audience", list(Audience))
def test_counterfactual_text_puts_the_new_class_and_its_probability_together(audience: Audience) -> None:
    # with the probability in a key of its own, a live reply read 0.82 as the probability of the old class
    facts = attrs.evolve(
        _text_facts_with_everything(0.95),
        counterfactual=local.Counterfactual(
            [local.Change("f0", "1", "[2, 3)", -0.1)],
            new_decision="loan approved",
            new_output=0.82,
        ),
    )
    counterfactual = orjson.loads(local_text.facts_to_texts(facts, audience).counterfactual)
    after = counterfactual["after_the_changes"]
    assert after["predicted_class"] == "loan approved"
    assert set(after) == {"predicted_class", "likelihood" if audience is Audience.LAYPERSON else "probability"}


@pytest.mark.parametrize(("confidence_level", "holds"), [(-0.5, False), (0.0, False), (0.5, True), (2.0, True)])
def test_layperson_is_told_whether_the_flip_survives_model_error(confidence_level: float, *, holds: bool) -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(confidence_level))
    assert orjson.loads(texts.counterfactual)["flip_holds_even_if_the_model_is_somewhat_off"] is holds


RENDERER_METHODS = ["contribution", "prediction", "change", "flip", "margin", "share"]


@pytest.mark.parametrize("missing", RENDERER_METHODS)
def test_a_renderer_must_define_every_method(missing: str) -> None:
    others = {name: returning(None) for name in RENDERER_METHODS if name != missing}
    incomplete = type("Incomplete", (local_text._AudienceRenderer,), others)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("name", RENDERER_METHODS)
@pytest.mark.parametrize("audience", list(Audience))
def test_renderers_mark_the_methods_they_override(audience: Audience, name: str) -> None:
    method = type(local_text._RENDERERS[audience]).__dict__[name]
    assert method.__override__ is True
