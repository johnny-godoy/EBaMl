"""Tests for the texts of the local facts."""

from typing import override

import attrs
import orjson
import pytest

from ebaml.core.custom_types import JSONValue
from ebaml.local import fact_text as local_text
from ebaml.local import facts as local
from ebaml.local.audience import Audience, DataScientist, Layperson
from ebaml.local.local_kinds import OutputKind
from tests.unit.ebaml.conftest import FittedClassifier, instance_of


def test_the_default_audience_is_a_layperson(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    facts = local.compute_local_facts(ebm, instance_of(data, 0))
    assert local_text.facts_to_texts(facts) == local_text.facts_to_texts(facts, Layperson())


def test_texts_hide_numbers_from_layperson(continuous_ebm: FittedClassifier) -> None:
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
    facts = local.LocalFacts(prediction, [], None, None, [], 0.95, min_support=0.01, n_samples=10_000)
    texts = local_text.facts_to_texts(facts, DataScientist())
    # a search that found nothing still names its conditions, because the result holds under them only
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": False,
        "flip_holds_at_confidence": 0.95,
        "rare_values_left_out": True,
        "min_support": 0.01,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": False,
        "guarantee_holds_at_confidence": 0.95,
        "rare_values_left_out": True,
        "min_support": 0.01,
    }


def _text_facts(
    logit: float,
    scores: list[float],
    confidence_level: float = 0.95,
    *,
    min_support: float = 0.01,
    n_samples: int = 10_000,
) -> local.LocalFacts:
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
    return local.LocalFacts(
        prediction,
        contributions,
        None,
        None,
        [],
        confidence_level,
        min_support=min_support,
        n_samples=n_samples,
    )


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


class _Marker(Layperson):
    """An audience that marks each key, so a test sees which method made it."""

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
    def flip(self, confidence_level: float) -> dict[str, JSONValue]:
        return {"fl": confidence_level > 0}

    @override
    def class_output(self, output: float, output_kind: OutputKind) -> dict[str, JSONValue]:
        return {"co": output}

    @override
    def margin(self, reason: local.SufficientReason, output_kind: OutputKind) -> dict[str, JSONValue]:
        return {"m": reason.features}

    @override
    def guarantee(self, confidence_level: float) -> dict[str, JSONValue]:
        return {"g": confidence_level}

    @override
    def support(self, min_support: float) -> dict[str, JSONValue]:
        return {"su": min_support}

    @override
    def lower_bound(self, lower_bound: int, items: str) -> dict[str, JSONValue]:
        return {f"lb_{items}": lower_bound}

    @override
    def share(self, share: float) -> str:
        return f"s{share}"

    @override
    def sample(self, n_samples: int) -> dict[str, JSONValue]:
        return {"sa": n_samples}


def test_a_custom_audience_decides_the_keys_of_every_fact() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95), _Marker())
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
        "number_of_changes": 1,
        "is_smallest_set": True,
        "lb_changes": 1,
        "after_the_changes": {"predicted_class": 1, "co": 0.7},
        "fl": True,
        "su": 0.01,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "is_smallest_set": True,
        "lb_features": 1,
        "features_not_needed": ["f1"],
        "m": ["f0"],
        "g": 0.95,
        "su": 0.01,
    }
    completions = orjson.loads(texts.completions)
    assert [step["share_with_same_prediction"] for step in completions["steps"]] == ["s0.5", "s0.12351"]
    assert completions["sa"] == 10_000


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
        counterfactual=local.Counterfactual([change], new_decision=1, new_output=0.7, minimum=True, lower_bound=1),
        sufficient_reason=local.SufficientReason(["f0"], 0.9876, minimum=True, lower_bound=1),
        completions=steps,
    )


def test_layperson_text_of_the_explanations() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95))
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)"}],
        "number_of_changes": 1,
        "is_smallest_set": True,
        "after_the_changes": {"predicted_class": 1, "likelihood": "likely"},
        "flip_holds_even_if_the_model_is_somewhat_off": True,
        "rare_values_left_out": True,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "is_smallest_set": True,
        "features_not_needed": ["f1"],
        "guarantee_holds_even_if_the_model_is_somewhat_off": True,
        "rare_values_left_out": True,
    }
    assert orjson.loads(texts.completions) == {
        "steps": [
            {"known_features": [], "share_with_same_prediction": "about as likely as not"},
            {"known_features": ["f0"], "share_with_same_prediction": "very unlikely"},
        ],
        "sample_size": "large",
    }


def test_data_scientist_text_of_the_explanations() -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(0.95), DataScientist())
    assert orjson.loads(texts.counterfactual) == {
        "flip_possible": True,
        "changes": [{"feature": "f0", "from": "1", "to": "[2, 3)", "score_change": -0.123}],
        "number_of_changes": 1,
        "is_smallest_set": True,
        "smallest_possible_number_of_changes": 1,
        "after_the_changes": {"predicted_class": 1, "probability": 0.7},
        "flip_holds_at_confidence": 0.95,
        "rare_values_left_out": True,
        "min_support": 0.01,
    }
    assert orjson.loads(texts.sufficient_reason) == {
        "exists": True,
        "features": ["f0"],
        "is_smallest_set": True,
        "smallest_possible_number_of_features": 1,
        "features_not_needed": ["f1"],
        "guaranteed_margin_logit": 0.988,
        "guarantee_holds_at_confidence": 0.95,
        "rare_values_left_out": True,
        "min_support": 0.01,
    }
    assert orjson.loads(texts.completions) == {
        "steps": [
            {"known_features": [], "share_with_same_prediction": 0.5},
            {"known_features": ["f0"], "share_with_same_prediction": 0.124},
        ],
        "number_of_samples": 10000,
    }


@pytest.mark.parametrize(("audience", "key"), [(Layperson(), "likelihood"), (DataScientist(), "probability")])
def test_counterfactual_text_puts_the_new_class_and_its_probability_together(audience: Audience, key: str) -> None:
    # with the probability in a key of its own, a live reply read 0.82 as the probability of the old class
    facts = attrs.evolve(
        _text_facts_with_everything(0.95),
        counterfactual=local.Counterfactual(
            [local.Change("f0", "1", "[2, 3)", -0.1)],
            new_decision="loan approved",
            new_output=0.82,
            minimum=True,
            lower_bound=1,
        ),
    )
    counterfactual = orjson.loads(local_text.facts_to_texts(facts, audience).counterfactual)
    after = counterfactual["after_the_changes"]
    assert after["predicted_class"] == "loan approved"
    assert set(after) == {"predicted_class", key}


@pytest.mark.parametrize(("confidence_level", "holds"), [(-0.5, False), (0.0, False), (0.5, True), (2.0, True)])
def test_layperson_is_told_whether_the_flip_survives_model_error(confidence_level: float, *, holds: bool) -> None:
    texts = local_text.facts_to_texts(_text_facts_with_everything(confidence_level))
    assert orjson.loads(texts.counterfactual)["flip_holds_even_if_the_model_is_somewhat_off"] is holds


@pytest.mark.parametrize("audience", [Layperson(), DataScientist()], ids=repr)
def test_the_facts_claim_no_condition_that_the_options_turn_off(audience: Audience) -> None:
    # contract 3: at confidence_level=0 and min_support=0 no score moved and no value was left out, so no key of the
    # facts may say so, with a result and without one
    with_results = attrs.evolve(_text_facts_with_everything(0.0), min_support=0.0, n_samples=50)
    without_results = _text_facts(1.4, [0.6], 0.0, min_support=0.0, n_samples=50)
    for facts in (with_results, without_results):
        texts = local_text.facts_to_texts(facts, audience)
        counterfactual = orjson.loads(texts.counterfactual)
        sufficient = orjson.loads(texts.sufficient_reason)
        assert counterfactual | audience.flip(0.0) | audience.support(0.0) == counterfactual
        assert sufficient | audience.guarantee(0.0) | audience.support(0.0) == sufficient
        assert counterfactual["rare_values_left_out"] is False
        assert sufficient["rare_values_left_out"] is False
        assert orjson.loads(texts.completions) | audience.sample(50) == orjson.loads(texts.completions)


def test_the_conditions_of_the_facts_follow_the_options() -> None:
    facts = attrs.evolve(_text_facts_with_everything(0.5), min_support=0.2, n_samples=500)
    texts = local_text.facts_to_texts(facts, DataScientist())
    assert orjson.loads(texts.counterfactual)["flip_holds_at_confidence"] == pytest.approx(0.5)
    assert orjson.loads(texts.counterfactual)["min_support"] == pytest.approx(0.2)
    assert orjson.loads(texts.sufficient_reason)["guarantee_holds_at_confidence"] == pytest.approx(0.5)
    assert orjson.loads(texts.sufficient_reason)["min_support"] == pytest.approx(0.2)
    assert orjson.loads(texts.completions)["number_of_samples"] == 500
    assert orjson.loads(local_text.facts_to_texts(facts).completions)["sample_size"] == "small"


def test_a_counterfactual_that_is_not_proved_is_not_called_the_smallest() -> None:
    facts = _text_facts_with_everything(0.95)
    assert facts.counterfactual is not None
    target = attrs.evolve(facts, counterfactual=attrs.evolve(facts.counterfactual, minimum=False))
    assert orjson.loads(local_text.facts_to_texts(facts).counterfactual)["is_smallest_set"] is True
    assert orjson.loads(local_text.facts_to_texts(target).counterfactual)["is_smallest_set"] is False


def test_a_minimal_set_is_not_called_the_smallest() -> None:
    facts = _text_facts_with_everything(0.95)
    minimal = attrs.evolve(
        facts,
        sufficient_reason=local.SufficientReason(["f0"], 0.5, minimum=False, lower_bound=1),
    )
    assert orjson.loads(local_text.facts_to_texts(minimal).sufficient_reason)["is_smallest_set"] is False


@pytest.mark.parametrize(
    ("audience", "keys"),
    [
        (Layperson(), set()),
        (DataScientist(), {"smallest_possible_number_of_changes", "smallest_possible_number_of_features"}),
    ],
)
def test_only_a_data_scientist_gets_the_lower_bound_of_a_set(audience: Audience, keys: set[str]) -> None:
    facts = _text_facts_with_everything(0.95)
    assert facts.counterfactual is not None
    unproved = attrs.evolve(
        facts,
        counterfactual=attrs.evolve(facts.counterfactual, minimum=False, lower_bound=0),
        sufficient_reason=local.SufficientReason(["f0", "f1"], 0.5, minimum=False, lower_bound=1),
    )
    texts = local_text.facts_to_texts(unproved, audience)
    counterfactual = orjson.loads(texts.counterfactual)
    sufficient = orjson.loads(texts.sufficient_reason)
    found = (set(counterfactual) | set(sufficient)) & {
        "smallest_possible_number_of_changes",
        "smallest_possible_number_of_features",
    }
    assert found == keys
    if keys:
        assert counterfactual["smallest_possible_number_of_changes"] == 0
        assert sufficient["smallest_possible_number_of_features"] == 1
