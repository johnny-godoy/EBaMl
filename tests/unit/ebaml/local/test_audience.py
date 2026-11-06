"""Tests for the audiences of the local explanations."""

from pathlib import Path
from typing import override

import pytest

from ebaml.core.exceptions import InvalidOptionError
from ebaml.local import audience as audiences
from ebaml.local import facts as local
from ebaml.local.local_kinds import OutputKind
from tests.unit.ebaml.conftest import returning

AUDIENCE_METHODS = ["guidance", "contribution", "prediction", "change", "flip", "class_output", "margin", "share"]
PROMPT_TESTS = Path(__file__).parents[4] / "baml_src" / "prompt_tests.baml"


@pytest.mark.parametrize(
    ("probability", "word"),
    [
        (0.99, "almost certain"),
        (0.95, "almost certain"),
        (0.85, "very likely"),
        (0.8, "very likely"),
        (0.7, "likely"),
        (0.6, "likely"),
        (0.5, "about as likely as not"),
        (0.4, "about as likely as not"),
        (0.3, "unlikely"),
        (0.2, "unlikely"),
        (0.1, "very unlikely"),
        (0.05, "very unlikely"),
        (0.01, "almost impossible"),
    ],
)
def test_likelihood_words_start_at_their_threshold(probability: float, word: str) -> None:
    assert audiences.likelihood_word(probability) == word


@pytest.mark.parametrize(
    ("score", "word"),
    [
        (-0.6, "strongly"),
        (0.5, "strongly"),
        (-0.5, "strongly"),
        (0.3, "moderately"),
        (0.2, "moderately"),
        (-0.2, "moderately"),
        (0.1, "slightly"),
    ],
)
def test_strength_words_start_at_their_threshold(score: float, word: str) -> None:
    assert audiences.strength_word(score) == word


def test_round_keeps_three_decimals() -> None:
    assert audiences.round_value(0.12351) == pytest.approx(0.124, abs=1e-12)
    assert audiences.round_value(0.12349) == pytest.approx(0.123, abs=1e-12)


@pytest.mark.parametrize(
    ("logit", "score", "towards"),
    [
        (1.4, 0.6, True),
        (1.4, -0.6, False),
        (-0.8, -0.6, True),
        (-0.8, 0.6, False),
        (0.0, -0.3, True),
        (0.0, 0.3, False),
    ],
)
def test_a_logit_of_zero_belongs_to_the_negative_class(logit: float, score: float, *, towards: bool) -> None:
    prediction = local.Prediction(decision="y", output=0.7, lower=0.6, upper=0.8, score=logit, score_std=0.1, margin=1)
    assert audiences.points_towards(local.Contribution("f", "v", score, 0.1), prediction) is towards


@pytest.mark.parametrize("missing", AUDIENCE_METHODS)
def test_an_audience_must_define_every_method(missing: str) -> None:
    others = {name: returning(None) for name in AUDIENCE_METHODS if name != missing}
    incomplete = type("Incomplete", (audiences.Audience,), others)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("name", AUDIENCE_METHODS)
@pytest.mark.parametrize("audience", ["layperson", "data_scientist"])
def test_built_in_audiences_mark_the_methods_they_override(audience: str, name: str) -> None:
    method = type(audiences.BUILT_IN_AUDIENCES[audience]).__dict__[name]
    assert method.__override__ is True


def test_the_built_in_audiences_are_the_two_named_ones() -> None:
    assert {name: type(audience) for name, audience in audiences.BUILT_IN_AUDIENCES.items()} == {
        "layperson": audiences.Layperson,
        "data_scientist": audiences.DataScientist,
    }


@pytest.mark.parametrize("name", ["layperson", "data_scientist"])
def test_resolve_audience_returns_the_built_in_audience(name: str) -> None:
    assert audiences.resolve_audience(name) is audiences.BUILT_IN_AUDIENCES[name]


def test_resolve_audience_returns_a_given_audience() -> None:
    class Custom(audiences.Layperson):
        @override
        def guidance(self) -> str:
            return "CUSTOM"

    custom = Custom()
    assert audiences.resolve_audience(custom) is custom


def test_resolve_audience_rejects_an_unknown_name_and_lists_the_valid_ones() -> None:
    with pytest.raises(InvalidOptionError, match=r"child.*layperson.*data_scientist"):
        audiences.resolve_audience("child")


@pytest.mark.parametrize(
    ("name", "function"),
    [("layperson", "live_layperson_guidance"), ("data_scientist", "live_data_scientist_guidance")],
)
def test_the_live_prompt_tests_run_the_guidance_that_users_get(name: str, function: str) -> None:
    source = PROMPT_TESTS.read_text(encoding="utf-8")
    body = source.split(f"function {function}() -> string {{", 1)[1].split("}", 1)[0]
    assert audiences.BUILT_IN_AUDIENCES[name].guidance() in body


@pytest.mark.parametrize(
    ("score", "contribution", "towards"),
    [
        (12.0, 1.5, True),  # above the threshold 10: a higher score supports the decision "above"
        (12.0, -1.5, False),
        (8.0, -1.5, True),  # below it: a lower score supports the decision "at most"
        (8.0, 1.5, False),
        (10.0, -0.3, True),  # a score on the threshold belongs to the low side, as the log-odds zero does
    ],
)
def test_a_contribution_points_towards_the_side_of_the_threshold(
    score: float,
    contribution: float,
    *,
    towards: bool,
) -> None:
    prediction = local.Prediction(
        decision="d",
        output=score,
        lower=score,
        upper=score,
        score=score,
        score_std=0.1,
        margin=abs(score - 10.0),
        threshold=10.0,
        output_kind=OutputKind.VALUE,
    )
    assert audiences.points_towards(local.Contribution("f", "v", contribution, 0.1), prediction) is towards


def _value_prediction() -> local.Prediction:
    return local.Prediction(
        decision="above 10",
        output=12.3456,
        lower=11.1111,
        upper=13.5555,
        score=12.3456,
        score_std=0.5,
        margin=2.3456,
        threshold=10.0,
        output_kind=OutputKind.VALUE,
    )


def test_a_layperson_gets_the_predicted_value_of_a_regressor() -> None:
    audience = audiences.Layperson()
    assert audience.prediction(_value_prediction(), 0.95) == {"predicted_value": 12.346}
    assert audience.class_output(12.3456, OutputKind.VALUE) == {"predicted_value": 12.346}
    assert audience.class_output(0.9, OutputKind.PROBABILITY) == {"likelihood": "very likely"}


def test_a_data_scientist_gets_the_value_interval_and_margin_of_a_regressor() -> None:
    audience = audiences.DataScientist()
    assert audience.prediction(_value_prediction(), 0.95) == {
        "predicted_value": 12.346,
        "confidence_interval": [11.111, 13.556],
        "confidence_level": 0.95,
        "score": 12.346,
    }
    assert audience.class_output(12.3456, OutputKind.VALUE) == {"predicted_value": 12.346}
    reason = local.SufficientReason(["f"], 1.23456, minimum=True)
    assert audience.margin(reason, OutputKind.VALUE) == {"guaranteed_margin": 1.235}
    assert audience.margin(reason, OutputKind.PROBABILITY) == {"guaranteed_margin_logit": 1.235}
