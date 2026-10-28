"""Put the local facts of one EBM prediction in words, as JSON texts for the LLM prompts.

``ebaml.local`` computes the facts. This module only renders them, for one audience: a layperson gets words in place
of the model numbers, so the LLM has no number to repeat, and a data scientist gets the numbers. The prompts that read
the texts are in ``baml_src/local_explanations.baml``.
"""

import abc
from collections.abc import Mapping
from typing import override

import attrs
import orjson

from ebaml.core.custom_types import Audience, JSONValue
from ebaml.local.facts import Change, Contribution, Counterfactual, LocalFacts, Prediction, SufficientReason

_LIKELIHOOD_WORDS = (
    (0.95, "almost certain"),
    (0.8, "very likely"),
    (0.6, "likely"),
    (0.4, "about as likely as not"),
    (0.2, "unlikely"),
    (0.05, "very unlikely"),
)
_STRENGTH_WORDS = ((0.5, "strongly"), (0.2, "moderately"))


def _likelihood(probability: float) -> str:
    for threshold, word in _LIKELIHOOD_WORDS:
        if probability >= threshold:
            return word
    return "almost impossible"


def _strength(score: float) -> str:
    for threshold, word in _STRENGTH_WORDS:
        if abs(score) >= threshold:
            return word
    return "slightly"


def _dump(data: Mapping[str, JSONValue]) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()


def _round(value: float) -> float:
    return round(value, 3)


@attrs.frozen
class FactTexts:
    """The JSON text of each explanation, as the prompts read them."""

    prediction: str
    counterfactual: str
    sufficient_reason: str
    completions: str


class _AudienceRenderer(abc.ABC):
    """The parts of the fact texts that depend on the audience.

    ``facts_to_texts`` builds the keys that all audiences share. Each method here returns the keys that one audience
    adds, so a new audience is a new subclass and one entry in ``_RENDERERS``.
    """

    @abc.abstractmethod
    def contribution(
        self,
        contribution: Contribution,
        prediction: Prediction,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        """Return the keys that describe the size of one contribution."""

    @abc.abstractmethod
    def prediction(
        self,
        prediction: Prediction,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        """Return the keys that describe how sure the model is of the prediction."""

    @abc.abstractmethod
    def change(
        self,
        change: Change,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        """Return the keys that describe the size of one change of the counterfactual."""

    @abc.abstractmethod
    def flip(
        self,
        counterfactual: Counterfactual,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        """Return the keys that tell how sure the flip of the counterfactual is."""

    @abc.abstractmethod
    def margin(
        self,
        reason: SufficientReason,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        """Return the keys that describe the margin of the sufficient reason."""

    @abc.abstractmethod
    def share(self, share: float) -> float | str:
        """Return the share of completions with the same prediction."""


class _LaypersonRenderer(_AudienceRenderer):
    """Replace every model number by a word, so the LLM has no number to repeat."""

    @override
    def contribution(
        self,
        contribution: Contribution,
        prediction: Prediction,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        # a score points towards the prediction if it has the sign of the score of the prediction (the logit); a logit
        # of 0 is the negative class
        direction = (
            "towards" if (contribution.score if prediction.score > 0 else -contribution.score) > 0 else "against"
        )
        return {"effect": f"{_strength(contribution.score)} {direction} this prediction"}

    @override
    def prediction(
        self,
        prediction: Prediction,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"likelihood": _likelihood(prediction.output)}

    @override
    def change(
        self,
        change: Change,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {}

    @override
    def flip(
        self,
        counterfactual: Counterfactual,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {
            "flip_holds_even_if_the_model_is_somewhat_off": confidence_level > 0,
            "likelihood_of_new_prediction": _likelihood(counterfactual.new_output),
        }

    @override
    def margin(
        self,
        reason: SufficientReason,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {}

    @override
    def share(self, share: float) -> float | str:
        return _likelihood(share)


class _DataScientistRenderer(_AudienceRenderer):
    """Give the scores, the probabilities and the confidence interval, rounded."""

    @override
    def contribution(
        self,
        contribution: Contribution,
        prediction: Prediction,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"score": _round(contribution.score), "std": _round(contribution.std)}

    @override
    def prediction(
        self,
        prediction: Prediction,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {
            "probability": _round(prediction.output),
            "confidence_interval": [_round(prediction.lower), _round(prediction.upper)],
            "confidence_level": confidence_level,
            "logit": _round(prediction.score),
        }

    @override
    def change(
        self,
        change: Change,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"score_change": _round(change.score_change)}

    @override
    def flip(
        self,
        counterfactual: Counterfactual,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {
            "flip_holds_at_confidence": confidence_level,
            "probability_of_new_prediction": _round(counterfactual.new_output),
        }

    @override
    def margin(
        self,
        reason: SufficientReason,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"guaranteed_margin_logit": _round(reason.guaranteed_margin)}

    @override
    def share(self, share: float) -> float | str:
        return _round(share)


_RENDERERS: Mapping[Audience, _AudienceRenderer] = {
    Audience.LAYPERSON: _LaypersonRenderer(),
    Audience.DATA_SCIENTIST: _DataScientistRenderer(),
}


def facts_to_texts(facts: LocalFacts, audience: Audience = Audience.LAYPERSON) -> FactTexts:
    """Render the facts as one JSON text per explanation, for the prompts.

    For a layperson the model numbers (scores, probabilities, intervals) are replaced by words, so the LLM has none to
    repeat. A data scientist gets the numbers.

    Parameters
    ----------
    facts: LocalFacts
        The facts of the prediction.
    audience: Audience, default=Audience.LAYPERSON
        Who will read the explanation.

    Returns
    -------
    FactTexts
        The texts of the prediction, the counterfactual, the sufficient reason and the completions.

    """
    renderer = _RENDERERS[audience]
    p = facts.prediction
    prediction: dict[str, JSONValue] = {"predicted_class": p.decision}
    prediction["contributions_largest_first"] = [
        {"feature": c.feature, "value": c.value} | renderer.contribution(c, p) for c in facts.contributions
    ]
    prediction |= renderer.prediction(p, facts.confidence_level)

    cf = facts.counterfactual
    counterfactual: dict[str, JSONValue] = {"flip_possible": cf is not None}
    if cf is not None:
        counterfactual["changes"] = [
            {"feature": c.feature, "from": c.from_value, "to": c.to_value} | renderer.change(c) for c in cf.changes
        ]
        counterfactual["minimum_number_of_changes"] = len(cf.changes)
        counterfactual |= renderer.flip(cf, facts.confidence_level)

    all_names = [c.feature for c in facts.contributions]
    sr = facts.sufficient_reason
    sufficient: dict[str, JSONValue] = {"exists": sr is not None}
    if sr is not None:
        sufficient["features"] = sr.features
        sufficient["features_not_needed"] = [f for f in all_names if f not in sr.features]
        sufficient |= renderer.margin(sr)

    completions = [
        {"known_features": s.fixed, "share_with_same_prediction": renderer.share(s.share)} for s in facts.completions
    ]
    return FactTexts(
        prediction=_dump(prediction),
        counterfactual=_dump(counterfactual),
        sufficient_reason=_dump(sufficient),
        completions=_dump({"steps": completions}),
    )
