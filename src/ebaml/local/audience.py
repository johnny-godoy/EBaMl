"""The audience of a local explanation: who reads it decides which facts the LLM gets and how it is told to write.

``Audience`` is the base class. A subclass fills in the rules of one kind of reader:

- ``guidance`` is the instruction that tells the LLM how to write for this reader.
- ``contribution``, ``prediction``, ``change``, ``flip``, ``margin`` and ``share`` return the keys that the audience
  adds to the facts of the prompt. Each is abstract, so the author of an audience decides on every model number.

``Layperson`` and ``DataScientist`` are the two audiences that ship with the package. To change one rule, subclass one
of them and override one method. Pass an instance of any subclass as ``audience=`` to ``explain_instance`` or to
``facts_to_texts``.
"""

import abc
from collections.abc import Mapping
from typing import override

from ebaml.core.custom_types import JSONValue
from ebaml.core.exceptions import InvalidOptionError
from ebaml.local.facts import Change, Contribution, Counterfactual, Prediction, SufficientReason

_LIKELIHOOD_WORDS = (
    (0.95, "almost certain"),
    (0.8, "very likely"),
    (0.6, "likely"),
    (0.4, "about as likely as not"),
    (0.2, "unlikely"),
    (0.05, "very unlikely"),
)
_STRENGTH_WORDS = ((0.5, "strongly"), (0.2, "moderately"))


def likelihood_word(probability: float) -> str:
    """Return the words for a probability, from ``almost certain`` to ``almost impossible``.

    Parameters
    ----------
    probability: float
        The probability, in ``[0, 1]``.

    Returns
    -------
    str
        The words. Each word starts at its threshold: ``0.8`` is ``very likely``.

    """
    for threshold, word in _LIKELIHOOD_WORDS:
        if probability >= threshold:
            return word
    return "almost impossible"


def strength_word(score: float) -> str:
    """Return the word for the size of a score, from ``strongly`` to ``slightly``.

    Parameters
    ----------
    score: float
        The score. Only its size counts, so ``-0.6`` and ``0.6`` give the same word.

    Returns
    -------
    str
        ``strongly`` from 0.5, ``moderately`` from 0.2, else ``slightly``.

    """
    for threshold, word in _STRENGTH_WORDS:
        if abs(score) >= threshold:
            return word
    return "slightly"


def round_value(value: float) -> float:
    """Round a number to three decimals, the precision that the prompts show.

    Parameters
    ----------
    value: float
        The number.

    Returns
    -------
    float
        The rounded number.

    """
    return round(value, 3)


def points_towards(contribution: Contribution, prediction: Prediction) -> bool:
    """Tell whether a contribution pushes towards the prediction.

    A score points towards the prediction if it has the sign of the score of the prediction (the logit). A logit of 0
    belongs to the negative class.

    Parameters
    ----------
    contribution: Contribution
        The contribution of one feature.
    prediction: Prediction
        The prediction.

    Returns
    -------
    bool
        True if the contribution supports the prediction.

    """
    return (contribution.score if prediction.score > 0 else -contribution.score) > 0


class Audience(abc.ABC):
    """The base class of the audiences: one subclass for each kind of reader.

    ``facts_to_texts`` builds the keys that all audiences share. The methods here return the keys that one audience
    adds. All methods are abstract, so a new audience cannot leak a model number that its author forgot. To change one
    rule of a built-in audience, subclass ``Layperson`` or ``DataScientist``.

    Examples
    --------
    A reader that wants every explanation in French, with no model numbers:

    ```python
    from ebaml.local.audience import Layperson


    class French(Layperson):
        def guidance(self) -> str:
            return "The reader speaks French and is not technical, so answer in French, in everyday words."
    ```

    """

    @abc.abstractmethod
    def guidance(self) -> str:
        """Return the instruction that tells the LLM how to write for this reader, and why."""

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


class Layperson(Audience):
    """Replace every model number by a word, so the LLM has no number to repeat."""

    @override
    def guidance(self) -> str:
        return (
            "The reader is not technical and knows no statistics, so write in everyday words. Express strength and "
            "likelihood in plain words such as 'strongly', 'slightly', 'very likely' or 'about as likely as not'. "
            "Mention only values of the person's own inputs (for example their age), because model numbers (scores, "
            "log-odds, probabilities, percentages, intervals, standard deviations) mean nothing to this reader."
        )

    @override
    def contribution(
        self,
        contribution: Contribution,
        prediction: Prediction,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        direction = "towards" if points_towards(contribution, prediction) else "against"
        return {"effect": f"{strength_word(contribution.score)} {direction} this prediction"}

    @override
    def prediction(
        self,
        prediction: Prediction,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"likelihood": likelihood_word(prediction.output)}

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
            "likelihood_of_new_prediction": likelihood_word(counterfactual.new_output),
        }

    @override
    def margin(
        self,
        reason: SufficientReason,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {}

    @override
    def share(self, share: float) -> float | str:
        return likelihood_word(share)


class DataScientist(Audience):
    """Give the scores, the probabilities and the confidence interval, rounded."""

    @override
    def guidance(self) -> str:
        return (
            "The reader is a data scientist, so be precise. Quote the scores (log-odds contributions), probabilities "
            "and confidence intervals from the facts. Say how uncertain a statement is when the facts show wide "
            "confidence intervals."
        )

    @override
    def contribution(
        self,
        contribution: Contribution,
        prediction: Prediction,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"score": round_value(contribution.score), "std": round_value(contribution.std)}

    @override
    def prediction(
        self,
        prediction: Prediction,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {
            "probability": round_value(prediction.output),
            "confidence_interval": [round_value(prediction.lower), round_value(prediction.upper)],
            "confidence_level": confidence_level,
            "logit": round_value(prediction.score),
        }

    @override
    def change(
        self,
        change: Change,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"score_change": round_value(change.score_change)}

    @override
    def flip(
        self,
        counterfactual: Counterfactual,
        confidence_level: float,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {
            "flip_holds_at_confidence": confidence_level,
            "probability_of_new_prediction": round_value(counterfactual.new_output),
        }

    @override
    def margin(
        self,
        reason: SufficientReason,
    ) -> dict[str, JSONValue]:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
        return {"guaranteed_margin_logit": round_value(reason.guaranteed_margin)}

    @override
    def share(self, share: float) -> float | str:
        return round_value(share)


BUILT_IN_AUDIENCES: Mapping[str, Audience] = {
    "layperson": Layperson(),
    "data_scientist": DataScientist(),
}


def resolve_audience(audience: Audience | str) -> Audience:
    """Return the audience for an ``Audience`` or for the name of a built-in one.

    Parameters
    ----------
    audience: Audience | str
        An audience, or ``"layperson"`` or ``"data_scientist"``.

    Returns
    -------
    Audience
        The audience.

    Raises
    ------
    InvalidOptionError
        If the name is not the name of a built-in audience.

    """
    # one check where the value enters ebaml: it narrows the union of the hint
    if isinstance(audience, str):
        try:
            return BUILT_IN_AUDIENCES[audience]
        except KeyError:
            raise InvalidOptionError(audience, list(BUILT_IN_AUDIENCES)) from None
    return audience
