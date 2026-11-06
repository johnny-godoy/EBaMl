"""What each kind of EBM tells the local explanations about one prediction.

The algorithms of ``ebaml.local.facts`` work on one margin: a sum of one term per feature, positive while the decision
holds. This module turns the prediction of a model into that margin. One ``LocalKind`` class does it for each model
kind (``ebaml.graphs.model_kinds``), so ``facts`` never tests the kind of the model. ``local_kind`` selects the class.

- ``BinaryLocal`` keeps the predicted class. The margin is the log-odds of that class.
- ``RegressorLocal`` keeps the answer to a ``Decision`` (``ebaml.local.decisions``). The margin is the distance of the
  predicted value from the threshold.
"""

import abc
import enum
import functools
import math
import statistics
from collections.abc import Callable
from typing import override

import attrs

from ebaml.core.custom_types import ClassLabel
from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.graphs.model_kinds import BinaryClassifier, ModelKind, Regressor
from ebaml.local.decisions import Decision, Side, Threshold


class OutputKind(enum.Enum):
    """What the output of the model is, so that the text can name it."""

    PROBABILITY = "probability"
    VALUE = "value"


@attrs.frozen
class PredictionFrame:
    """One prediction, seen as a margin.

    Attributes
    ----------
    side: Side
        The decision that holds, its opposite, and the margin.
    output: float
        The output of the model for the decision.
    lower: float
        The lower end of the confidence interval of ``output``.
    upper: float
        The upper end of the confidence interval of ``output``.
    output_after: Callable[[float], float]
        The output of the model for the opposite decision, from the margin after some changes.

    """

    side: Side
    output: float
    lower: float
    upper: float
    output_after: Callable[[float], float]


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


class LocalKind(abc.ABC):
    """The kind of a model, as the local explanations need it."""

    @property
    @abc.abstractmethod
    def output_kind(self) -> OutputKind:
        """What the output of the model is."""

    @abc.abstractmethod
    def frame(
        self,
        score: float,
        score_std: float,
        *,
        intercept: float,
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        """Turn the score of one prediction into a margin.

        Parameters
        ----------
        score: float
            The intercept and the scores of the features, on the scale of the link.
        score_std: float
            The standard deviation of ``score``.
        intercept: float
            The intercept of the model.
        confidence_level: float
            The confidence level of the interval of the output, in ``[0, 1)``.
        decision: Decision | None
            The decision that the user chose, or ``None`` for the default of the kind.

        Returns
        -------
        PredictionFrame
            The decision, the output and its interval.

        Raises
        ------
        OptionRangeError
            If the kind has no use for a decision and the user gave one.

        """


def _z(confidence_level: float) -> float:
    return statistics.NormalDist().inv_cdf((1 + confidence_level) / 2)


@attrs.frozen
class BinaryLocal(LocalKind):
    """A binary classifier. The decision is the predicted class, and the threshold is the log-odds zero."""

    classes: tuple[ClassLabel, ClassLabel]

    @property
    @override
    def output_kind(self) -> OutputKind:
        return OutputKind.PROBABILITY

    @override
    def frame(
        self,
        score: float,
        score_std: float,
        *,
        intercept: float,
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        if decision is not None:
            msg = "A classifier has its predicted class as the decision. Only a regressor accepts a decision."
            raise OptionRangeError(msg)
        sign = 1 if score > 0 else -1
        z = _z(confidence_level)
        # the probability of the predicted class is the sigmoid of the logit seen from that class, which is sign * logit
        lower, upper = sorted((_sigmoid(sign * (score - z * score_std)), _sigmoid(sign * (score + z * score_std))))
        return PredictionFrame(
            side=Side(self.classes[int(score > 0)], self.classes[int(score <= 0)], sign, 0.0),
            output=_sigmoid(sign * score),
            lower=lower,
            upper=upper,
            output_after=lambda new_margin: _sigmoid(-new_margin),  # the opposite class has the opposite log-odds
        )


@attrs.frozen
class RegressorLocal(LocalKind):
    """A regressor. The default decision is whether the value is above the average prediction, the intercept."""

    @property
    @override
    def output_kind(self) -> OutputKind:
        return OutputKind.VALUE

    @override
    def frame(
        self,
        score: float,
        score_std: float,
        *,
        intercept: float,
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        side = (decision or Threshold(intercept)).side(score)
        spread = _z(confidence_level) * score_std
        return PredictionFrame(
            side=side,
            output=score,
            lower=score - spread,
            upper=score + spread,
            output_after=lambda new_margin: side.threshold + side.sign * new_margin,
        )


@functools.singledispatch
def local_kind(kind: ModelKind) -> LocalKind:
    """Select the local explanations of a model kind.

    Parameters
    ----------
    kind: ModelKind
        The kind of the model.

    Raises
    ------
    UnsupportedModelError
        If the local explanations do not support the kind.

    """
    msg = f"Local explanations need a binary classifier or a regressor. Got a {type(kind).__name__}."
    raise UnsupportedModelError(msg)


@local_kind.register(BinaryClassifier)
def _binary(kind: BinaryClassifier) -> LocalKind:
    return BinaryLocal(kind.classes)


@local_kind.register(Regressor)
def _regressor(_kind: Regressor) -> LocalKind:
    return RegressorLocal()
