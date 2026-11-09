"""What each kind of EBM tells the local explanations about one prediction.

The algorithms of ``ebaml.local.facts`` work on margins: each is a sum of one term per feature, and the decision holds
while every margin is positive. This module turns the prediction of a model into its margins. One ``LocalKind`` class
does it for each model kind (``ebaml.graphs.model_kinds``), so ``facts`` never tests the kind of the model.
``local_kind`` selects the class.

- ``BinaryLocal`` keeps the predicted class. The margin is the log-odds of that class.
- ``RegressorLocal`` keeps the answer to a ``Decision`` (``ebaml.local.decisions``). The margin is the distance of the
  predicted value from the threshold.
- ``MulticlassLocal`` keeps the predicted class ``c``. It has one margin for each other class ``j``: the score of ``c``
  minus the score of ``j``. The class holds while every margin is positive.
"""

import abc
import enum
import functools
import math
import statistics
from collections.abc import Callable, Sequence
from typing import NoReturn, override

import attrs
import numpy as np
import numpy.typing as npt
import scipy

from ebaml.core.custom_types import ClassLabel
from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.graphs.model_kinds import BinaryClassifier, ModelKind, MulticlassClassifier, Regressor
from ebaml.local.decisions import Decision, Margin, Side, Threshold


class OutputKind(enum.Enum):
    """What the output of the model is, so that the text can name it."""

    PROBABILITY = "probability"
    CLASS_PROBABILITY = "class_probability"
    VALUE = "value"


@attrs.frozen
class Outcome:
    """What the model says after some changes.

    Attributes
    ----------
    decision: ClassLabel
        The decision after the changes.
    output: float
        The output of the model for that decision.

    """

    decision: ClassLabel
    output: float


@attrs.frozen
class PredictionFrame:
    """One prediction, seen as a margin.

    Attributes
    ----------
    side: Side
        The decision that holds, its opposite, and the margins.
    column: int
        The output of the model that ``Prediction.score`` reports: the predicted class of a classifier with several
        outputs, otherwise the only output.
    output: float
        The output of the model for the decision.
    lower: float
        The lower end of the confidence interval of ``output``.
    upper: float
        The upper end of the confidence interval of ``output``.
    after: Callable[[Margin, Sequence[float]], Outcome]
        The decision and the output of the model after some changes, from the failed margin and the score of each
        output after the changes.

    """

    side: Side
    column: int
    output: float
    lower: float
    upper: float
    after: Callable[[Margin, Sequence[float]], Outcome]


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
        scores: Sequence[float],
        score_stds: Sequence[float],
        *,
        intercepts: Sequence[float],
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        """Turn the scores of one prediction into margins.

        Parameters
        ----------
        scores: Sequence[float]
            For each output of the model, the intercept and the scores of the features, on the scale of the link.
        score_stds: Sequence[float]
            The standard deviation of each score.
        intercepts: Sequence[float]
            The intercept of each output of the model.
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

    @abc.abstractmethod
    def target_margins(self, target: ClassLabel, frame: PredictionFrame) -> tuple[Margin, ...]:
        """Return the margins that must hold for a chosen class to be the predicted class.

        Parameters
        ----------
        target: ClassLabel
            The class that the counterfactual must reach.
        frame: PredictionFrame
            The prediction.

        Returns
        -------
        tuple[Margin, ...]
            One margin for each other class.

        Raises
        ------
        OptionRangeError
            If the kind has no use for a target class, the class is unknown, or it is the predicted class already.

        """


def _reject_target(target: ClassLabel) -> NoReturn:
    msg = f"A target class needs a multiclass classifier. Got the target {target!r} for another kind of model."
    raise OptionRangeError(msg)


def _reject_decision(decision: Decision | None) -> None:
    """Raise an error if a classifier gets a decision.

    Raises
    ------
    OptionRangeError
        If ``decision`` is not ``None``, because only a regressor accepts one.

    """
    if decision is not None:
        msg = "A classifier has its predicted class as the decision. Only a regressor accepts a decision."
        raise OptionRangeError(msg)


def _z(confidence_level: float) -> float:
    return statistics.NormalDist().inv_cdf((1 + confidence_level) / 2)


@attrs.frozen
class BinaryLocal(LocalKind):
    """A binary classifier. The decision is the predicted class, and the threshold is the log-odds zero."""

    classes: tuple[ClassLabel, ClassLabel]

    @override
    def target_margins(self, target: ClassLabel, frame: PredictionFrame) -> tuple[Margin, ...]:
        _reject_target(target)

    @property
    @override
    def output_kind(self) -> OutputKind:
        return OutputKind.PROBABILITY

    @override
    def frame(
        self,
        scores: Sequence[float],
        score_stds: Sequence[float],
        *,
        intercepts: Sequence[float],
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        _reject_decision(decision)
        score, score_std = scores[0], score_stds[0]
        sign = 1 if score > 0 else -1
        z = _z(confidence_level)
        # the probability of the predicted class is the sigmoid of the logit seen from that class, which is sign * logit
        lower, upper = sorted((_sigmoid(sign * (score - z * score_std)), _sigmoid(sign * (score + z * score_std))))
        return PredictionFrame(
            side=Side(self.classes[int(score > 0)], (Margin(sign, 0.0, self.classes[int(score <= 0)]),)),
            column=0,
            output=_sigmoid(sign * score),
            lower=lower,
            upper=upper,
            # the opposite class has the opposite log-odds
            after=lambda margin, new_scores: Outcome(margin.opposite, _sigmoid(-margin.sign * new_scores[0])),
        )


@attrs.frozen
class RegressorLocal(LocalKind):
    """A regressor. The default decision is whether the value is above the average prediction, the intercept."""

    @override
    def target_margins(self, target: ClassLabel, frame: PredictionFrame) -> tuple[Margin, ...]:
        _reject_target(target)

    @property
    @override
    def output_kind(self) -> OutputKind:
        return OutputKind.VALUE

    @override
    def frame(
        self,
        scores: Sequence[float],
        score_stds: Sequence[float],
        *,
        intercepts: Sequence[float],
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        score = scores[0]
        side = (decision or Threshold(intercepts[0])).side(score)
        spread = _z(confidence_level) * score_stds[0]
        return PredictionFrame(
            side=side,
            column=0,
            output=score,
            lower=score - spread,
            upper=score + spread,
            after=lambda margin, new_scores: Outcome(margin.opposite, new_scores[0]),
        )


def _probabilities(scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(scipy.special.softmax(np.asarray(scores, dtype=float)), dtype=np.float64)


@attrs.frozen
class MulticlassLocal(LocalKind):
    """A multiclass classifier. The decision is the predicted class, and each other class is a rival.

    The class ``c`` holds while the score of ``c`` is above the score of every other class ``j``. Each difference
    ``s_c - s_j`` is a margin. EBMs store no covariance between the scores of two classes, so the standard deviation
    of a margin is the sum of the two standard deviations. For any correlation, the sum is an upper bound.
    The proof is in ``proofs/papers/local-explanations/paper.typ``.
    """

    classes: tuple[ClassLabel, ...]

    @property
    @override
    def output_kind(self) -> OutputKind:
        return OutputKind.CLASS_PROBABILITY

    @override
    def target_margins(self, target: ClassLabel, frame: PredictionFrame) -> tuple[Margin, ...]:
        if target not in self.classes:
            msg = f"The target class {target!r} is not a class of the model. The classes are {list(self.classes)}."
            raise OptionRangeError(msg)
        column = self.classes.index(target)
        if column == frame.column:
            msg = f"The target class {target!r} is the predicted class already, so no change is needed."
            raise OptionRangeError(msg)
        return tuple(
            Margin(1, 0.0, rival_class, (column, rival))
            for rival, rival_class in enumerate(self.classes)
            if rival != column
        )

    @override
    def frame(
        self,
        scores: Sequence[float],
        score_stds: Sequence[float],
        *,
        intercepts: Sequence[float],
        confidence_level: float,
        decision: Decision | None,
    ) -> PredictionFrame:
        _reject_decision(decision)
        winner = int(np.argmax(scores))  # the first class on a tie
        spread = _z(confidence_level) * np.asarray(score_stds, dtype=float)
        against = np.asarray(scores, dtype=float) + spread
        against[winner] = scores[winner] - spread[winner]
        towards = np.asarray(scores, dtype=float) - spread
        towards[winner] = scores[winner] + spread[winner]
        rivals = [rival for rival, _ in enumerate(self.classes) if rival != winner]
        margins = tuple(Margin(1, 0.0, self.classes[rival], (winner, rival)) for rival in rivals)

        def after(_margin: Margin, new_scores: Sequence[float]) -> Outcome:
            new_winner = int(np.argmax(new_scores))
            return Outcome(self.classes[new_winner], float(_probabilities(new_scores)[new_winner]))

        return PredictionFrame(
            side=Side(self.classes[winner], margins),
            column=winner,
            output=float(_probabilities(scores)[winner]),
            lower=float(_probabilities(against)[winner]),
            upper=float(_probabilities(towards)[winner]),
            after=after,
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
    msg = f"Local explanations need a classifier or a regressor. Got a {type(kind).__name__}."
    raise UnsupportedModelError(msg)


@local_kind.register(BinaryClassifier)
def _binary(kind: BinaryClassifier) -> LocalKind:
    return BinaryLocal(kind.classes)


@local_kind.register(MulticlassClassifier)
def _multiclass(kind: MulticlassClassifier) -> LocalKind:
    return MulticlassLocal(kind.classes)


@local_kind.register(Regressor)
def _regressor(_kind: Regressor) -> LocalKind:
    return RegressorLocal()
