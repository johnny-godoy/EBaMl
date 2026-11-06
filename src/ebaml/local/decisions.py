"""The decision of a regressor that a local explanation keeps or changes.

A classifier has a decision already: its predicted class. A regressor predicts a number, so the user says which
question about the number to explain. Each question is one ``Decision`` class. A ``Decision`` looks at the predicted
value and returns the ``Side`` of the value: the claim that holds, and its margins. Each margin is a number that is a
sum of one term per feature, and the claim holds while every margin is positive.

- ``Threshold`` asks whether the value is above a number.
- ``Interval`` asks whether the value is at most the low end, inside the interval, or above the high end.

To add a question, write a new subclass of ``Decision``. No other code changes.
"""

import abc
import math
from typing import override

import attrs

from ebaml.core.custom_types import ClassLabel
from ebaml.core.exceptions import OptionRangeError


@attrs.frozen
class Margin:
    """One condition of a claim, as a number that is positive while the claim holds.

    The margin is ``sign * (score - threshold)``. A margin of zero belongs to the claim only when ``sign`` is ``-1``, as
    the logit zero belongs to the negative class.

    Attributes
    ----------
    sign: int
        ``1`` if the condition says that the score is high, ``-1`` if it says that the score is low.
    threshold: float
        The score at which the condition stops to hold.
    opposite: ClassLabel
        The claim after this margin becomes negative.

    """

    sign: int
    threshold: float
    opposite: ClassLabel

    def value(self, score: float) -> float:
        """Return the margin of a score.

        Parameters
        ----------
        score: float
            The score of the prediction.

        Returns
        -------
        float
            The margin, positive while the condition holds.

        """
        return self.sign * (score - self.threshold)


@attrs.frozen
class Side:
    """The claim about one prediction. It holds while every margin is positive.

    Attributes
    ----------
    label: ClassLabel
        The claim that holds for the prediction.
    margins: tuple[Margin, ...]
        The conditions of the claim. Most claims have one margin. The claim "inside an interval" has two.

    """

    label: ClassLabel
    margins: tuple[Margin, ...]

    def binding(self, score: float) -> Margin:
        """Return the margin that is closest to failing.

        Parameters
        ----------
        score: float
            The score of the prediction.

        Returns
        -------
        Margin
            The margin with the smallest value.

        """
        return min(self.margins, key=lambda margin: margin.value(score))


def _number(value: float) -> str:
    return f"{value:.4g}"


def _finite(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        msg = f"The decision needs a finite number. Got {value}."
        raise OptionRangeError(msg)
    return number


class Decision(abc.ABC):
    """A question about the predicted value of a regressor. The explanations keep the answer to it."""

    @abc.abstractmethod
    def side(self, value: float) -> Side:
        """Return the claim that holds for a predicted value.

        Parameters
        ----------
        value: float
            The predicted value.

        Returns
        -------
        Side
            The claim and its margins.

        """


@attrs.frozen
class Threshold(Decision):
    """Ask whether the predicted value is above a number.

    Attributes
    ----------
    value: float
        The number. A predicted value equal to it is not above it.

    Examples
    --------
    >>> Threshold(10.0).side(12.5)
    Side(label='above 10', margins=(Margin(sign=1, threshold=10.0, opposite='at most 10'),))
    >>> Threshold(10.0).side(10.0).label
    'at most 10'

    """

    value: float = attrs.field(converter=_finite)

    @override
    def side(self, value: float) -> Side:
        above, at_most = f"above {_number(self.value)}", f"at most {_number(self.value)}"
        if value > self.value:
            return Side(above, (Margin(1, self.value, at_most),))
        return Side(at_most, (Margin(-1, self.value, above),))


@attrs.frozen
class Interval(Decision):
    """Ask whether the predicted value is at most the low end, inside the interval, or above the high end.

    The interval is open at the low end and closed at the high end, as ``Threshold`` is: a value equal to a number is
    not above it. The answer "inside" has two margins, one for each end.

    Attributes
    ----------
    low: float
        The lower end. A predicted value equal to it is at most the low end.
    high: float
        The upper end. A predicted value equal to it is inside.

    Examples
    --------
    >>> Interval(10.0, 20.0).side(25.0).label
    'above 20'
    >>> Interval(10.0, 20.0).side(5.0).label
    'at most 10'
    >>> Interval(10.0, 20.0).side(15.0).label
    'above 10 and at most 20'

    """

    low: float = attrs.field(converter=_finite)
    high: float = attrs.field(converter=_finite)

    def __attrs_post_init__(self) -> None:
        """Check that the interval is not empty.

        Raises
        ------
        OptionRangeError
            If ``low`` is not below ``high``.

        """
        if not self.low < self.high:
            msg = f"The interval needs low < high. Got {self.low} and {self.high}."
            raise OptionRangeError(msg)

    @override
    def side(self, value: float) -> Side:
        low, high = _number(self.low), _number(self.high)
        if value <= self.low:
            return Side(f"at most {low}", (Margin(-1, self.low, f"above {low}"),))
        if value > self.high:
            return Side(f"above {high}", (Margin(1, self.high, f"at most {high}"),))
        return Side(
            f"above {low} and at most {high}",
            (Margin(1, self.low, f"at most {low}"), Margin(-1, self.high, f"above {high}")),
        )
