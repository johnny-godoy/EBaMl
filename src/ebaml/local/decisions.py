"""The decision of a regressor that a local explanation keeps or changes.

A classifier has a decision already: its predicted class. A regressor predicts a number, so the user says which
question about the number to explain. Each question is one ``Decision`` class. A ``Decision`` looks at the predicted
value and returns the ``Side`` of the value: the claim that holds, and how the claim is a margin of the score.

- ``Threshold`` asks whether the value is above a number.
- ``Interval`` asks whether the value is below, inside or above two numbers.

To add a question, write a new subclass of ``Decision``. No other code changes.
"""

import abc
import math
from typing import override

import attrs

from ebaml.core.custom_types import ClassLabel
from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError


@attrs.frozen
class Side:
    """The claim about one prediction, as a margin of the score.

    The margin is ``sign * (score - threshold)``. The claim holds while the margin is positive, and a margin of zero
    belongs to the claim only when ``sign`` is ``-1``, as the logit zero belongs to the negative class.

    Attributes
    ----------
    label: ClassLabel
        The claim that holds for the prediction.
    opposite: ClassLabel
        The claim after the margin becomes negative.
    sign: int
        ``1`` if the claim says that the score is high, ``-1`` if it says that the score is low.
    threshold: float
        The score at which the claim stops to hold.

    """

    label: ClassLabel
    opposite: ClassLabel
    sign: int
    threshold: float


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
            The claim, its opposite and the margin.

        Raises
        ------
        UnsupportedModelError
            If the explanations cannot keep the answer for this value.

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
    Side(label='above 10', opposite='at most 10', sign=1, threshold=10.0)
    >>> Threshold(10.0).side(10.0).label
    'at most 10'

    """

    value: float = attrs.field(converter=_finite)

    @override
    def side(self, value: float) -> Side:
        above, at_most = f"above {_number(self.value)}", f"at most {_number(self.value)}"
        if value > self.value:
            return Side(above, at_most, 1, self.value)
        return Side(at_most, above, -1, self.value)


@attrs.frozen
class Interval(Decision):
    """Ask whether the predicted value is below, inside or above two numbers.

    An explanation keeps the answer "below" or "above". The answer "inside" has two margins, and the explanations do not
    support it.

    Attributes
    ----------
    low: float
        The lower end. A predicted value equal to it is inside.
    high: float
        The upper end. A predicted value equal to it is inside.

    Examples
    --------
    >>> Interval(10.0, 20.0).side(25.0)
    Side(label='above 20', opposite='at most 20', sign=1, threshold=20.0)
    >>> Interval(10.0, 20.0).side(5.0).label
    'below 10'

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
        if value < self.low:
            return Side(f"below {_number(self.low)}", f"at least {_number(self.low)}", -1, self.low)
        if value > self.high:
            return Side(f"above {_number(self.high)}", f"at most {_number(self.high)}", 1, self.high)
        msg = (
            f"The predicted value {_number(value)} is inside the interval [{_number(self.low)}, {_number(self.high)}]."
            " The explanations of the answer 'inside' are not supported yet. Use a Threshold, or an interval that"
            " does not contain the prediction."
        )
        raise UnsupportedModelError(msg)
