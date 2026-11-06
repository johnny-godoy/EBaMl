"""The search for a sufficient reason when the decision has several margins.

A decision with one margin has a sufficient reason that sorting finds exactly (``ebaml.local.facts``). A decision with
several margins, such as "inside an interval", needs every margin to stay positive at the same time. Fixing a feature
raises the worst case of each margin by a different amount, so the smallest set of features is a covering problem, and
sorting does not solve it in general.

A ``ReasonSearch`` class takes the numbers of the problem and returns a set of features:

- ``worst[margin]`` is the worst case of the margin when no feature is fixed.
- ``gains[margin][feature]`` is how much the worst case of the margin rises when the feature keeps its value.

A set of features is *sufficient* when ``worst[margin]`` plus the gains of the set is positive for every margin.

- ``Minimal`` returns a set from which no feature can be removed. It does not say that no smaller set exists.
- ``ExactOrMinimal`` searches the sets by size, so its answer is the smallest set. Above a limit on the number of
  features, it returns what ``Minimal`` returns.

To add a search, write a new subclass of ``ReasonSearch``.
"""

import abc
import itertools
from collections.abc import Sequence
from typing import override

import attrs


@attrs.frozen
class Selection:
    """The features that a search chose.

    Attributes
    ----------
    indices: list[int]
        The positions of the features, largest guaranteed gain first.
    guaranteed_margin: float
        The smallest margin in the worst case when these features keep their values.
    minimum: bool
        True if no smaller set is sufficient. False if the set is only minimal: no feature can be removed.

    """

    indices: list[int]
    guaranteed_margin: float
    minimum: bool


def _guaranteed(worst: Sequence[float], gains: Sequence[Sequence[float]], chosen: Sequence[int]) -> float:
    """Return the smallest margin in the worst case when the chosen features keep their values.

    Returns
    -------
    float
        The smallest over the margins of the worst case plus the gains of the chosen features.

    """
    return min(
        margin_worst + sum(margin_gains[feature] for feature in chosen)
        for margin_worst, margin_gains in zip(worst, gains, strict=True)
    )


def by_smallest_gain(gains: Sequence[Sequence[float]]) -> list[int]:
    """Order the features by the smallest gain over the margins, largest first.

    A feature that helps one margin and does nothing for another one is no help for the claim, so the smallest gain is
    the measure.

    Returns
    -------
    list[int]
        The positions of the features.

    """
    feature_count = len(gains[0])
    return sorted(range(feature_count), key=lambda feature: -min(margin_gains[feature] for margin_gains in gains))


class ReasonSearch(abc.ABC):
    """A way to choose the features of a sufficient reason for a decision with several margins."""

    @abc.abstractmethod
    def select(self, worst: Sequence[float], gains: Sequence[Sequence[float]]) -> Selection | None:
        """Choose a sufficient set of features.

        Parameters
        ----------
        worst: Sequence[float]
            The worst case of each margin when no feature is fixed.
        gains: Sequence[Sequence[float]]
            For each margin, how much each feature raises the worst case of the margin when it is fixed.

        Returns
        -------
        Selection | None
            The features, or None if even all of them do not make every margin positive.

        """


@attrs.frozen
class Minimal(ReasonSearch):
    """Add features by the smallest gain over the margins, then remove each feature that is not needed.

    The set is minimal: no feature can be removed. A set of at most one feature is also the minimum, because the empty
    set is sufficient or it is not.

    Examples
    --------
    >>> Minimal().select([-1.0, -1.0], [[2.0, 0.0], [0.0, 2.0]])
    Selection(indices=[0, 1], guaranteed_margin=1.0, minimum=False)

    """

    @override
    def select(self, worst: Sequence[float], gains: Sequence[Sequence[float]]) -> Selection | None:
        order = by_smallest_gain(gains)
        chosen: list[int] = []
        for feature in [None, *order]:
            if feature is not None:
                chosen.append(feature)
            if _guaranteed(worst, gains, chosen) > 0:
                break
        else:
            return None
        # remove the features of the smallest gain first. A feature that was needed stays needed after a removal,
        # because a removal lowers the margins.
        for feature in reversed(list(chosen)):
            without = [other for other in chosen if other != feature]
            if _guaranteed(worst, gains, without) > 0:
                chosen = without
        return Selection(chosen, _guaranteed(worst, gains, chosen), minimum=len(chosen) <= 1)


@attrs.frozen
class ExactOrMinimal(ReasonSearch):
    """Search the sets of features by size, and use ``Minimal`` above a limit on the number of features.

    The search tries every set of 0 features, then of 1 feature, and so on, so the first sufficient set is the smallest.
    The cost grows as 2 to the power of the number of features, which is why the limit exists. Above the limit the
    result is only minimal, and ``Selection.minimum`` says so.

    Attributes
    ----------
    max_features: int
        The most features for which the search runs. The default is 12, which is at most 4096 sets.

    Examples
    --------
    >>> ExactOrMinimal().select([-1.0, -1.0], [[2.0, 0.0], [0.0, 2.0]])
    Selection(indices=[0, 1], guaranteed_margin=1.0, minimum=True)

    """

    max_features: int = attrs.field(default=12, validator=attrs.validators.ge(0))

    @override
    def select(self, worst: Sequence[float], gains: Sequence[Sequence[float]]) -> Selection | None:
        order = by_smallest_gain(gains)
        if len(order) > self.max_features:
            return Minimal().select(worst, gains)
        for size in range(len(order) + 1):
            for chosen in itertools.combinations(order, size):
                margin = _guaranteed(worst, gains, chosen)
                if margin > 0:
                    return Selection(list(chosen), margin, minimum=True)
        return None
