"""The bins that an edit changes.

A selection names bins by their bin number, their category or their place on the x axis. It never names a position of
``term_scores_``, because that position is off by one and moves when the bins change. ``resolve`` turns a selection
into positions, and it raises a ``SelectionError`` with the fix steps for a selection that does not fit the feature.
"""

import abc
import math
from collections.abc import Collection
from typing import override

import attrs

from ebaml.core.exceptions import SelectionError
from ebaml.edit.model_view import MISSING_POSITION, TermView


class Selection(abc.ABC):
    """A group of bins of one feature."""

    @abc.abstractmethod
    def resolve(self, view: TermView) -> tuple[int, ...]:
        """Find the positions in ``term_scores_`` of the selected bins.

        Parameters
        ----------
        view : TermView
            The term of the feature.

        Returns
        -------
        tuple[int, ...]
            The positions, in increasing order, with no repeat.

        Raises
        ------
        SelectionError
            If the selection does not fit the feature.

        """


@attrs.frozen
class Bins(Selection):
    """The real bins of a continuous or ordinal feature from bin number ``first`` to ``last``, both included.

    The bin numbers start at 0 and exclude the missing bin and the unseen bin, so they are the same as in the graph
    text.
    """

    first: int
    last: int

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        if not view.ordered:
            msg = (
                f"Bins needs a continuous or ordinal feature, but {view.feature_name!r} is {view.feature_type}."
                " Use Categories."
            )
            raise SelectionError(msg)
        if not 0 <= self.first <= self.last < view.real_bin_count:
            msg = (
                f"Bins({self.first}, {self.last}) is not a range of the bins of {view.feature_name!r}."
                f" The bin numbers go from 0 to {view.real_bin_count - 1}, and first must not be above last."
            )
            raise SelectionError(msg)
        return tuple(range(self.first + 1, self.last + 2))


def _unique(names: Collection[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


@attrs.frozen
class Categories(Selection):
    """The bins of some categories of a nominal feature."""

    names: tuple[str, ...] = attrs.field(converter=_unique)

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        if view.feature_type != "nominal":
            msg = f"Categories needs a nominal feature, but {view.feature_name!r} is {view.feature_type}. Use Bins."
            raise SelectionError(msg)
        if not self.names:
            msg = "Categories needs at least one category."
            raise SelectionError(msg)
        known = view.categories()
        for name in self.names:
            if name not in known:
                msg = f"The feature {view.feature_name!r} has no category {name!r}. The categories are {known}."
                raise SelectionError(msg)
        return tuple(sorted(known.index(name) + 1 for name in self.names))


@attrs.frozen
class MissingBin(Selection):
    """The bin of the missing values. It is the first row of the term, and the graph does not show it."""

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        return (MISSING_POSITION,)


@attrs.frozen
class UnseenBin(Selection):
    """The bin of the values that the model did not see in training. It is the last row of the term."""

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        return (view.unseen_position,)


@attrs.frozen
class WholeGraph(Selection):
    """Every real bin of the graph. The missing bin and the unseen bin are not part of it."""

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        return tuple(range(1, view.real_bin_count + 1))


@attrs.frozen
class ValueInterval(Selection):
    """Every bin of a continuous feature that overlaps the interval from ``low`` to ``high``.

    A bound of ``None`` is open. The report of the edit names the bins that the interval resolved to.
    """

    low: float | None = None
    high: float | None = None

    @override
    def resolve(self, view: TermView) -> tuple[int, ...]:
        if view.feature_type != "continuous":
            msg = f"ValueInterval needs a continuous feature, but {view.feature_name!r} is {view.feature_type}."
            raise SelectionError(msg)
        low = -math.inf if self.low is None else self.low
        high = math.inf if self.high is None else self.high
        if math.isnan(low) or math.isnan(high) or low > high:
            msg = f"ValueInterval({self.low}, {self.high}) needs numbers, and low must not be above high."
            raise SelectionError(msg)
        edges = view.bin_edges()
        last_bin = len(edges) - 1
        # a bin holds its left edge and not its right edge, except the last bin, which holds both
        positions = tuple(
            bin_number + 1
            for bin_number, (left, right) in enumerate(edges)
            if left <= high and (right > low or (bin_number == last_bin and right >= low))
        )
        if not positions:
            msg = (
                f"No bin of {view.feature_name!r} overlaps the interval from {self.low} to {self.high}."
                f" The values go from {edges[0][0]:g} to {edges[-1][1]:g}."
            )
            raise SelectionError(msg)
        return positions
