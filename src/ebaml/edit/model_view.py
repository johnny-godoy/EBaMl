"""The main-effect term of one feature, seen as a row of bins that an edit addresses.

``term_scores_`` has one row for the missing bin first and one row for the unseen bin last, so the position of a real
bin in ``term_scores_`` is one more than its position in the graph. A *position* in this package is an index of
``term_scores_``. A *bin number* is an index among the real bins, counted from 0, as in the graph text.
"""

import operator
from collections.abc import Sequence
from typing import cast

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import SelectionError
from ebaml.graphs.terms import main_term

ORDERED_TYPES = ("continuous", "ordinal")
MISSING_POSITION = 0
_LABEL_LIMIT = 6


def _format_number(value: float) -> str:
    """Write a number with at most six significant digits.

    Returns
    -------
    str
        The text of the number.

    """
    return format(value, ".6g")


@attrs.frozen(eq=False)
class TermView:
    """The main-effect term of one feature in one EBM.

    The view reads the model on demand, so it shows the scores that the model holds when you read them.
    """

    ebm: ExplainableBoostingModel
    feature_index: int
    feature_name: str
    feature_type: str
    term: int
    real_bin_count: int

    @property
    def ordered(self) -> bool:
        """Whether the bins have an order that an edit can use (a continuous or an ordinal feature)."""
        return self.feature_type in ORDERED_TYPES

    @property
    def unseen_position(self) -> int:
        """The position of the unseen bin, which is the last row of the term."""
        return self.real_bin_count + 1

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """The scores of every row of the term, the missing bin first and the unseen bin last."""
        return np.asarray(self.ebm.term_scores_[self.term], dtype=float)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """The training weight of every row of the term."""
        return np.asarray(self.ebm.bin_weights_[self.term], dtype=float)

    def is_real(self, positions: Sequence[int]) -> bool:
        """Tell whether every position is a real bin, and not the missing bin or the unseen bin.

        Returns
        -------
        bool
            ``True`` if no position is the missing bin or the unseen bin.

        """
        return all(1 <= position <= self.real_bin_count for position in positions)

    def categories(self) -> list[str]:
        """List the categories of a nominal or ordinal feature, in the order of the bins.

        Returns
        -------
        list[str]
            The category of each real bin.

        Raises
        ------
        SelectionError
            If the feature is continuous.

        """
        if self.feature_type == "continuous":
            msg = f"The feature {self.feature_name!r} is continuous, so its bins have no categories."
            raise SelectionError(msg)
        names = cast(dict[str, int], self.ebm.bins_[self.feature_index][0])
        return [name for name, _ in sorted(names.items(), key=operator.itemgetter(1))]

    def _cuts(self) -> list[float]:
        cuts: list[float] = np.asarray(self.ebm.bins_[self.feature_index][0], dtype=float).tolist()
        return cuts

    # nosemgrep: ebaml-return-attrs-not-container -- each item is a homogeneous interval of one bin
    def bin_edges(self) -> list[tuple[float, float]]:
        """List the left and the right edge of each real bin of a continuous feature.

        The first left edge and the last right edge are the bounds that the model saw.

        Returns
        -------
        list[tuple[float, float]]
            One edge pair for each real bin.

        """
        low, high = (float(bound) for bound in self.ebm.feature_bounds_[self.feature_index])
        cuts = self._cuts()
        return list(zip([low, *cuts], [*cuts, high], strict=True))

    def left_edges(self) -> npt.NDArray[np.float64]:
        """Give the place of each real bin on the x axis: its left edge, or its number for an ordinal feature.

        Returns
        -------
        npt.NDArray[np.float64]
            One value for each real bin.

        Raises
        ------
        SelectionError
            If the feature has no order.

        """
        if self.feature_type == "continuous":
            return np.array([left for left, _ in self.bin_edges()], dtype=float)
        if self.feature_type == "ordinal":
            return np.arange(self.real_bin_count, dtype=float)
        msg = f"The feature {self.feature_name!r} is {self.feature_type}, so its bins have no order."
        raise SelectionError(msg)

    def label(self, position: int) -> str:
        """Name the bin at a position, as the report shows it.

        Parameters
        ----------
        position : int
            The index in ``term_scores_``.

        Returns
        -------
        str
            ``missing``, ``unseen``, a pair of edges for a continuous bin, or the category.

        """
        if position == MISSING_POSITION:
            return "missing"
        if position == self.unseen_position:
            return "unseen"
        if self.feature_type == "continuous":
            left, right = self.bin_edges()[position - 1]
            return f"{_format_number(left)} to {_format_number(right)}"
        return self.categories()[position - 1]

    def describe_positions(self, positions: Sequence[int]) -> str:
        """Describe a group of bins for the text of an edit.

        Parameters
        ----------
        positions : Sequence[int]
            The positions in ``term_scores_``, in increasing order.

        Returns
        -------
        str
            For example ``6 bins of age (95.5 to 105)``.

        """
        if len(positions) == 1 and not self.is_real(positions):
            return f"the {self.label(positions[0])} bin of {self.feature_name}"
        count = len(positions)
        noun = "bin" if count == 1 else "bins"
        return f"{count} {noun} of {self.feature_name} ({self.range_text(positions)})"

    def range_text(self, positions: Sequence[int]) -> str:
        """Name the range that a group of bins covers.

        Parameters
        ----------
        positions : Sequence[int]
            The positions in ``term_scores_``, in increasing order.

        Returns
        -------
        str
            The edges of a continuous range, or the names of the bins.

        """
        if self.feature_type == "continuous" and self.is_real(positions):
            edges = self.bin_edges()
            first_left = edges[positions[0] - 1][0]
            last_right = edges[positions[-1] - 1][1]
            return f"{_format_number(first_left)} to {_format_number(last_right)}"
        labels = [self.label(position) for position in positions]
        if len(labels) > _LABEL_LIMIT:
            return f"{labels[0]} to {labels[-1]}"
        return ", ".join(labels)


def term_view(ebm: ExplainableBoostingModel, feature: str) -> TermView:
    """Find the main-effect term of a feature by its name.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The fitted EBM.
    feature : str
        The name of the feature, as in ``feature_names_in_``.

    Returns
    -------
    TermView
        The view of the term.

    Raises
    ------
    SelectionError
        If the model has no feature with this name.

    """
    names = list(ebm.feature_names_in_)
    if feature not in names:
        msg = f"The model has no feature {feature!r}. The features are {names}."
        raise SelectionError(msg)
    feature_index = names.index(feature)
    term = main_term(ebm, feature_index)
    real_bin_count = len(ebm.term_scores_[term]) - 2
    return TermView(ebm, feature_index, feature, ebm.feature_types_in_[feature_index], term, real_bin_count)


def interaction_names(ebm: ExplainableBoostingModel, feature_name: str) -> tuple[str, ...]:
    """List the pair terms that still use a feature.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The fitted EBM.
    feature_name : str
        The name of the feature.

    Returns
    -------
    tuple[str, ...]
        The name of each term with two or more features, one of them this feature.

    """
    feature_index = list(ebm.feature_names_in_).index(feature_name)
    return tuple(
        name
        for name, features in zip(ebm.term_names_, ebm.term_features_, strict=True)
        if len(features) > 1 and feature_index in features
    )
