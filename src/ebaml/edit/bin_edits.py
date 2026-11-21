"""The edits that change the bins of a continuous feature: split a bin, rebin a range, and extend the range.

GAM Changer (Wang et al., KDD 2022, MIT licence) changes bins in its ``equal-interpolate`` and ``equal-regression``
tools only, and it cannot split or extend the first or the last bin (its issue 1). These edits do both. Each one writes
through ``model_write.replace_bins`` or ``model_write.set_bounds``, which change the cuts, the scores, the bags, the
standard deviations and the training weights together. If a pair term reads the same cuts as the main effect, the main
effect gets its own cuts first, so the pair term does not change.

A bin edit changes the bins of every class of a multiclass model, because the classes share the bins. So it takes no
class: each new bin gets the scores of every class.

The training weight of a new bin is an estimate. The model keeps the weight of each bin, not the values of the rows,
so an edit that cuts a bin shares its weight by width, as if the values were spread evenly over the bin.
"""

from typing import override

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureType
from ebaml.core.exceptions import EditError
from ebaml.edit import model_write
from ebaml.edit.edits import Edit, EditOutcome
from ebaml.edit.model_view import TermView, term_view
from ebaml.edit.selections import Selection


def _number(value: float) -> str:
    return format(value, ".6g")


def _twice(values: npt.NDArray[np.float64], position: int) -> npt.NDArray[np.float64]:
    """Give two copies of the row of one bin: one score, or one score per class of a multiclass model.

    Returns
    -------
    npt.NDArray[np.float64]
        The two rows.

    """
    return np.repeat(values[[position]], 2, axis=0)


def _continuous_view(ebm: ExplainableBoostingModel, feature: str, edit_name: str) -> TermView:
    model_write.check_editable(ebm)
    view = term_view(ebm, feature)
    if view.feature_type is not FeatureType.CONTINUOUS:
        msg = (
            f"{edit_name} needs a continuous feature, because only its bins have edges, but {feature!r} is"
            f" {view.feature_type}."
        )
        raise EditError(msg)
    return view


@attrs.frozen
class SplitBin(Edit):
    """Split the bin that holds the value ``at`` into two bins, at that value.

    Both new bins keep the score, the bags and the standard deviation of the old bin, so the split changes no
    prediction. A later edit can then give each part its own score. The weight of the old bin is shared by width, as an
    estimate. To split beyond the range that the model saw, extend the range first with ``ExtendRange``.
    """

    feature: str
    at: float

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        view = _continuous_view(ebm, self.feature, "SplitBin")
        edges = view.bin_edges()
        low, high = edges[0][0], edges[-1][1]
        containing = [number for number, (left, right) in enumerate(edges) if left < self.at < right]
        if not containing:
            reason = (
                "is already an edge of a bin"
                if low <= self.at <= high
                else f"is outside the range {_number(low)} to {_number(high)}. Extend the range first with ExtendRange"
            )
            msg = f"SplitBin cannot split {self.feature!r} at {self.at}: the value {reason}."
            raise EditError(msg)
        bin_number = containing[0]
        position = bin_number + 1
        left, right = edges[bin_number]
        left_share = (self.at - left) / (right - left)
        bags = np.asarray(ebm.bagged_scores_[view.term], dtype=float)[:, [position, position]]
        model_write.replace_bins(
            ebm,
            view.feature_index,
            view.term,
            model_write.BinReplacement(
                first=position,
                last=position,
                inner_cuts=[self.at],
                scores=_twice(view.scores, position),
                weights=view.weights[position] * np.array([left_share, 1 - left_share]),
                deviations=_twice(np.asarray(ebm.standard_deviations_[view.term], dtype=float), position),
                bags=bags,
            ),
        )
        text = f"Split the bin {_number(left)} to {_number(right)} of {self.feature} at {_number(self.at)}"
        return EditOutcome(self.feature, (position, position + 1), text)


def _overlaps(
    old_edges: npt.NDArray[np.float64],
    new_edges: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Give the share of each old bin that each new bin covers.

    Returns
    -------
    npt.NDArray[np.float64]
        One row for each new bin and one column for each old bin.

    """
    lefts = np.maximum.outer(new_edges[:-1], old_edges[:-1])
    rights = np.minimum.outer(new_edges[1:], old_edges[1:])
    return np.clip(rights - lefts, 0.0, None) / np.diff(old_edges)


def _equal_bins(view: TermView, first: int, last: int, bin_count: int) -> model_write.BinReplacement:
    """Describe ``bin_count`` bins of equal width that replace the real bins from ``first`` to ``last``.

    Returns
    -------
    model_write.BinReplacement
        The new bins, with the weights shared by width, the mixed scores, no fitted uncertainty, and one copy of the
        scores in each bag.

    """
    edges = view.bin_edges()
    old_edges = np.array([edges[first - 1][0], *(right for _, right in edges[first - 1 : last])], dtype=float)
    new_edges = np.linspace(old_edges[0], old_edges[-1], bin_count + 1)
    overlaps = _overlaps(old_edges, new_edges)
    old_weights = view.weights[first : last + 1]
    taken = overlaps * old_weights
    # a new bin that takes no weight averages the old scores by the width that it covers
    mix = np.where(taken.sum(axis=1, keepdims=True) > 0, taken, overlaps)
    # the scores of a multiclass model have one column for each class, and the mix applies to each column
    new_scores = (mix / mix.sum(axis=1, keepdims=True)) @ view.scores[first : last + 1]
    bag_count = np.asarray(view.ebm.bagged_scores_[view.term]).shape[0]
    return model_write.BinReplacement(
        first=first,
        last=last,
        inner_cuts=new_edges[1:-1].tolist(),
        scores=new_scores,
        weights=overlaps @ old_weights,
        deviations=np.zeros_like(new_scores),
        bags=np.broadcast_to(new_scores, (bag_count, *new_scores.shape)).copy(),
    )


@attrs.frozen
class Rebin(Edit):
    """Replace the selected bins by ``bin_count`` bins of equal width that cover the same range of values.

    Follow it with ``Interpolate`` or ``FitLine`` on the new bins to do what the ``equal-interpolate`` and
    ``equal-regression`` tools of GAM Changer do. The weight of an old bin is shared by width, as an estimate. The score
    of a new bin is the mean of the scores of the old bins that it covers, weighted by the weight that it takes from
    each, or by the width that it covers if it takes no weight. A new bin is a bin that a person made, so it has no
    fitted uncertainty: its standard deviation is 0. The rebin changes the predictions of the rows in the selected bins.
    """

    feature: str
    selection: Selection
    bin_count: int

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        view = _continuous_view(ebm, self.feature, "Rebin")
        positions = self.selection.resolve(view)
        if not view.is_real(positions):
            msg = (
                f"Rebin needs real bins, and the selection holds the missing bin or the unseen bin of {self.feature!r}."
            )
            raise EditError(msg)
        if self.bin_count < 1:
            msg = f"Rebin needs a bin_count of 1 or more, but it is {self.bin_count}."
            raise EditError(msg)
        first, last = positions[0], positions[-1]
        # the view reads the model as it is, so describe the old bins before the edit replaces them
        selected = view.describe_positions(positions)
        replacement = _equal_bins(view, first, last, self.bin_count)
        model_write.replace_bins(ebm, view.feature_index, view.term, replacement)
        width = (view.bin_edges()[first + self.bin_count - 2][1] - view.bin_edges()[first - 1][0]) / self.bin_count
        text = f"Replace {selected} by {self.bin_count} bins of equal width ({_number(width)} each)"
        return EditOutcome(self.feature, tuple(range(first, first + self.bin_count)), text)


def _copy_bin(
    ebm: ExplainableBoostingModel,
    view: TermView,
    position: int,
    *,
    new_first: bool,
) -> model_write.BinReplacement:
    """Describe two bins that replace one: the old bin, and a copy of its score with no training weight.

    Returns
    -------
    model_write.BinReplacement
        The two bins, the copy first if ``new_first``, without the cut between them.

    """
    order = [1, 0] if new_first else [0, 1]
    deviations = np.asarray(ebm.standard_deviations_[view.term], dtype=float)
    return model_write.BinReplacement(
        first=position,
        last=position,
        inner_cuts=[],
        scores=_twice(view.scores, position),
        weights=np.array([view.weights[position], 0.0])[order],
        deviations=_twice(deviations, position),
        bags=np.asarray(ebm.bagged_scores_[view.term], dtype=float)[:, [position, position]],
    )


@attrs.frozen
class ExtendRange(Edit):
    """Add a bin beyond the lowest or the highest value that the model saw, to give such values their own score.

    ``interpret`` bins a value by the cuts only, so a value above the last cut falls in the last bin, at any distance,
    and a value below the first cut falls in the first bin. ``ExtendRange(feature, high=110)`` adds a cut at the old
    high bound and a bin from there to 110. A value at the old high bound or above then falls in the new bin. The new
    bin has the score of the old last bin, so the edit changes no prediction, and no training weight, because the
    model saw no value there. Set its score with a later edit. ``low`` adds a bin below the old low bound in the same
    way. GAM Changer cannot do this (its issue 1).
    """

    feature: str
    low: float | None = None
    high: float | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        view = _continuous_view(ebm, self.feature, "ExtendRange")
        old_low, old_high = view.bin_edges()[0][0], view.bin_edges()[-1][1]
        if (self.low is None and self.high is None) or not (
            (self.low is None or self.low < old_low) and (self.high is None or self.high > old_high)
        ):
            msg = (
                f"ExtendRange needs a low bound below {_number(old_low)} or a high bound above {_number(old_high)},"
                f" the range of {self.feature!r}."
            )
            raise EditError(msg)
        added: list[str] = []
        if self.high is not None:
            last = view.real_bin_count
            replacement = attrs.evolve(_copy_bin(ebm, view, last, new_first=False), inner_cuts=[old_high])
            model_write.replace_bins(ebm, view.feature_index, view.term, replacement)
            added.append(f"{_number(old_high)} to {_number(self.high)}")
        if self.low is not None:
            view = term_view(ebm, self.feature)
            replacement = attrs.evolve(_copy_bin(ebm, view, 1, new_first=True), inner_cuts=[old_low])
            model_write.replace_bins(ebm, view.feature_index, view.term, replacement)
            added.append(f"{_number(self.low)} to {_number(old_low)}")
        model_write.set_bounds(
            ebm,
            view.feature_index,
            old_low if self.low is None else self.low,
            old_high if self.high is None else self.high,
        )
        view = term_view(ebm, self.feature)
        positions = ((1,) if self.low is not None else ()) + ((view.real_bin_count,) if self.high is not None else ())
        text = f"Add the bins {' and '.join(added)} to {self.feature}, with the scores of the outer bins"
        return EditOutcome(self.feature, positions, text)
