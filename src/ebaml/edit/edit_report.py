"""The report of an edit: what changed, how many training rows it reaches, and what it leaves out.

The report holds numbers and the names of the bins. The text claims no more than the numbers show, so it never says
that an edit is safer or better.
"""

import itertools

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EBaMlError
from ebaml.edit.edits import EditOutcome
from ebaml.edit.model_view import TermView, interaction_names, interval_label, term_view
from ebaml.graphs.ebm_graph import EBMGraph

_TEXT_CHANGE_LIMIT = 10


@attrs.frozen
class BinChange:
    """The score of one bin before and after an edit.

    ``position`` is the position in ``term_scores_`` of the model after the edit. After an edit that changes the bins
    (``SplitBin``, ``Rebin``), one change covers the range of values in ``label`` where the old and the new bins
    overlap.
    """

    position: int
    label: str
    before: float
    after: float


@attrs.frozen
class EditReport:
    """What an edit changes in a model.

    ``graph_before`` and ``graph_after`` hold the graph text without the preamble. They are ``None`` for an edit that
    removes terms, and for a feature that the graph text does not cover (an ordinal feature). The missing bin and the
    unseen bin are not in the graph text, so ``changes`` lists them too.

    ``weight_share`` is the share of the training weight that the changed bins hold: the number of training rows that
    the edit reaches, divided by the number of training rows. It needs no data. A ``bins_changed`` of 0 means that
    the edit set every selected score to the value that it had.

    ``base_revision`` is the revision that the edit started from. ``revision`` is the revision that ``apply`` made, and
    it is ``None`` for a ``preview``, which makes none.
    """

    feature: str | None
    edit_type: str
    description: str
    changes: tuple[BinChange, ...]
    largest_change: float
    covered_range: str | None
    weight_share: float
    weighted_mean_shift: float
    intercept_change: float
    interactions_still_using: tuple[str, ...]
    removed_terms: tuple[str, ...]
    graph_before: str | None
    graph_after: str | None
    base_revision: str
    revision: str | None = None

    @property
    def bins_changed(self) -> int:
        """The number of bins whose score changed."""
        return len(self.changes)

    def to_text(self) -> str:
        """Write the report as plain text.

        Returns
        -------
        str
            The description, the changed bins, the training weight that they hold, and the pair terms that still use
            the feature.

        """
        lines = [self.description + "."]
        if self.removed_terms:
            lines.append(f"Removed terms: {', '.join(self.removed_terms)}.")
        else:
            noun = "bin" if self.bins_changed == 1 else "bins"
            lines.append(
                f"{self.bins_changed} {noun} changed. The largest change of a score is {self.largest_change:.4g}.",
            )
            lines.extend(
                f"  {change.label}: {change.before:.4g} to {change.after:.4g}"
                for change in self.changes[:_TEXT_CHANGE_LIMIT]
            )
            if self.bins_changed > _TEXT_CHANGE_LIMIT:
                lines.append(f"  and {self.bins_changed - _TEXT_CHANGE_LIMIT} more bins")
            lines.append(
                f"The changed bins hold {self.weight_share:.1%} of the training weight."
                f" The weighted mean of the graph moves by {self.weighted_mean_shift:.4g}"
                f" and the intercept by {self.intercept_change:.4g}.",
            )
        if self.interactions_still_using:
            names = ", ".join(self.interactions_still_using)
            lines.append(f"The pair terms {names} still use the feature, and the edit does not change them.")
        return "\n".join(lines)


def _graph_text(ebm: ExplainableBoostingModel, feature: str) -> str | None:
    try:
        feature_index = list(ebm.feature_names_in_).index(feature)
        return EBMGraph.from_ebm(ebm, feature_index).to_text(include_description=False)
    except EBaMlError:
        # the graph text covers continuous and nominal features, and an ordinal feature raises
        return None


def _intercept(ebm: ExplainableBoostingModel) -> float:
    return float(np.ravel(ebm.intercept_)[0])


@attrs.frozen
class _Comparison:
    """The changed bins of one graph, the largest change, and the training weight of the changed bins."""

    changes: tuple[BinChange, ...]
    largest_change: float
    changed_weight: float


def _compare_positions(before: TermView, after: TermView) -> _Comparison:
    """Compare two graphs with the same bins, position by position.

    Returns
    -------
    _Comparison
        One change for each bin whose score differs.

    """
    scores_before, scores_after = before.scores, after.scores
    changed = np.flatnonzero(scores_after != scores_before)
    changes = tuple(
        BinChange(
            int(position),
            after.label(int(position)),
            float(scores_before[position]),
            float(scores_after[position]),
        )
        for position in changed
    )
    return _Comparison(
        changes,
        float(np.abs(scores_after - scores_before).max(initial=0.0)),
        float(before.weights[changed].sum()),
    )


def _cuts(view: TermView) -> npt.NDArray[np.float64]:
    return np.array([left for left, _ in view.bin_edges()[1:]], dtype=float)


def _range_weight(view: TermView, bin_number: int, left: float, right: float) -> float:
    """Estimate the training weight of a range of one bin: the weight of the bin, shared by width.

    Returns
    -------
    float
        The weight, which is 0 for a range outside the bin.

    """
    bin_left, bin_right = view.bin_edges()[bin_number]
    covered = max(0.0, min(right, bin_right) - max(left, bin_left)) / (bin_right - bin_left)
    return float(view.weights[bin_number + 1]) * covered


def _bin_of(cuts: npt.NDArray[np.float64], value: float) -> int:
    # interpret bins a value by the cuts only: a value below the first cut falls in the first bin, at any distance
    return int(np.searchsorted(cuts, value, side="right"))


def _compare_outer_bins(before: TermView, after: TermView) -> _Comparison:
    """Compare the missing bins and the unseen bins of two graphs.

    Returns
    -------
    _Comparison
        One change for each of the two bins whose score differs.

    """
    changes: list[BinChange] = []
    differences = [0.0]
    changed_weight = 0.0
    for old_position, new_position in ((0, 0), (before.unseen_position, after.unseen_position)):
        old, new = float(before.scores[old_position]), float(after.scores[new_position])
        differences.append(abs(new - old))
        if new != old:
            changes.append(BinChange(new_position, after.label(new_position), old, new))
            changed_weight += float(before.weights[old_position])
    return _Comparison(tuple(changes), max(differences), changed_weight)


def _compare_ranges(before: TermView, after: TermView) -> _Comparison:
    """Compare two graphs of a continuous feature whose bins differ, on the ranges where the old and new bins overlap.

    The weight of a range is the weight of its old bin, shared by width, as an estimate. A range outside the old bounds
    has no weight, because the model saw no value there.

    Returns
    -------
    _Comparison
        One change for each range whose score differs, and for the missing bin and the unseen bin.

    """
    cuts_before, cuts_after = _cuts(before), _cuts(after)
    boundaries = sorted({edge for pair in (*before.bin_edges(), *after.bin_edges()) for edge in pair})
    outer = _compare_outer_bins(before, after)
    changes = list(outer.changes)
    largest, changed_weight = outer.largest_change, outer.changed_weight
    for left, right in itertools.pairwise(boundaries):
        old_bin, new_bin = _bin_of(cuts_before, (left + right) / 2), _bin_of(cuts_after, (left + right) / 2)
        old, new = float(before.scores[old_bin + 1]), float(after.scores[new_bin + 1])
        largest = max(largest, abs(new - old))
        if new != old:
            changes.append(BinChange(new_bin + 1, interval_label(left, right), old, new))
            changed_weight += _range_weight(before, old_bin, left, right)
    return _Comparison(tuple(sorted(changes, key=lambda change: change.position)), largest, changed_weight)


def _compare(before: TermView, after: TermView) -> _Comparison:
    same_bins = before.real_bin_count == after.real_bin_count and (
        before.feature_type != "continuous"
        or [left for left, _ in before.bin_edges()[1:]] == [left for left, _ in after.bin_edges()[1:]]
    )
    return _compare_positions(before, after) if same_bins else _compare_ranges(before, after)


def build_report(
    before: ExplainableBoostingModel,
    after: ExplainableBoostingModel,
    edit_type: str,
    outcome: EditOutcome,
    base_revision: str,
) -> EditReport:
    """Compare a model before and after an edit.

    Parameters
    ----------
    before : ExplainableBoostingModel
        The model that the edit started from.
    after : ExplainableBoostingModel
        The model that the edit made.
    edit_type : str
        The name of the class of the edit.
    outcome : EditOutcome
        What the edit reported.
    base_revision : str
        The revision that ``before`` is, as ``EditSession.revision_info`` names it.

    Returns
    -------
    EditReport
        The report.

    """
    if outcome.removed_terms or outcome.feature is None:
        return EditReport(
            feature=outcome.feature,
            edit_type=edit_type,
            description=outcome.description,
            changes=(),
            largest_change=0.0,
            covered_range=None,
            weight_share=0.0,
            weighted_mean_shift=0.0,
            intercept_change=_intercept(after) - _intercept(before),
            interactions_still_using=(),
            removed_terms=outcome.removed_terms,
            graph_before=None,
            graph_after=None,
            base_revision=base_revision,
        )
    view_before = term_view(before, outcome.feature)
    view_after = term_view(after, outcome.feature)
    comparison = _compare(view_before, view_after)
    weights_before, weights_after = view_before.weights, view_after.weights
    total_weight = float(weights_before.sum())
    mean_before = float(np.average(view_before.scores, weights=weights_before)) if total_weight > 0 else 0.0
    mean_after = float(np.average(view_after.scores, weights=weights_after)) if weights_after.sum() > 0 else 0.0
    return EditReport(
        feature=outcome.feature,
        edit_type=edit_type,
        description=outcome.description,
        changes=comparison.changes,
        largest_change=comparison.largest_change,
        covered_range=view_after.range_text(outcome.positions) if outcome.positions else None,
        weight_share=comparison.changed_weight / total_weight if total_weight > 0 else 0.0,
        weighted_mean_shift=mean_after - mean_before,
        intercept_change=_intercept(after) - _intercept(before),
        interactions_still_using=interaction_names(after, outcome.feature),
        removed_terms=(),
        graph_before=_graph_text(before, outcome.feature),
        graph_after=_graph_text(after, outcome.feature),
        base_revision=base_revision,
    )
