"""The report of an edit: what changed, how many training rows it reaches, and what it leaves out.

The report holds numbers and the names of the bins. The text claims no more than the numbers show, so it never says
that an edit is safer or better. A multiclass model has one score per class in each bin, so each change names its
class, and the report holds the shift of the weighted mean and of the intercept of each class.
"""

import itertools

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ClassLabel, ExplainableBoostingModel, FeatureType
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
    overlap. ``class_label`` is the class of the score in a multiclass model, and ``None`` for a model with one score
    per bin.
    """

    position: int
    label: str
    before: float
    after: float
    class_label: ClassLabel | None = None


@attrs.frozen
class ClassShift:
    """The shift of the weighted mean of a graph and of the intercept for one class of a multiclass model."""

    class_label: ClassLabel
    weighted_mean_shift: float
    intercept_change: float


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

    A multiclass model has one weighted mean and one intercept per class. ``class_shifts`` then holds the shifts of
    each class, and ``weighted_mean_shift`` and ``intercept_change`` hold the shift with the largest size over the
    classes, with its sign. ``class_shifts`` is empty for a model with one score per bin.
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
    class_shifts: tuple[ClassShift, ...] = ()

    @property
    def bins_changed(self) -> int:
        """The number of bins whose score changed, in one class or more of a multiclass model."""
        return len({(change.position, change.label) for change in self.changes})

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
                f"  {change.label}{_class_suffix(change.class_label)}: {change.before:.4g} to {change.after:.4g}"
                for change in self.changes[:_TEXT_CHANGE_LIMIT]
            )
            if len(self.changes) > _TEXT_CHANGE_LIMIT:
                lines.append(f"  and {len(self.changes) - _TEXT_CHANGE_LIMIT} more changes")
            lines.append(
                f"The changed bins hold {self.weight_share:.1%} of the training weight."
                f" The weighted mean of the graph moves by {self.weighted_mean_shift:.4g}"
                f" and the intercept by {self.intercept_change:.4g}.",
            )
            lines.extend(
                f"  Class {shift.class_label!r}: the weighted mean moves by {shift.weighted_mean_shift:.4g}"
                f" and the intercept by {shift.intercept_change:.4g}."
                for shift in self.class_shifts
            )
        if self.interactions_still_using:
            names = ", ".join(self.interactions_still_using)
            lines.append(f"The pair terms {names} still use the feature, and the edit does not change them.")
        return "\n".join(lines)


def _class_suffix(class_label: ClassLabel | None) -> str:
    return "" if class_label is None else f" (class {class_label!r})"


def _graph_text(ebm: ExplainableBoostingModel, feature: str) -> str | None:
    try:
        feature_index = list(ebm.feature_names_in_).index(feature)
        return EBMGraph.from_ebm(ebm, feature_index).to_text(include_description=False)
    except EBaMlError:
        # the graph text covers continuous and nominal features, and an ordinal feature raises
        return None


def _intercepts(ebm: ExplainableBoostingModel) -> npt.NDArray[np.float64]:
    """Read the intercept as one value per score column: one value, or one per class of a multiclass model.

    Returns
    -------
    npt.NDArray[np.float64]
        The intercepts.

    """
    return np.atleast_1d(np.asarray(ebm.intercept_, dtype=float))


def _class_labels(view: TermView) -> list[ClassLabel | None]:
    """Name the score columns of a term: ``None`` for a model with one score per bin, else the classes.

    Returns
    -------
    list[ClassLabel | None]
        One label for each column of ``_columns``.

    """
    if view.all_scores.ndim == 1:
        return [None]
    labels: list[ClassLabel | None] = np.asarray(view.ebm.classes_).tolist()
    return labels


def _columns(view: TermView) -> npt.NDArray[np.float64]:
    """Give the scores of a term with one column per class, and one column for a model with one score per bin.

    Returns
    -------
    npt.NDArray[np.float64]
        One row for each row of the term.

    """
    scores = view.all_scores
    return scores.reshape(len(scores), -1)


@attrs.frozen
class _Comparison:
    """The changed bins of one graph, the largest change, and the training weight of the changed bins."""

    changes: tuple[BinChange, ...]
    largest_change: float
    changed_weight: float


def _row_changes(
    position: int,
    label: str,
    old: npt.NDArray[np.float64],
    new: npt.NDArray[np.float64],
    classes: list[ClassLabel | None],
) -> list[BinChange]:
    """List the changes of the scores of one bin, one for each class whose score differs.

    Returns
    -------
    list[BinChange]
        The changes, in the order of the classes.

    """
    return [
        BinChange(position, label, float(before), float(after), class_label)
        for before, after, class_label in zip(old.tolist(), new.tolist(), classes, strict=True)
        if before != after
    ]


def _compare_positions(before: TermView, after: TermView) -> _Comparison:
    """Compare two graphs with the same bins, position by position.

    Returns
    -------
    _Comparison
        One change for each bin and class whose score differs.

    """
    scores_before, scores_after = _columns(before), _columns(after)
    classes = _class_labels(after)
    changed = np.flatnonzero((scores_after != scores_before).any(axis=1))
    changes = tuple(
        change
        for position in changed.tolist()
        for change in _row_changes(
            position,
            after.label(position),
            scores_before[position],
            scores_after[position],
            classes,
        )
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
    scores_before, scores_after, classes = _columns(before), _columns(after), _class_labels(after)
    for old_position, new_position in ((0, 0), (before.unseen_position, after.unseen_position)):
        old, new = scores_before[old_position], scores_after[new_position]
        differences.append(float(np.abs(new - old).max()))
        bin_changes = _row_changes(new_position, after.label(new_position), old, new, classes)
        changes.extend(bin_changes)
        if bin_changes:
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
    scores_before, scores_after, classes = _columns(before), _columns(after), _class_labels(after)
    for left, right in itertools.pairwise(boundaries):
        old_bin, new_bin = _bin_of(cuts_before, (left + right) / 2), _bin_of(cuts_after, (left + right) / 2)
        old, new = scores_before[old_bin + 1], scores_after[new_bin + 1]
        largest = max(largest, float(np.abs(new - old).max()))
        range_changes = _row_changes(new_bin + 1, interval_label(left, right), old, new, classes)
        changes.extend(range_changes)
        if range_changes:
            changed_weight += _range_weight(before, old_bin, left, right)
    return _Comparison(tuple(sorted(changes, key=lambda change: change.position)), largest, changed_weight)


def _compare(before: TermView, after: TermView) -> _Comparison:
    same_bins = before.real_bin_count == after.real_bin_count and (
        before.feature_type is not FeatureType.CONTINUOUS
        or [left for left, _ in before.bin_edges()[1:]] == [left for left, _ in after.bin_edges()[1:]]
    )
    return _compare_positions(before, after) if same_bins else _compare_ranges(before, after)


def _weighted_means(view: TermView) -> npt.NDArray[np.float64]:
    """Compute the weighted mean of each score column of a term, or 0 for a term with no training weight.

    Returns
    -------
    npt.NDArray[np.float64]
        One mean for each column of ``_columns``.

    """
    weights = view.weights
    scores = _columns(view)
    if weights.sum() <= 0:
        return np.zeros(scores.shape[1])
    return np.asarray(np.average(scores, axis=0, weights=weights), dtype=float)


def _largest(values: npt.NDArray[np.float64]) -> float:
    """Give the value with the largest size, with its sign.

    Returns
    -------
    float
        The value.

    """
    return float(values[np.argmax(np.abs(values))])


def _class_shifts(
    after: ExplainableBoostingModel,
    mean_shifts: npt.NDArray[np.float64],
    intercept_changes: npt.NDArray[np.float64],
) -> tuple[ClassShift, ...]:
    """Give the shifts of each class of a multiclass model, and nothing for a model with one score per bin.

    Returns
    -------
    tuple[ClassShift, ...]
        One shift for each class.

    """
    if len(intercept_changes) == 1:
        return ()
    classes: list[ClassLabel] = np.asarray(after.classes_).tolist()
    return tuple(
        ClassShift(class_label, float(mean_shift), float(intercept_change))
        for class_label, mean_shift, intercept_change in zip(classes, mean_shifts, intercept_changes, strict=True)
    )


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
    intercept_changes = _intercepts(after) - _intercepts(before)
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
            intercept_change=_largest(intercept_changes),
            interactions_still_using=(),
            removed_terms=outcome.removed_terms,
            graph_before=None,
            graph_after=None,
            base_revision=base_revision,
            class_shifts=_class_shifts(after, np.zeros_like(intercept_changes), intercept_changes),
        )
    view_before = term_view(before, outcome.feature)
    view_after = term_view(after, outcome.feature)
    comparison = _compare(view_before, view_after)
    total_weight = float(view_before.weights.sum())
    mean_shifts = _weighted_means(view_after) - _weighted_means(view_before)
    return EditReport(
        feature=outcome.feature,
        edit_type=edit_type,
        description=outcome.description,
        changes=comparison.changes,
        largest_change=comparison.largest_change,
        covered_range=view_after.range_text(outcome.positions) if outcome.positions else None,
        weight_share=comparison.changed_weight / total_weight if total_weight > 0 else 0.0,
        weighted_mean_shift=_largest(mean_shifts),
        intercept_change=_largest(intercept_changes),
        interactions_still_using=interaction_names(after, outcome.feature),
        removed_terms=(),
        graph_before=_graph_text(before, outcome.feature),
        graph_after=_graph_text(after, outcome.feature),
        base_revision=base_revision,
        class_shifts=_class_shifts(after, mean_shifts, intercept_changes),
    )
