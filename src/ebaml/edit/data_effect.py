"""The effect of an edit on rows of data: how many predictions change, and by how much.

The count needs no labels. It uses the prediction of the model itself (``predict`` and ``predict_proba``), so it does
not assume a link function. A row with a missing value stays in the count, because an edit of the missing bin affects
only such rows. The count is not a measure of quality: it says how many rows an edit reaches, not whether the edit is
good.
"""

import functools
from typing import cast

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ClassLabel, ExplainableBoostingModel, Frame
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.graphs.model_kinds import BinaryClassifier, ModelKind, MulticlassClassifier, model_kind


@attrs.frozen
class ClassChange:
    """The number of rows whose predicted class goes from ``from_class`` to ``to_class`` after an edit."""

    from_class: ClassLabel
    to_class: ClassLabel
    row_count: int


@attrs.frozen
class RowChanges:
    """How an edit changes the predictions for a set of rows.

    ``value`` is the probability of the positive class for a binary classifier, and the predicted value for a
    regressor. For a multiclass classifier, it is the probability of the class that the model predicts for the row
    before the edit, so a negative change means that the model is less sure of that class. The counts of classes are
    ``None`` for a regressor, which has no class. The positive class is the second class of ``classes_``.

    Attributes
    ----------
    row_count: int
        The number of rows that the count covers.
    rows_with_changed_value: int
        The number of rows whose value differs after the edit.
    rows_with_changed_class: int | None
        The number of rows whose predicted class differs after the edit.
    largest_value_change: float
        The largest absolute change of the value in one row.
    mean_value_change: float
        The mean change of the value over all rows. A positive number means that the edit raises the value on average.
    rows_to_positive_class: int | None
        The number of rows whose predicted class becomes the positive class of a binary classifier.
    rows_to_negative_class: int | None
        The number of rows whose predicted class stops being the positive class of a binary classifier.
    class_changes: tuple[ClassChange, ...]
        For a multiclass classifier, the number of rows that go from each class to each other class, for each pair
        with one row or more. It is empty for the other kinds.

    """

    row_count: int
    rows_with_changed_value: int
    rows_with_changed_class: int | None
    largest_value_change: float
    mean_value_change: float
    rows_to_positive_class: int | None = None
    rows_to_negative_class: int | None = None
    class_changes: tuple[ClassChange, ...] = ()


def _value_counts(value_change: npt.NDArray[np.float64], changed_classes: int | None) -> RowChanges:
    """Count the rows whose value changes, with the number of rows whose class changes.

    Returns
    -------
    RowChanges
        The counts, with no count of the moves between classes.

    """
    return RowChanges(
        row_count=len(value_change),
        rows_with_changed_value=int(np.count_nonzero(value_change)),
        rows_with_changed_class=changed_classes,
        largest_value_change=float(np.abs(value_change).max(initial=0.0)),
        mean_value_change=float(value_change.mean()) if len(value_change) else 0.0,
    )


@functools.singledispatch
def _changes(
    _kind: ModelKind,
    before: ExplainableBoostingModel,
    after: ExplainableBoostingModel,
    rows: npt.ArrayLike,
) -> RowChanges:
    """Count the changes of a regressor. Each kind of classifier registers its own count.

    Returns
    -------
    RowChanges
        The counts, with no class.

    """
    value_change = np.asarray(after.predict(rows), dtype=float) - np.asarray(before.predict(rows), dtype=float)
    return _value_counts(value_change, None)


@_changes.register
def _binary_changes(
    kind: BinaryClassifier,
    before: ExplainableBoostingModel,
    after: ExplainableBoostingModel,
    rows: npt.ArrayLike,
) -> RowChanges:
    old, new = cast(ExplainableBoostingClassifier, before), cast(ExplainableBoostingClassifier, after)
    value_change = (
        np.asarray(new.predict_proba(rows), dtype=float)[:, 1] - np.asarray(old.predict_proba(rows), dtype=float)[:, 1]
    )
    positive = kind.classes[1]
    was_positive: npt.NDArray[np.bool_] = np.asarray(old.predict(rows)) == positive
    is_positive: npt.NDArray[np.bool_] = np.asarray(new.predict(rows)) == positive
    return attrs.evolve(
        _value_counts(value_change, int(np.count_nonzero(was_positive != is_positive))),
        rows_to_positive_class=int(np.count_nonzero(is_positive & ~was_positive)),
        rows_to_negative_class=int(np.count_nonzero(was_positive & ~is_positive)),
    )


@_changes.register
def _multiclass_changes(
    kind: MulticlassClassifier,
    before: ExplainableBoostingModel,
    after: ExplainableBoostingModel,
    rows: npt.ArrayLike,
) -> RowChanges:
    old, new = cast(ExplainableBoostingClassifier, before), cast(ExplainableBoostingClassifier, after)
    old_probabilities = np.asarray(old.predict_proba(rows), dtype=float)
    new_probabilities = np.asarray(new.predict_proba(rows), dtype=float)
    # interpret predicts the class with the highest probability, so the argmax is the column of the predicted class
    old_classes, new_classes = old_probabilities.argmax(axis=1), new_probabilities.argmax(axis=1)
    row_numbers = np.arange(len(old_classes))
    value_change = new_probabilities[row_numbers, old_classes] - old_probabilities[row_numbers, old_classes]
    class_count = len(kind.classes)
    moves = np.bincount(old_classes * class_count + new_classes, minlength=class_count**2).reshape(class_count, -1)
    class_changes = tuple(
        ClassChange(kind.classes[from_column], kind.classes[to_column], int(moves[from_column, to_column]))
        for from_column, to_column in np.argwhere(moves).tolist()
        if from_column != to_column
    )
    counts = _value_counts(value_change, int(np.count_nonzero(old_classes != new_classes)))
    return attrs.evolve(counts, class_changes=class_changes)


def row_changes(before: ExplainableBoostingModel, after: ExplainableBoostingModel, data: Frame) -> RowChanges:
    """Count the rows whose prediction differs between two models.

    Parameters
    ----------
    before : ExplainableBoostingModel
        The model that the edit started from.
    after : ExplainableBoostingModel
        The model that the edit made. It has the same features as ``before``.
    data : Frame
        The rows. The frame needs a column for each feature of the model, and may have more columns.

    Returns
    -------
    RowChanges
        The counts and the size of the change.

    Raises
    ------
    InvalidInstanceError
        If the frame lacks a column for a feature of the model.

    """
    names = list(before.feature_names_in_)
    missing = [name for name in names if name not in data.columns]
    if missing:
        msg = f"The data has no column for these features: {missing}."
        raise InvalidInstanceError(msg)
    # interpret reads the columns of a frame by position, so select them in the order of the model
    rows = data.select(names).to_native()
    return _changes(model_kind(before), before, after, rows)
