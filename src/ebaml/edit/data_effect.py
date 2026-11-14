"""The effect of an edit on rows of data: how many predictions change, and by how much.

The count needs no labels. It uses the prediction of the model itself (``predict`` and ``predict_proba``), so it does
not assume a link function. A row with a missing value stays in the count, because an edit of the missing bin affects
only such rows. The count is not a measure of quality: it says how many rows an edit reaches, not whether the edit is
good.
"""

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ExplainableBoostingModel, Frame
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import InvalidInstanceError


@attrs.frozen
class RowChanges:
    """How an edit changes the predictions for a set of rows.

    ``value`` is the probability of the positive class for a classifier, and the predicted value for a regressor.
    ``rows_with_changed_class`` is ``None`` for a regressor, which has no class.

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

    """

    row_count: int
    rows_with_changed_value: int
    rows_with_changed_class: int | None
    largest_value_change: float
    mean_value_change: float


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

    def values(model: ExplainableBoostingModel) -> npt.NDArray[np.float64]:
        # a classifier gives the probability of its positive class, and a regressor gives the predicted value
        if isinstance(model, ExplainableBoostingClassifier):
            return np.asarray(model.predict_proba(rows), dtype=float)[:, 1]
        return np.asarray(model.predict(rows), dtype=float)

    value_change = values(after) - values(before)
    class_change = (
        int(np.count_nonzero(np.asarray(after.predict(rows)) != np.asarray(before.predict(rows))))
        if isinstance(before, ExplainableBoostingClassifier)
        else None
    )
    return RowChanges(
        row_count=len(value_change),
        rows_with_changed_value=int(np.count_nonzero(value_change)),
        rows_with_changed_class=class_change,
        largest_value_change=float(np.abs(value_change).max(initial=0.0)),
        mean_value_change=float(value_change.mean()) if len(value_change) else 0.0,
    )
