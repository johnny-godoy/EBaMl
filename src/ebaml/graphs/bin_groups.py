"""The groups of neighbouring bins that ``EBMGraph.simplify`` merges.

The merge is greedy. The first bin opens a group. Each next bin joins the open group if no score of the bin differs
from the score of the first bin of the group by more than the threshold, and opens a new group otherwise.
"""

import numpy as np
import numpy.typing as npt


def _single_column_starts(values: list[float], threshold: float) -> list[int]:
    """Find the first bin of each group of a graph with one score for each bin.

    Parameters
    ----------
    values: list[float]
        The score of each bin, as Python floats, because NumPy calls on single bins are slow in this loop.
    threshold: float
        The largest change of the score inside a group.

    Returns
    -------
    list[int]
        The index of the first bin of each group, in order.

    """
    starts = [0]
    anchor = values[0]
    for bin_index, value in enumerate(values[1:], start=1):
        # `not <=` and not `>`, so that a NaN score opens a group, as the test of several columns does
        if not abs(anchor - value) <= threshold:
            starts.append(bin_index)
            anchor = value
    return starts


def _several_column_starts(rows: list[list[float]], threshold: float) -> list[int]:
    """Find the first bin of each group of a graph with one score for each class in each bin.

    Parameters
    ----------
    rows: list[list[float]]
        The scores of each bin, one row for each bin.
    threshold: float
        The largest change of each score inside a group.

    Returns
    -------
    list[int]
        The index of the first bin of each group, in order.

    """
    starts = [0]
    anchor = rows[0]
    for bin_index, row in enumerate(rows[1:], start=1):
        if not max(abs(first - score) for first, score in zip(anchor, row, strict=True)) <= threshold:
            starts.append(bin_index)
            anchor = row
    return starts


def group_starts(score_rows: npt.NDArray[np.float64], threshold: float) -> list[int]:
    """Find the first bin of each group that the greedy merge makes.

    Parameters
    ----------
    score_rows: npt.NDArray[np.float64]
        The scores, one row for each bin and one column for each class. A graph with one score has one column.
    threshold: float
        The largest change of a score inside a group.

    Returns
    -------
    list[int]
        The index of the first bin of each group, in order. The first entry is 0, and group ``k`` ends where group
        ``k + 1`` starts.

    Examples
    --------
    >>> group_starts(np.array([[0.0], [0.1], [1.0], [1.05]]), 0.2)
    [0, 2]

    """
    if score_rows.shape[1] == 1:
        return _single_column_starts(score_rows[:, 0].tolist(), threshold)
    return _several_column_starts(score_rows.tolist(), threshold)
