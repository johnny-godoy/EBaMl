"""Tests of the greedy merge of neighbouring bins, against the loop that ``EBMGraph.simplify`` ran before."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from ebaml.graphs.bin_groups import group_starts


def _reference_starts(score_rows: np.ndarray, threshold: float) -> list[int]:
    """Run the merge of ``EBMGraph.simplify`` of version 0.15.0, the oracle of the differential test.

    Returns
    -------
    list[int]
        The first bin of each group.

    """
    rows = score_rows.tolist()
    groups = [[0]]
    for bin_index, row in enumerate(rows[1:], start=1):
        first = rows[groups[-1][0]]
        if max(abs(a - b) for a, b in zip(first, row, strict=True)) <= threshold:
            groups[-1].append(bin_index)
        else:
            groups.append([bin_index])
    return [group[0] for group in groups]


_SCORES = st.floats(min_value=-5, max_value=5, allow_nan=False) | st.sampled_from([0.0, 0.5, 1.0, float("nan")])


@given(
    score_rows=hnp.arrays(
        np.float64,
        st.tuples(st.integers(1, 40), st.integers(1, 4)),
        elements=_SCORES,
    ),
    threshold=st.sampled_from([0.0, 0.01, 0.1, 0.5, 2.0]) | st.floats(min_value=0, max_value=3),
)
def test_the_groups_equal_the_groups_of_the_loop_of_simplify(score_rows: np.ndarray, threshold: float) -> None:
    assert group_starts(score_rows, threshold) == _reference_starts(score_rows, threshold)


def test_a_nan_score_opens_a_group_with_one_column_as_with_several() -> None:
    nan = float("nan")
    assert group_starts(np.array([[0.0], [nan], [0.0]]), 1.0) == [0, 1, 2]
    assert group_starts(np.array([[0.0, 0.0], [nan, nan], [0.0, 0.0]]), 1.0) == [0, 1, 2]


def test_a_bin_of_several_classes_joins_only_if_every_class_stays_within_the_threshold() -> None:
    rows = np.array([[0.0, 0.0], [0.1, 0.5], [0.15, 0.45]])
    assert group_starts(rows, 0.2) == [0, 1]
