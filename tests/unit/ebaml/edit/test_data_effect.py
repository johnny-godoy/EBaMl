"""Tests for the effect of an edit on rows of data."""

import narwhals as nw
import numpy as np
import polars as pl
import pytest

from ebaml.core.custom_types import Frame
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.data_effect import RowChanges, row_changes
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import Edit, SetScores, Shift
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, MissingBin, WholeGraph
from tests.unit.ebaml.conftest import PandasFrame, frame_of_columns


def _frame(data: PandasFrame) -> Frame:
    return nw.from_native(data)  # pyrefly: ignore[no-matching-overload] -- pyrefly rejects every overload of narwhals.from_native for a pandas frame


def _changes(ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor, edit: Edit, data: Frame) -> RowChanges:
    session = EditSession(ebm)
    session.apply(edit, "Test the effect on rows")
    return row_changes(session.model_at("original"), session.model, data)


def test_an_edit_of_the_missing_bin_reaches_only_the_rows_with_a_missing_value(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    missing_rows = edit_data["age"].isna().sum()
    changes = _changes(edit_ebm, SetScores("age", MissingBin(), 5.0), _frame(edit_data))
    assert missing_rows > 0
    assert changes.row_count == len(edit_data)
    assert changes.rows_with_changed_value == missing_rows
    assert changes.rows_with_changed_class is not None
    assert 0 < changes.rows_with_changed_class <= missing_rows
    assert changes.largest_value_change > 0
    assert changes.mean_value_change > 0


def test_an_edit_of_some_bins_reaches_the_rows_in_those_bins(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    weights = term_view(edit_ebm, "score").weights
    changes = _changes(edit_ebm, Shift("score", Bins(2, 4), -0.5), _frame(edit_data))
    assert changes.rows_with_changed_value == round(float(weights[3:6].sum()))
    assert changes.mean_value_change < 0


def test_an_edit_that_changes_no_score_reaches_no_row(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    changes = _changes(edit_ebm, Shift("score", WholeGraph(), 0.0), _frame(edit_data))
    assert changes == RowChanges(len(edit_data), 0, 0, 0.0, 0.0)


def test_a_polars_frame_with_more_columns_and_another_order_gives_the_same_count(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    edit = Shift("score", Bins(2, 4), -0.5)
    shuffled = pl.from_pandas(edit_data).select(["grade", "group", "score", "age"]).with_columns(extra=pl.lit(1))
    frame = nw.from_native(shuffled)  # pyrefly: ignore[no-matching-overload] -- pyrefly rejects every overload of narwhals.from_native for a polars frame
    assert _changes(edit_ebm, edit, frame) == _changes(edit_ebm, edit, _frame(edit_data))


def test_a_regressor_has_no_class_to_change(regression_ebm: ExplainableBoostingRegressor) -> None:
    random_generator = np.random.default_rng(5)
    data = frame_of_columns({"a": random_generator.normal(size=80), "c": random_generator.choice(["x", "y"], 80)})
    changes = _changes(regression_ebm, Shift("a", WholeGraph(), 1.0), _frame(data))
    assert changes.rows_with_changed_class is None
    assert changes.rows_with_changed_value == len(data)
    assert changes.largest_value_change == pytest.approx(1.0)
    assert changes.mean_value_change == pytest.approx(1.0)


def test_data_with_no_row_gives_a_count_of_zero(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    changes = _changes(edit_ebm, Shift("score", Bins(2, 4), -0.5), _frame(edit_data.head(0)))
    assert changes == RowChanges(0, 0, 0, 0.0, 0.0)


def test_data_without_a_column_of_the_model_is_an_error(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    with pytest.raises(InvalidInstanceError, match=r"no column for these features: \['grade'\]"):
        _changes(edit_ebm, Shift("score", Bins(2, 4), -0.5), _frame(edit_data.drop(columns="grade")))
