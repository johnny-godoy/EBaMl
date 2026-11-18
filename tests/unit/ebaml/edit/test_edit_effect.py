"""Tests for the effect of an edit on data: the metrics in three scopes and the profile of the affected rows."""

import warnings

import numpy as np
import pytest
from sklearn import metrics

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.edit_effect import (
    AFFECTED_ROWS,
    ALL_ROWS,
    SUBGROUP_ROWS,
    ClassifierMetrics,
    ConfusionMatrix,
    RegressorMetrics,
    Subgroup,
    affected_profile,
    edit_effect,
)
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import SetScores, Shift
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, MissingBin
from tests.unit.ebaml.conftest import PandasFrame, as_frame
from tests.unit.ebaml.edit.conftest import fixed_clock


def _labels(ebm: ExplainableBoostingClassifier, data: PandasFrame) -> np.ndarray:
    random_generator = np.random.default_rng(11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probability = ebm.predict_proba(data)[:, 1]
    return (random_generator.uniform(size=len(data)) < probability).astype(int)


def _session_with(ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor, *edits: object) -> EditSession:
    session = EditSession(ebm, clock=fixed_clock())
    for edit in edits:
        session.apply(edit, "A test edit")  # pyrefly: ignore[bad-argument-type] -- the tests pass edits only
    return session


def _reference_metrics(
    ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor,
    data: PandasFrame,
    labels: np.ndarray,
) -> ClassifierMetrics:
    assert isinstance(ebm, ExplainableBoostingClassifier)
    predicted = ebm.predict(data)
    true_negative, false_positive, false_negative, true_positive = metrics.confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    ).ravel()
    correct: np.ndarray = np.asarray(predicted) == labels
    accuracy = float(correct.mean())
    both = len(set(labels.tolist())) > 1
    return ClassifierMetrics(
        accuracy=accuracy,
        balanced_accuracy=float(metrics.balanced_accuracy_score(labels, predicted)) if both else accuracy,
        roc_auc=float(metrics.roc_auc_score(labels, ebm.predict_proba(data)[:, 1])) if both else None,
        confusion_matrix=ConfusionMatrix(
            int(true_negative),
            int(false_positive),
            int(false_negative),
            int(true_positive),
        ),
    )


def test_the_original_model_has_no_last_edit_and_no_affected_row(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    labels = _labels(edit_ebm, edit_data)
    effect = edit_effect(_session_with(edit_ebm), as_frame(edit_data), labels)
    assert (effect.revision, effect.last_revision, effect.edited_feature) == ("original", None, None)
    assert effect.changes_since_last is None
    assert effect.changes_since_original.rows_with_changed_value == 0
    everything, affected = effect.scopes
    assert (everything.scope, everything.row_count) == (ALL_ROWS, len(edit_data))
    assert everything.original == everything.current == _reference_metrics(edit_ebm, edit_data, labels)
    assert everything.last is None
    assert (affected.scope, affected.row_count, affected.current) == (AFFECTED_ROWS, 0, None)


def test_the_metrics_of_each_scope_match_a_reference_on_the_same_rows(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    labels = _labels(edit_ebm, edit_data)
    session = _session_with(edit_ebm, Shift("age", Bins(10, 20), 1.5))
    edited = session.model
    effect = edit_effect(session, as_frame(edit_data), labels, Subgroup("group", "x"))
    view = term_view(edit_ebm, "age")
    positions = view.positions_of(edit_data["age"].to_numpy())
    in_bins = (positions >= 11) & (positions <= 21)
    in_group = (edit_data["group"] == "x").to_numpy()
    everything, affected, group = effect.scopes
    assert effect.last_revision == "original"
    assert effect.edited_feature == "age"
    assert affected.row_count == int(in_bins.sum())
    assert affected.original == _reference_metrics(edit_ebm, edit_data[in_bins], labels[in_bins])
    assert affected.current == _reference_metrics(edited, edit_data[in_bins], labels[in_bins])
    assert affected.last == affected.original
    assert everything.current == _reference_metrics(edited, edit_data, labels)
    assert (group.scope, group.row_count) == (SUBGROUP_ROWS, int(in_group.sum()))
    assert group.current == _reference_metrics(edited, edit_data[in_group], labels[in_group])
    assert effect.changes_since_last == effect.changes_since_original
    assert effect.subgroup == Subgroup("group", "x")


def test_the_last_edit_is_the_edit_of_the_revision_and_the_model_before_it_is_the_last_model(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    session = _session_with(edit_ebm, Shift("age", Bins(10, 20), 1.5), SetScores("group", Categories(["y"]), 0.0))
    first, second = (row.id for row in session.history_table())
    latest = edit_effect(session, as_frame(edit_data))
    assert (latest.revision, latest.last_revision, latest.edited_feature) == (second, first, "group")
    assert latest.changes_since_last is not None
    assert latest.changes_since_last.rows_with_changed_value == (edit_data["group"] == "y").sum()
    earlier = edit_effect(session, as_frame(edit_data), revision=first)
    assert (earlier.revision, earlier.last_revision, earlier.edited_feature) == (first, "original", "age")


def test_without_labels_the_effect_counts_rows_only(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    effect = edit_effect(_session_with(edit_ebm, Shift("age", MissingBin(), 1.0)), as_frame(edit_data))
    everything, affected = effect.scopes
    assert everything.original is everything.last is everything.current is None
    assert affected.row_count == edit_data["age"].isna().sum()


def test_a_scope_with_one_class_has_no_roc_auc_and_its_balanced_accuracy_is_its_accuracy(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    labels = np.ones(len(edit_data), dtype=int)
    effect = edit_effect(_session_with(edit_ebm), as_frame(edit_data), labels)
    current = effect.scopes[0].current
    assert isinstance(current, ClassifierMetrics)
    assert current.roc_auc is None
    assert current.balanced_accuracy == current.accuracy


def test_a_regressor_gets_the_errors_and_no_percentage_error_for_a_true_value_of_zero(
    edit_regressor: ExplainableBoostingRegressor,
    edit_data: PandasFrame,
) -> None:
    truth = 2 * np.nan_to_num(edit_data["age"].to_numpy()) + 1
    session = _session_with(edit_regressor, Shift("score", Bins(0, 4), 0.5))
    effect = edit_effect(session, as_frame(edit_data), truth)
    current = effect.scopes[0].current
    error = session.model.predict(edit_data) - truth
    assert isinstance(current, RegressorMetrics)
    assert current.rmse == pytest.approx(float(np.sqrt(np.mean(error**2))))
    assert current.mae == pytest.approx(float(np.mean(np.abs(error))))
    assert current.mape == pytest.approx(float(np.mean(np.abs(error / truth))))
    assert effect.changes_since_original.rows_with_changed_class is None
    truth[0] = 0.0
    zero = edit_effect(session, as_frame(edit_data), truth).scopes[0].current
    assert isinstance(zero, RegressorMetrics)
    assert zero.mape is None


def test_a_subgroup_of_a_numeric_column_compares_numbers_and_may_use_a_column_outside_the_model(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    """Bug: the subgroup compared texts, so the value 1 found no row with the value 1.0."""
    data = edit_data.assign(siblings=(np.arange(len(edit_data)) % 3).astype(float))
    session = _session_with(edit_ebm)
    for value in (1, 1.0, "1"):
        effect = edit_effect(session, as_frame(data), None, Subgroup("siblings", value))
        assert effect.scopes[2].row_count == len(edit_data) // 3
    with pytest.raises(InvalidInstanceError, match="holds numbers, so the subgroup needs a number, not 'one'"):
        edit_effect(session, as_frame(data), None, Subgroup("siblings", "one"))


def test_the_effect_names_what_the_data_and_the_labels_lack(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    session = _session_with(edit_ebm)
    with pytest.raises(InvalidInstanceError, match=r"The data has no column for these features: \['score'\]"):
        edit_effect(session, as_frame(edit_data.drop(columns="score")))
    with pytest.raises(InvalidInstanceError, match="the data has 600 rows and the labels 3"):
        edit_effect(session, as_frame(edit_data), [0, 1, 0])
    with pytest.raises(InvalidInstanceError, match="The data has no column 'sex' for the subgroup"):
        edit_effect(session, as_frame(edit_data), None, Subgroup("sex", "female"))


def test_the_profile_compares_the_bins_of_the_affected_rows_with_all_rows_by_l1_distance(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    session = _session_with(edit_ebm, SetScores("group", Categories(["x"]), 2.0))
    profile = affected_profile(session, as_frame(edit_data))
    in_group = (edit_data["group"] == "x").to_numpy()
    assert (profile.edited_feature, profile.affected_rows, profile.total_rows) == ("group", int(in_group.sum()), 600)
    assert [feature.feature for feature in profile.features] == sorted(
        ["age", "score", "grade"],
        key=lambda name: -next(item.distance for item in profile.features if item.feature == name),
    )
    grade = next(feature for feature in profile.features if feature.feature == "grade")
    shares_all = edit_data["grade"].value_counts(normalize=True)
    shares_affected = edit_data.loc[in_group, "grade"].value_counts(normalize=True)
    assert dict(zip(grade.bins, grade.all_shares, strict=True)) == pytest.approx(shares_all.to_dict())
    assert dict(zip(grade.bins, grade.affected_shares, strict=True)) == pytest.approx(shares_affected.to_dict())
    affected_by_name = {str(name): float(share) for name, share in shares_affected.items()}
    expected = sum(abs(affected_by_name.get(str(name), 0.0) - float(share)) for name, share in shares_all.items())
    assert grade.distance == pytest.approx(expected)
    age = next(feature for feature in profile.features if feature.feature == "age")
    assert age.bins[0] == "missing"


def test_the_profile_is_empty_when_the_last_edit_reaches_no_row(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    profile = affected_profile(_session_with(edit_ebm), as_frame(edit_data))
    assert (profile.affected_rows, profile.features) == (0, ())
