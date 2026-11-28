"""The effect of an edit on data: the metrics of the models in three scopes, and the subgroups that the edit reaches.

GAM Changer (Wang et al., KDD 2022, MIT licence) shows two panels after each edit. The Metric Panel compares three
models: the original model, the model before the last edit, and the current model. It shows each metric for all rows,
for the rows that the last edit changes, and for a subgroup. The paper gives the reason (sections 4.2.1 and 6): a
global metric hardly moves when a few bins change, so the harm of an edit shows only in the affected rows or in a
subgroup. The Feature Panel compares, for each other feature, the bins of the affected rows with the bins of all rows,
so that a person sees which subgroup an edit reaches. ``edit_effect`` and ``affected_profile`` give the same numbers.

Each prediction comes from the model itself (``predict`` and ``predict_proba``), never from a link function that the
code assumes. A row with a missing value stays in every count, because an edit of the missing bin reaches only such
rows. A metric describes the rows that the caller passes. It is not a measure of the quality of the edit.
"""

import abc
import functools
from collections.abc import Sequence
from typing import cast, override

import attrs
import narwhals as nw
import numpy as np
import numpy.typing as npt
from sklearn import metrics

from ebaml.core.custom_types import ClassLabel, ExplainableBoostingModel, FeatureValue, Frame
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.data_effect import ModelPredictions, RowChanges, changes_between
from ebaml.edit.edit_session import CURRENT, ORIGINAL, EditSession
from ebaml.edit.history import HistoryKind
from ebaml.edit.model_view import term_view
from ebaml.graphs.model_kinds import BinaryClassifier, ModelKind, MulticlassClassifier, model_kind
from ebaml.graphs.terms import main_effect_features

ALL_ROWS = "all rows"
AFFECTED_ROWS = "affected rows"
SUBGROUP_ROWS = "subgroup"


@attrs.frozen
class Subgroup:
    """The rows whose value of one feature is ``value``, for example the rows with ``sex`` equal to ``"female"``."""

    feature: str
    value: str | float


@attrs.frozen
class ConfusionMatrix:
    """The counts of the predicted and the true classes of a binary classifier, with ``classes_[1]`` as positive."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@attrs.frozen
class ClassifierMetrics:
    """The metrics of a binary classifier on a set of rows.

    ``roc_auc`` is ``None`` when the rows hold one class only, because the area is not defined then.
    """

    accuracy: float
    balanced_accuracy: float
    roc_auc: float | None
    confusion_matrix: ConfusionMatrix


@attrs.frozen
class MulticlassMetrics:
    """The metrics of a multiclass classifier on a set of rows.

    ``roc_auc`` is the mean over the classes of the area under the ROC curve of one class against the others (the macro
    one-versus-rest area). It is ``None`` when the rows do not hold every class, because the area of a class with no
    row is not defined. ``confusion_matrix`` has one row for each true class and one column for each predicted class,
    in the order of ``classes``.
    """

    accuracy: float
    balanced_accuracy: float
    roc_auc: float | None
    classes: tuple[ClassLabel, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@attrs.frozen
class RegressorMetrics:
    """The errors of a regressor on a set of rows.

    ``mape`` is the mean absolute percentage error as a share (0.1 means 10 %). It is ``None`` when a true value is 0,
    because the percentage error is not defined then.
    """

    rmse: float
    mae: float
    mape: float | None


type Metrics = ClassifierMetrics | MulticlassMetrics | RegressorMetrics


@attrs.frozen
class ScopeMetrics:
    """The metrics of the three models on one set of rows.

    ``last`` is the model before the last edit, and it is ``None`` for the original model, which no edit made. Each
    metric is ``None`` when the caller passes no labels, or when the scope holds no row.
    """

    scope: str
    row_count: int
    original: Metrics | None
    last: Metrics | None
    current: Metrics | None


@attrs.frozen
class EditEffect:
    """The effect of the edits of a session on rows of data.

    ``revision`` is the revision that the effect describes, and ``last_revision`` is the revision before its last edit.
    The affected rows are the rows whose prediction the last edit changes: the rows in the bins that it changed.
    """

    revision: str
    last_revision: str | None
    edited_feature: str | None
    changes_since_original: RowChanges
    changes_since_last: RowChanges | None
    scopes: tuple[ScopeMetrics, ...]
    subgroup: Subgroup | None


@attrs.frozen
class FeatureProfile:
    """The bins of one feature among the affected rows and among all rows.

    ``distance`` is the l1 distance between the two shares: the sum over the bins of the absolute difference of the
    shares. It goes from 0 (the same mix) to 2 (no bin in common). GAM Changer computes the same distance
    (``draw-feature.js``).
    """

    feature: str
    bins: tuple[str, ...]
    affected_shares: tuple[float, ...]
    all_shares: tuple[float, ...]
    distance: float


@attrs.frozen
class AffectedProfile:
    """Which subgroups the last edit reaches: for each other feature, the mix of its bins among the affected rows.

    ``features`` is sorted by distance, the largest first. It is empty when the last edit affects no row.
    """

    revision: str
    edited_feature: str | None
    affected_rows: int
    total_rows: int
    features: tuple[FeatureProfile, ...]


@attrs.frozen(eq=False)
class _Versions:
    """The models of the Metric Panel, and the revisions of the current model and of the model before its last edit."""

    revision: str
    last_revision: str | None
    edited_feature: str | None
    original: ExplainableBoostingModel
    last: ExplainableBoostingModel | None
    current: ExplainableBoostingModel


def _versions(session: EditSession, revision: str) -> _Versions:
    info = session.revision_info(revision)
    edits = [row for row in session.history_table() if row.kind is HistoryKind.EDIT]
    ids = [row.id for row in edits]
    if info.revision == ORIGINAL:
        last_revision, edited_feature = None, None
    else:
        position = ids.index(info.revision)
        last_revision = ids[position - 1] if position else ORIGINAL
        edited_feature = edits[position].feature
    return _Versions(
        revision=info.revision,
        last_revision=last_revision,
        edited_feature=edited_feature,
        original=session.model_at(ORIGINAL),
        last=None if last_revision is None else session.model_at(last_revision),
        current=session.model_at(info.revision),
    )


class _Outputs(abc.ABC):
    """The predictions and the metrics of one kind of model."""

    @abc.abstractmethod
    def predictions(self, source: ModelPredictions) -> npt.NDArray[np.float64]:
        """Give the prediction of each row: the probability of each class, or the predicted value.

        Returns
        -------
        npt.NDArray[np.float64]
            One row of values for each row of data.

        """

    @abc.abstractmethod
    def metrics(self, source: ModelPredictions, labels: npt.NDArray[np.generic]) -> Metrics:
        """Compute the metrics of the model on rows with their labels.

        Returns
        -------
        Metrics
            The metrics.

        """


@attrs.frozen(eq=False)
class _Confusion:
    """The classes of a classifier, the confusion matrix of rows in their order, and the accuracy on the rows."""

    classes: list[ClassLabel]
    matrix: npt.NDArray[np.int64]
    accuracy: float


@attrs.frozen(eq=False)
class _ClassifierOutputs(_Outputs):
    """The predictions and the metrics of a binary classifier. The multiclass outputs extend it."""

    model: ExplainableBoostingClassifier

    @override
    def predictions(self, source: ModelPredictions) -> npt.NDArray[np.float64]:
        return source.probabilities()

    def confusion(self, source: ModelPredictions, labels: npt.NDArray[np.generic]) -> _Confusion:
        """Compare the predicted classes of rows with their labels.

        Returns
        -------
        _Confusion
            The classes, the confusion matrix with one row for each true class, and the accuracy.

        """
        classes: list[ClassLabel] = np.asarray(self.model.classes_).tolist()
        predicted = source.predicted()
        matrix = np.asarray(metrics.confusion_matrix(labels, predicted, labels=classes), dtype=np.int64)
        return _Confusion(classes, matrix, float(metrics.accuracy_score(labels, predicted)))

    @override
    def metrics(self, source: ModelPredictions, labels: npt.NDArray[np.generic]) -> Metrics:
        confusion = self.confusion(source, labels)
        positive: npt.NDArray[np.bool_] = np.asarray(labels) == confusion.classes[1]
        # with one class in the rows, the balanced accuracy is the recall of that class, which is the accuracy
        has_both_classes = bool(positive.any() and not positive.all())
        recalls = np.diag(confusion.matrix) / confusion.matrix.sum(axis=1).clip(min=1)
        scores = self.predictions(source)[:, 1]
        return ClassifierMetrics(
            accuracy=confusion.accuracy,
            balanced_accuracy=float(recalls.mean()) if has_both_classes else confusion.accuracy,
            roc_auc=float(metrics.roc_auc_score(positive, scores)) if has_both_classes else None,
            confusion_matrix=ConfusionMatrix(*(int(count) for count in confusion.matrix.ravel())),
        )


@attrs.frozen(eq=False)
class _MulticlassOutputs(_ClassifierOutputs):
    """The predictions and the metrics of a multiclass classifier."""

    @override
    def metrics(self, source: ModelPredictions, labels: npt.NDArray[np.generic]) -> Metrics:
        confusion = self.confusion(source, labels)
        # the balanced accuracy is the mean recall of the classes in the rows; a class with no row has no recall
        totals = confusion.matrix.sum(axis=1)
        seen = totals > 0
        probabilities = self.predictions(source)
        return MulticlassMetrics(
            accuracy=confusion.accuracy,
            balanced_accuracy=float(np.mean(np.diag(confusion.matrix)[seen] / totals[seen])),
            roc_auc=float(metrics.roc_auc_score(labels, probabilities, multi_class="ovr", labels=confusion.classes))
            if seen.all()
            else None,
            classes=tuple(confusion.classes),
            confusion_matrix=tuple(tuple(int(count) for count in row) for row in confusion.matrix.tolist()),
        )


@attrs.frozen(eq=False)
class _RegressorOutputs(_Outputs):
    model: ExplainableBoostingRegressor

    @override
    def predictions(self, source: ModelPredictions) -> npt.NDArray[np.float64]:
        return np.asarray(source.predicted(), dtype=float).reshape(-1, 1)

    @override
    def metrics(self, source: ModelPredictions, labels: npt.NDArray[np.generic]) -> Metrics:
        truth = np.asarray(labels, dtype=float)
        error = self.predictions(source)[:, 0] - truth
        return RegressorMetrics(
            rmse=float(np.sqrt(np.mean(error**2))),
            mae=float(np.mean(np.abs(error))),
            mape=float(np.mean(np.abs(error / truth))) if np.all(truth != 0) else None,
        )


@functools.singledispatch
def _outputs_of_kind(_kind: ModelKind, model: ExplainableBoostingModel) -> _Outputs:
    """Select the predictions and the metrics of a kind of model. Each kind of classifier registers its own.

    Returns
    -------
    _Outputs
        The outputs of a regressor, the kind that registers no function.

    """
    return _RegressorOutputs(cast(ExplainableBoostingRegressor, model))


@_outputs_of_kind.register
def _binary(_kind: BinaryClassifier, model: ExplainableBoostingModel) -> _Outputs:
    return _ClassifierOutputs(cast(ExplainableBoostingClassifier, model))


@_outputs_of_kind.register
def _multiclass(_kind: MulticlassClassifier, model: ExplainableBoostingModel) -> _Outputs:
    return _MulticlassOutputs(cast(ExplainableBoostingClassifier, model))


def _outputs(model: ExplainableBoostingModel) -> _Outputs:
    return _outputs_of_kind(model_kind(model), model)


def _model_rows(ebm: ExplainableBoostingModel, data: Frame) -> Frame:
    names = list(ebm.feature_names_in_)
    missing = [name for name in names if name not in data.columns]
    if missing:
        msg = f"The data has no column for these features: {missing}."
        raise InvalidInstanceError(msg)
    # interpret reads the columns of a frame by position, so select them in the order of the model
    return data.select(names)


@attrs.frozen(eq=False)
class _VersionPredictions:
    """The predictions of the three models of the Metric Panel for one set of rows."""

    row_count: int
    original: ModelPredictions
    last: ModelPredictions | None
    current: ModelPredictions


def _version_predictions(versions: _Versions, rows: Frame) -> _VersionPredictions:
    native = rows.to_native()
    return _VersionPredictions(
        len(rows),
        ModelPredictions(versions.original, native),
        None if versions.last is None else ModelPredictions(versions.last, native),
        ModelPredictions(versions.current, native),
    )


def _affected(predictions: _VersionPredictions) -> npt.NDArray[np.bool_]:
    """Find the rows whose prediction the last edit changes.

    Returns
    -------
    npt.NDArray[np.bool_]
        One flag for each row. Every flag is false for the original model, which no edit made.

    """
    if predictions.last is None:
        return np.zeros(predictions.row_count, dtype=bool)
    last, current = predictions.last, predictions.current
    changed: npt.NDArray[np.bool_] = _outputs(last.model).predictions(last) != _outputs(current.model).predictions(
        current,
    )
    return changed.any(axis=1)


def _scope(name: str, predictions: _VersionPredictions, labels: npt.NDArray[np.generic] | None) -> ScopeMetrics:
    if labels is None or predictions.row_count == 0:
        return ScopeMetrics(name, predictions.row_count, None, None, None)
    last = predictions.last
    return ScopeMetrics(
        name,
        predictions.row_count,
        _outputs(predictions.original.model).metrics(predictions.original, labels),
        None if last is None else _outputs(last.model).metrics(last, labels),
        _outputs(predictions.current.model).metrics(predictions.current, labels),
    )


def _subgroup_mask(data: Frame, subgroup: Subgroup) -> npt.NDArray[np.bool_]:
    """Find the rows of a subgroup. A numeric column compares numbers, so ``0`` finds the rows with ``0.0``.

    Returns
    -------
    npt.NDArray[np.bool_]
        One flag for each row.

    Raises
    ------
    InvalidInstanceError
        If the data has no such column, or the value is not a number for a numeric column.

    """
    if subgroup.feature not in data.columns:
        msg = f"The data has no column {subgroup.feature!r} for the subgroup. The columns are {list(data.columns)}."
        raise InvalidInstanceError(msg)
    column = data.get_column(subgroup.feature)
    if column.dtype.is_numeric():
        try:
            value = float(subgroup.value)
        except ValueError as error:
            msg = (
                f"The column {subgroup.feature!r} holds numbers, so the subgroup needs a number,"
                f" not {subgroup.value!r}."
            )
            raise InvalidInstanceError(msg) from error
        numeric: npt.NDArray[np.bool_] = column.to_numpy().astype(float) == value
        return numeric
    texts: npt.NDArray[np.bool_] = np.asarray(
        [str(item) == str(subgroup.value) for item in column.to_numpy().tolist()],
        dtype=bool,
    )
    return texts


def edit_effect(
    session: EditSession,
    data: Frame,
    labels: Sequence[FeatureValue] | npt.ArrayLike | None = None,
    subgroup: Subgroup | None = None,
    *,
    revision: str = CURRENT,
) -> EditEffect:
    """Measure the effect of the edits of a session on rows of data, as the Metric Panel of GAM Changer does.

    Parameters
    ----------
    session : EditSession
        The session.
    data : Frame
        The rows, with a column for each feature of the model. Wrap a pandas or polars frame with
        ``narwhals.from_native``.
    labels : Sequence[FeatureValue] | npt.ArrayLike, optional
        The true outcome of each row: the class for a classifier, or the value for a regressor. Without labels, the
        result holds the counts of rows only.
    subgroup : Subgroup, optional
        The rows of the third scope, for example the rows of one level of a nominal feature.
    revision : str, default="current"
        The revision to describe: ``current``, ``original`` or the id of an edit.

    Returns
    -------
    EditEffect
        The counts of the rows whose prediction changes, and the metrics of the three models in each scope.

    Raises
    ------
    InvalidInstanceError
        If the data lacks a column of the model or of the subgroup, or the labels do not have one value for each row.

    """
    frame = data
    versions = _versions(session, revision)
    rows = _model_rows(versions.current, frame)
    truth = None if labels is None else np.asarray(labels)
    if truth is not None and len(truth) != len(rows):
        msg = f"The labels need one value for each row: the data has {len(rows)} rows and the labels {len(truth)}."
        raise InvalidInstanceError(msg)
    every_row = _version_predictions(versions, rows)
    affected = _affected(every_row)
    scopes = [
        _scope(ALL_ROWS, every_row, truth),
        _scope(
            AFFECTED_ROWS,
            _version_predictions(
                versions,
                rows.filter(nw.new_series("affected", affected, backend=nw.get_native_namespace(rows))),
            ),
            None if truth is None else truth[affected],
        ),
    ]
    if subgroup is not None:
        mask = _subgroup_mask(frame, subgroup)
        scopes.append(
            _scope(
                SUBGROUP_ROWS,
                _version_predictions(
                    versions,
                    rows.filter(nw.new_series("subgroup", mask, backend=nw.get_native_namespace(rows))),
                ),
                None if truth is None else truth[mask],
            ),
        )
    # the counts read the predictions of the scope of every row, so no model predicts those rows twice
    return EditEffect(
        revision=versions.revision,
        last_revision=versions.last_revision,
        edited_feature=versions.edited_feature,
        changes_since_original=changes_between(every_row.original, every_row.current),
        changes_since_last=None if every_row.last is None else changes_between(every_row.last, every_row.current),
        scopes=tuple(scopes),
        subgroup=subgroup,
    )


def _profile(
    ebm: ExplainableBoostingModel,
    feature: str,
    values: npt.NDArray[np.generic],
    affected: npt.NDArray[np.bool_],
) -> FeatureProfile:
    view = term_view(ebm, feature)
    positions = view.positions_of(values)
    bin_count = view.unseen_position + 1
    all_counts = np.bincount(positions, minlength=bin_count)
    affected_counts = np.bincount(positions[affected], minlength=bin_count)
    seen = np.flatnonzero(all_counts)
    all_shares = all_counts[seen] / all_counts.sum()
    affected_shares = affected_counts[seen] / affected_counts.sum()
    return FeatureProfile(
        feature=feature,
        bins=tuple(view.label(int(position)) for position in seen),
        affected_shares=tuple(affected_shares.tolist()),
        all_shares=tuple(all_shares.tolist()),
        distance=float(np.abs(affected_shares - all_shares).sum()),
    )


def affected_profile(session: EditSession, data: Frame, *, revision: str = CURRENT) -> AffectedProfile:
    """Compare the rows that the last edit reaches with all rows, feature by feature, as the Feature Panel does.

    The purpose is fairness: a person sees which subgroup an edit reaches. An edit of high ages that reaches mostly the
    rows of one sex shows a large distance for that feature.

    Parameters
    ----------
    session : EditSession
        The session.
    data : Frame
        The rows, with a column for each feature of the model, or ``InvalidInstanceError`` names the missing ones.
        Wrap a pandas or polars frame with ``narwhals.from_native``.
    revision : str, default="current"
        The revision whose last edit to describe.

    Returns
    -------
    AffectedProfile
        For each feature with a main effect, other than the edited one, the share of each bin among the affected rows
        and among all rows, and their l1 distance, sorted by distance.

    """
    frame = data
    versions = _versions(session, revision)
    rows = _model_rows(versions.current, frame)
    affected = _affected(_version_predictions(versions, rows))
    model = versions.current
    names = [model.feature_names_in_[index] for index in main_effect_features(model)]
    profiles = (
        [
            _profile(model, name, rows.get_column(name).to_numpy(), affected)
            for name in names
            if name != versions.edited_feature
        ]
        if affected.any()
        else []
    )
    return AffectedProfile(
        revision=versions.revision,
        edited_feature=versions.edited_feature,
        affected_rows=int(affected.sum()),
        total_rows=len(rows),
        features=tuple(sorted(profiles, key=lambda profile: profile.distance, reverse=True)),
    )
