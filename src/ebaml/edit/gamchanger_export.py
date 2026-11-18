"""Write the edits of a session as a ``.gamchanger`` file that the GAM Changer widget opens.

GAM Changer (Wang et al., KDD 2022, MIT licence) opens a JSON document with ``modelData`` (the model before the
edits, as ``get_model_data`` of its Python package writes it), ``sampleData`` (rows with labels) and ``historyList``
(the commits). ``save_gamchanger`` writes the original model of a session, one commit for each applied edit with the
graph of the edited feature after the edit, and an ``original`` commit before the first edit of each feature, as the
widget does. The format follows a file that GAM Changer 0.1.13 saved.

The file cannot hold everything that EBaMl edits. The result names each part that it leaves out, so nothing is lost in
silence: the missing bin and the unseen bin, which GAM Changer cannot show; the pair terms, which this file leaves out,
so the metrics of the widget leave out their effect; the edits that name terms and no feature; and the rows of the data
with a missing value, which the widget cannot read. ``load_gamchanger`` reads the file back, except a commit whose new
bins are not of equal width (after ``SplitBin`` or ``ExtendRange``).
"""

import hashlib
import os
from collections.abc import Sequence
from itertools import starmap
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
import orjson

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue, Frame, JSONObject, JSONValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.edit.edit_session import ORIGINAL, EditSession
from ebaml.edit.gamchanger_files import GAMCHANGER_DECIMALS
from ebaml.edit.history import HistoryRow
from ebaml.edit.model_view import TermView, interaction_names, term_view
from ebaml.graphs.terms import main_effect_features

_MILLISECONDS = 1000
_HASH_LENGTH = 32
# the edits of EBaMl that do what one GAM Changer tool does; another edit keeps the name of its class as the type
_GAMCHANGER_TYPES = {
    "Shift": "move",
    "SetScores": "align",
    "Align": "align",
    "Interpolate": "inplace-interpolate",
    "FitLine": "inplace-regression",
    "ZeroTerm": "delete",
}
_DIRECTIONS = ("increasing", "decreasing")


@attrs.frozen
class GamChangerExport:
    """What ``save_gamchanger`` wrote.

    Attributes
    ----------
    path: Path
        The file.
    commit_count: int
        The number of commits of edits in the file, without the ``original`` commits.
    sample_count: int
        The number of rows in the file.
    synthetic_samples: bool
        Whether the rows are made from the bins of the model, because the caller passed no data.
    left_out: tuple[str, ...]
        Each part of the session or of the data that the file does not hold, in words.

    """

    path: Path
    commit_count: int
    sample_count: int
    synthetic_samples: bool
    left_out: tuple[str, ...]


def _rounded(values: npt.ArrayLike) -> list[float]:
    rounded: list[float] = np.round(np.asarray(values, dtype=float), GAMCHANGER_DECIMALS).tolist()
    return rounded


def _gamchanger_type(view: TermView) -> str:
    if view.feature_type == "continuous":
        return "continuous"
    if view.feature_type == "nominal":
        return "categorical"
    msg = (
        f"GAM Changer edits continuous and nominal features only, and {view.feature_name!r} is {view.feature_type}."
        " The model cannot become a .gamchanger file."
    )
    raise UnsupportedModelError(msg)


# nosemgrep: ebaml-return-attrs-not-container -- the keys are the categories of the feature, so they are data
def _codes(view: TermView) -> dict[str, int]:
    # the code of a category in GAM Changer is the position of its bin in term_scores_, as get_model_data writes it
    return {name: position + 1 for position, name in enumerate(view.categories())}


def _feature_data(ebm: ExplainableBoostingModel, view: TermView, importance: float) -> JSONObject:
    deviations = np.asarray(ebm.standard_deviations_[view.term], dtype=float)
    data: JSONObject = {
        "name": view.feature_name,
        "type": _gamchanger_type(view),
        "id": [view.feature_index],
        "importance": importance,
        "additive": _rounded(view.scores[1:-1]),
        "error": _rounded(deviations[1:-1]),
        "count": view.weights[1:-1].tolist(),
        "histCount": _rounded(np.asarray(ebm.histogram_weights_[view.feature_index], dtype=float)[1:-1]),
    }
    if view.feature_type == "continuous":
        edges = view.bin_edges()
        data["binEdge"] = [left for left, _ in edges] + [edges[-1][1]]
        data["histEdge"] = _rounded(np.asarray(ebm.histogram_edges_[view.feature_index], dtype=float))
    else:
        codes = list(_codes(view).values())
        data["binLabel"] = codes
        data["histEdge"] = codes
    return data


def _model_data(ebm: ExplainableBoostingModel) -> JSONObject:
    importances = ebm.term_importances()
    views = [term_view(ebm, ebm.feature_names_in_[index]) for index in main_effect_features(ebm)]
    features: list[JSONValue] = [_feature_data(ebm, view, float(importances[view.term])) for view in views]
    encoder: dict[str, JSONValue] = {
        view.feature_name: {str(code): name for name, code in _codes(view).items()}
        for view in views
        if view.feature_type == "nominal"
    }
    lows = [float(np.min(view.scores - np.asarray(ebm.standard_deviations_[view.term]))) for view in views]
    highs = [float(np.max(view.scores + np.asarray(ebm.standard_deviations_[view.term]))) for view in views]
    return {
        "intercept": float(np.ravel(ebm.intercept_)[0]),
        # one check where the model enters: GAM Changer tells a classifier from a regressor by this flag
        "isClassifier": isinstance(ebm, ExplainableBoostingClassifier),
        "features": features,
        "labelEncoder": encoder,
        "scoreRange": [round(min(lows), GAMCHANGER_DECIMALS), round(max(highs), GAMCHANGER_DECIMALS)],
    }


def _continuous_state(view: TermView, scores_of_bins: npt.NDArray[np.float64]) -> JSONObject:
    """Write the graph of a continuous feature as the linked list of points and the line segments of the widget.

    Returns
    -------
    JSONObject
        ``pointData`` and ``additiveData``, as the function that draws the graph in GAM Changer builds them.

    """
    edges = view.bin_edges()
    places = [left for left, _ in edges]
    scores = _rounded(scores_of_bins[1:-1])
    last = len(places) - 1
    points: dict[str, JSONValue] = {}
    segments: list[JSONValue] = []
    for number, (place, score) in enumerate(zip(places, scores, strict=True)):
        point: JSONObject = {
            "x": place,
            "y": score,
            "count": float(view.weights[number + 1]),
            "id": number,
            "ebmID": number,
            "leftPointID": number - 1 if number else None,
            "rightPointID": number + 1 if number < last else None,
            "leftLineIndex": 2 * number - 1 if number else None,
            "rightLineIndex": 2 * number,
        }
        if number == last:
            # the widget reads the upper bound of the graph from the last point
            point["maxX"] = edges[-1][1]
        points[str(number)] = point
        right_place = places[number + 1] if number < last else edges[-1][1]
        target = f"path-{number}-{number + 1 if number < last else number}"
        next_score = scores[number + 1] if number < last else score
        segments.append({
            "x1": place,
            "y1": score,
            "x2": right_place,
            "y2": score,
            "id": f"{target}-r",
            "pos": "r",
            "sx": place,
            "sy": score,
            "tx": right_place,
            "ty": next_score,
        })
        if number < last:
            segments.append({
                "x1": right_place,
                "y1": score,
                "x2": right_place,
                "y2": next_score,
                "id": f"{target}-l",
                "pos": "l",
                "sx": place,
                "sy": score,
                "tx": right_place,
                "ty": next_score,
            })
    return {"pointData": points, "additiveData": segments}


def _nominal_state(
    ebm: ExplainableBoostingModel,
    view: TermView,
    scores_of_bins: npt.NDArray[np.float64],
) -> JSONObject:
    deviations = np.asarray(ebm.standard_deviations_[view.term], dtype=float)
    points: dict[str, JSONValue] = {
        str(code): {
            "x": name,
            "y": round(float(scores_of_bins[code]), GAMCHANGER_DECIMALS),
            "id": code,
            "error": round(float(deviations[code]), GAMCHANGER_DECIMALS),
            "count": float(view.weights[code]),
        }
        for name, code in _codes(view).items()
    }
    return {"pointData": points}


def _state(ebm: ExplainableBoostingModel, feature: str, *, zeroed: bool = False) -> JSONObject:
    """Write the graph of a feature as the state of a commit.

    Returns
    -------
    JSONObject
        The state, with every score 0 if ``zeroed``: GAM Changer cannot remove a term, so a removed term is a graph of
        zeros.

    """
    view = term_view(ebm, feature)
    scores = np.zeros_like(view.scores) if zeroed else view.scores
    return _continuous_state(view, scores) if view.feature_type == "continuous" else _nominal_state(ebm, view, scores)


def _commit(row: HistoryRow, state: JSONObject, commit_type: str, description: str) -> JSONObject:
    time = round(row.time.timestamp() * _MILLISECONDS)
    digest = hashlib.sha256(f"{commit_type}{description}{time}".encode()).hexdigest()[:_HASH_LENGTH]
    return {
        "featureName": row.feature,
        "type": commit_type,
        "description": description,
        "time": time,
        "hash": digest,
        "reviewed": row.reviewed or commit_type == ORIGINAL,
        "metrics": {},
        "state": state,
    }


def _commit_type(row: HistoryRow) -> str:
    if row.edit_type == "Monotonize":
        # the description of a monotone edit ends with the direction that the edit chose
        last_word = row.description.rsplit(" ", 1)[-1]
        return last_word if last_word in _DIRECTIONS else "increasing"
    return _GAMCHANGER_TYPES.get(str(row.edit_type), str(row.edit_type))


@attrs.frozen
class _History:
    commits: list[JSONValue]
    edit_count: int
    left_out: list[str]


def _history(session: EditSession, original: ExplainableBoostingModel) -> _History:
    info = session.revision_info()
    rows = [row for row in session.history_table() if row.kind == "edit"][: info.edit_count]
    known = {original.feature_names_in_[index] for index in main_effect_features(original)}
    commits: list[JSONValue] = []
    left_out: list[str] = []
    started: set[str] = set()
    edit_count = 0
    for row in rows:
        if row.feature is None or row.feature not in known:
            left_out.append(f"the edit {row.short_id} ({row.description}), which names terms and not one graph")
            continue
        if row.feature not in started:
            started.add(row.feature)
            commits.append(_commit(row, _state(original, row.feature), ORIGINAL, "Original graph"))
        model = session.model_at(row.id)
        if row.feature in {model.feature_names_in_[index] for index in main_effect_features(model)}:
            state = _state(model, row.feature)
        else:
            # the edit removed the main effect, which GAM Changer writes as a graph of zeros
            state = _state(original, row.feature, zeroed=True)
        commits.append(_commit(row, state, _commit_type(row), row.description))
        edit_count += 1
    return _History(commits, edit_count, left_out)


def _outer_bins_left_out(original: ExplainableBoostingModel, current: ExplainableBoostingModel) -> list[str]:
    names = {current.feature_names_in_[index] for index in main_effect_features(current)}
    changed = []
    for index in main_effect_features(original):
        feature = original.feature_names_in_[index]
        if feature not in names:
            continue
        before, after = term_view(original, feature), term_view(current, feature)
        if before.scores[0] != after.scores[0] or before.scores[-1] != after.scores[-1]:
            changed.append(feature)
    return [f"the edited scores of the missing bin and the unseen bin of {', '.join(changed)}"] if changed else []


@attrs.frozen
class _Samples:
    data: JSONObject
    count: int
    left_out: list[str]


def _labels_of(ebm: ExplainableBoostingModel, labels: npt.NDArray[np.generic]) -> list[JSONValue]:
    if isinstance(ebm, ExplainableBoostingClassifier):  # GAM Changer reads the index of the class
        classes = list(ebm.classes_)
        return [classes.index(label) for label in labels.tolist()]
    as_numbers: list[JSONValue] = np.asarray(labels, dtype=float).tolist()
    return as_numbers


def _synthetic_rows(views: Sequence[TermView]) -> list[list[FeatureValue]]:
    """Make one row for each bin number: each feature takes the middle of that bin, or the last bin if it has fewer.

    Returns
    -------
    list[list[FeatureValue]]
        The rows, with the values of the features in the order of the model.

    """
    row_count = max(view.real_bin_count for view in views)
    rows: list[list[FeatureValue]] = []
    for number in range(row_count):
        row: list[FeatureValue] = []
        for view in views:
            bin_number = min(number, view.real_bin_count - 1)
            if view.feature_type == "continuous":
                left, right = view.bin_edges()[bin_number]
                row.append((left + right) / 2)
            else:
                row.append(view.categories()[bin_number])
        rows.append(row)
    return rows


def _encoded(view: TermView, value: FeatureValue) -> float:
    """Write one value as GAM Changer reads it: a number, or the code of a category.

    Returns
    -------
    float
        The number, or the code. An unseen category takes the code of the unseen bin, one above the last code, as
        ``get_sample_data`` of GAM Changer writes it.

    """
    if view.feature_type == "continuous":
        return float(np.asarray([value], dtype=float)[0])
    return int(view.positions_of([value])[0])


def _samples(ebm: ExplainableBoostingModel, data: Frame | None, labels: npt.NDArray[np.generic] | None) -> _Samples:
    views = [term_view(ebm, name) for name in ebm.feature_names_in_]
    names = [view.feature_name for view in views]
    left_out: list[str] = []
    if data is None:
        rows = _synthetic_rows(views)
        truth = np.asarray(ebm.predict(np.array(rows, dtype=object)))
    else:
        missing = [name for name in names if name not in data.columns]
        if missing:
            msg = f"The data has no column for these features: {missing}."
            raise InvalidInstanceError(msg)
        rows = [list(row) for row in data.select(names).rows()]
        truth = np.asarray(ebm.predict(np.array(rows, dtype=object))) if labels is None else np.asarray(labels)
        if len(truth) != len(rows):
            msg = f"The labels need one value for each row: the data has {len(rows)} rows and the labels {len(truth)}."
            raise InvalidInstanceError(msg)
        complete = [
            not any(view.positions_of([value])[0] == 0 for view, value in zip(views, row, strict=True)) for row in rows
        ]
        if not all(complete):
            left_out.append(f"{complete.count(False)} rows with a missing value, which GAM Changer cannot read")
        rows = [row for row, keep in zip(rows, complete, strict=True) if keep]
        truth = truth[np.asarray(complete, dtype=bool)]
    encoded = [list(starmap(_encoded, zip(views, row, strict=True))) for row in rows]
    sample_data: JSONObject = {
        "featureNames": names,
        "featureTypes": [_gamchanger_type(view) for view in views],
        "samples": encoded,
        "labels": _labels_of(ebm, truth),
    }
    return _Samples(sample_data, len(encoded), left_out)


def save_gamchanger(
    session: EditSession,
    path: str | os.PathLike[str],
    data: Frame | None = None,
    labels: Sequence[FeatureValue] | npt.NDArray[np.generic] | None = None,
) -> GamChangerExport:
    """Write the original model and the applied edits of a session as a file that the GAM Changer widget opens.

    Warning: with ``data``, the file holds those rows in plain text. A ``.gamchanger`` file of personal data is
    personal data. Without ``data``, the file holds one synthetic row for each bin number, made from the bins of the
    model and labelled with the predictions of the original model, because the widget needs rows to start. Its metrics
    then compare the models with the original model, not with the truth.

    Parameters
    ----------
    session : EditSession
        The session. The file holds the edits up to the head.
    path : str | os.PathLike[str]
        The new file. The function refuses a path that exists.
    data : Frame, optional
        Rows for the metrics of the widget, with a column for each feature of the model. Wrap a pandas or polars frame
        with ``narwhals.from_native``.
    labels : Sequence[FeatureValue] | npt.NDArray[np.generic], optional
        The true outcome of each row of ``data``. Without it, the labels are the predictions of the original model.

    Returns
    -------
    GamChangerExport
        The file, the number of commits and rows, and each part that the file leaves out. A model with an ordinal
        feature raises an ``UnsupportedModelError``, data that lacks a column of the model or labels of another length
        raise an ``InvalidInstanceError``, and a path that exists raises a ``FileExistsError``.

    """
    original = session.model_at(ORIGINAL)
    model_data = _model_data(original)
    history = _history(session, original)
    samples = _samples(original, data, None if labels is None else np.asarray(labels))
    pairs = sorted({
        name
        for index in main_effect_features(original)
        for name in interaction_names(original, original.feature_names_in_[index])
    })
    left_out = [
        *history.left_out,
        *_outer_bins_left_out(original, session.model),
        *([f"the pair terms {', '.join(pairs)}, so the metrics of the widget leave out their effect"] if pairs else []),
        *samples.left_out,
    ]
    document = {"modelData": model_data, "sampleData": samples.data, "historyList": history.commits}
    target = Path(path)
    with target.open("xb") as handle:
        handle.write(orjson.dumps(document, option=orjson.OPT_SERIALIZE_NUMPY))
    return GamChangerExport(target, history.edit_count, samples.count, data is None, tuple(left_out))
