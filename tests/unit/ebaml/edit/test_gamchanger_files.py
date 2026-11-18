"""Tests for the reader of .gamchanger files.

The tests build each document in the format that GAM Changer writes: ``get_model_data`` of
``notebook-widget/gamchanger/gamchanger.py`` for ``modelData``, and the history entries of the widget, with the fields
that a file saved by GAM Changer 0.1.13 holds. The builder shares no code with ``ebaml.edit``.
"""

import copy
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import pytest

from ebaml.core.custom_types import JSONValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import HistoryError
from ebaml.edit.bin_edits import ExtendRange, Rebin, SplitBin
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import AssignScores
from ebaml.edit.gamchanger_files import GamChangerCommit, apply_gamchanger, load_gamchanger
from ebaml.edit.selections import Bins, Categories
from tests.unit.ebaml.edit.conftest import fixed_clock

type Document = dict[str, JSONValue]
TIME_MS = 1794676628395


def _main_term(ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor, feature_index: int) -> int:
    return next(term for term, features in enumerate(ebm.term_features_) if tuple(features) == (feature_index,))


def _rounded(ebm: ExplainableBoostingClassifier, name: str) -> list[float]:
    index = list(ebm.feature_names_in_).index(name)
    rounded: list[float] = np.round(ebm.term_scores_[_main_term(ebm, index)], 4)[1:-1].tolist()
    return rounded


def _left_edges(ebm: ExplainableBoostingClassifier, name: str) -> list[float]:
    index = list(ebm.feature_names_in_).index(name)
    return [float(ebm.feature_bounds_[index][0]), *np.asarray(ebm.bins_[index][0], dtype=float).tolist()]


def _model_data(ebm: ExplainableBoostingClassifier) -> Document:
    features: list[JSONValue] = []
    encoder: dict[str, JSONValue] = {}
    for index, name in enumerate(ebm.feature_names_in_):
        scores = _rounded(ebm, name)
        if ebm.feature_types_in_[index] == "continuous":
            edges = [*_left_edges(ebm, name), float(ebm.feature_bounds_[index][1])]
            features.append({"name": name, "type": "continuous", "additive": scores, "binEdge": edges})
        else:
            mapping = dict(ebm.bins_[index][0])
            features.append({
                "name": name,
                "type": "categorical",
                "additive": scores,
                "binLabel": list(mapping.values()),
            })
            encoder[name] = {str(code): level for level, code in mapping.items()}
    for features_of_term in ebm.term_features_:
        if len(features_of_term) > 1:
            names = [ebm.feature_names_in_[index] for index in features_of_term]
            features.append({"name": " x ".join(names), "type": "interaction"})
    return {
        "intercept": 0.0,
        "isClassifier": True,
        "features": features,
        "labelEncoder": encoder,
        "scoreRange": [-1, 1],
    }


def _continuous_points(places: Sequence[float], scores: Sequence[float]) -> Document:
    last = len(places) - 1
    return {
        str(point_id): {
            "id": point_id,
            "x": place,
            "y": score,
            "count": 1,
            "leftPointID": point_id - 1 if point_id else None,
            "rightPointID": point_id + 1 if point_id < last else None,
        }
        for point_id, (place, score) in enumerate(zip(places, scores, strict=True))
    }


def _nominal_points(ebm: ExplainableBoostingClassifier, name: str, scores: Mapping[str, float]) -> Document:
    mapping = dict(ebm.bins_[list(ebm.feature_names_in_).index(name)][0])
    return {
        str(code): {"id": code, "x": level, "y": scores[level], "count": 1, "error": 0.0}
        for level, code in mapping.items()
    }


def _entry(feature: str, commit_type: str, points: Document, **fields: JSONValue) -> Document:
    """Build a history entry. The keyword fields replace the defaults: description, hash and reviewed.

    Returns
    -------
    Document
        The entry.

    """
    defaults: Document = {"description": "An edit", "hash": "0123456789abcdef", "reviewed": True}
    return (
        {
            "featureName": feature,
            "type": commit_type,
            "time": TIME_MS,
            "metrics": {"barData": {}, "confusionMatrixData": {}},
            "state": {"pointData": points},
        }
        | defaults
        | fields
    )


def _original(ebm: ExplainableBoostingClassifier, name: str) -> Document:
    if name == "group":
        levels = dict(ebm.bins_[2][0])
        scores = dict(zip(levels, _rounded(ebm, name), strict=True))
        return _entry(name, "original", _nominal_points(ebm, name, scores), description="Original graph")
    return _entry(name, "original", _continuous_points(_left_edges(ebm, name), _rounded(ebm, name)))


def _document(ebm: ExplainableBoostingClassifier, *entries: Document) -> Document:
    history = [_original(ebm, name) for name in ("age", "score", "group")]
    return {"modelData": _model_data(ebm), "sampleData": None, "historyList": [*history, *entries]}


def _write(tmp_path: Path, document: Document) -> Path:
    path = tmp_path / "edit.gamchanger"
    path.write_bytes(orjson.dumps(document))
    return path


def _age_edit(ebm: ExplainableBoostingClassifier, changes: Mapping[int, float], **fields: JSONValue) -> Document:
    scores = _rounded(ebm, "age")
    for bin_number, score in changes.items():
        scores[bin_number] = score
    return _entry("age", "increasing", _continuous_points(_left_edges(ebm, "age"), scores), **fields)


def test_a_history_with_originals_only_gives_no_commit(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    assert load_gamchanger(_write(tmp_path, _document(gamchanger_ebm)), gamchanger_ebm) == []


def test_a_continuous_commit_becomes_one_edit_of_the_bins_that_it_changed(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    entry = _age_edit(gamchanger_ebm, {3: 0.1, 4: 0.2, 5: 0.2}, description="Made 3 bins increasing.", reviewed=False)
    (commit,) = load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, entry)), gamchanger_ebm)
    assert commit == GamChangerCommit(
        feature="age",
        commit_type="increasing",
        description="Made 3 bins increasing.",
        reviewed=False,
        time=datetime.fromtimestamp(TIME_MS / 1000, UTC),
        commit_hash="0123456789abcdef",
        edits=(AssignScores("age", Bins(3, 5), (0.1, 0.2, 0.2)),),
    )


def test_each_range_of_changed_neighbour_bins_becomes_its_own_edit(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    entry = _age_edit(gamchanger_ebm, {0: 0.5, 1: 0.5, 6: -0.25})
    (commit,) = load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, entry)), gamchanger_ebm)
    assert commit.edits == (AssignScores("age", Bins(0, 1), (0.5, 0.5)), AssignScores("age", Bins(6, 6), (-0.25,)))


def test_the_newest_commit_of_each_feature_wins_and_an_original_entry_never_does(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    older = _age_edit(gamchanger_ebm, {2: 9.0}, hash="older")
    newer = _age_edit(gamchanger_ebm, {2: 1.5}, hash="newer")
    reloaded = _original(gamchanger_ebm, "age")  # GAM Changer adds such an entry when a person opens a feature again
    nominal = _entry("group", "align", _nominal_points(gamchanger_ebm, "group", _group_scores(gamchanger_ebm, y=0.5)))
    document = _document(gamchanger_ebm, older, nominal, newer, reloaded)
    commits = load_gamchanger(_write(tmp_path, document), gamchanger_ebm)
    assert [(commit.feature, commit.commit_hash) for commit in commits] == [
        ("group", "0123456789abcdef"),
        ("age", "newer"),
    ]
    assert commits[1].edits == (AssignScores("age", Bins(2, 2), (1.5,)),)


def _group_scores(ebm: ExplainableBoostingClassifier, **changes: float) -> dict[str, float]:
    levels = dict(ebm.bins_[2][0])
    return dict(zip(levels, _rounded(ebm, "group"), strict=True)) | changes


def test_a_nominal_commit_becomes_one_edit_of_the_categories_that_it_changed_in_the_order_of_the_bins(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    points = _nominal_points(gamchanger_ebm, "group", _group_scores(gamchanger_ebm, z=-1.0, x=2.0))
    (commit,) = load_gamchanger(
        _write(tmp_path, _document(gamchanger_ebm, _entry("group", "align", points))),
        gamchanger_ebm,
    )
    assert commit.edits == (AssignScores("group", Categories(("x", "z")), (2.0, -1.0)),)


def test_a_commit_that_changes_no_score_is_left_out(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    entry = _age_edit(gamchanger_ebm, {})
    nominal = _entry("group", "align", _nominal_points(gamchanger_ebm, "group", _group_scores(gamchanger_ebm)))
    assert load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, entry, nominal)), gamchanger_ebm) == []


def test_apply_sets_the_scores_of_the_file_and_keeps_the_full_precision_of_the_other_bins(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    age = _age_edit(gamchanger_ebm, {3: 0.1, 4: 0.2}, description="Made 2 bins increasing.", hash="aaaaaaa111")
    points = _nominal_points(gamchanger_ebm, "group", _group_scores(gamchanger_ebm, y=0.5))
    group = _entry("group", "align", points, reviewed=False, hash="bbbbbbb222")
    commits = load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, age, group)), gamchanger_ebm)
    session = EditSession(gamchanger_ebm, clock=fixed_clock())
    reports = apply_gamchanger(session, commits)
    assert len(reports) == len(session.history_table()) == 2
    rows = session.history_table()
    assert [row.reason for row in rows] == [
        "GAM Changer commit aaaaaaa: Made 2 bins increasing.",
        "GAM Changer commit bbbbbbb: An edit",
    ]
    assert [row.reviewed for row in rows] == [True, False]
    before = gamchanger_ebm.term_scores_[_main_term(gamchanger_ebm, 0)]
    after = session.model.term_scores_[_main_term(session.model, 0)]
    assert after[4:6].tolist() == [0.1, 0.2]
    unchanged = np.r_[0:4, 6 : len(before)]
    assert np.array_equal(after[unchanged], before[unchanged])
    # get_edited_model of GAM Changer writes the rounded score to every bin: the result is within 0.00005 of it
    assert np.abs(after[1:-1] - np.round(after[1:-1], 4)).max() <= 0.00005


def _equal_bins(
    ebm: ExplainableBoostingClassifier,
    first: int,
    stop: int,
    new_scores: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Make the points of an equal-interpolate commit: bins ``first`` to ``stop - 1`` become equal bins.

    Returns
    -------
    tuple[list[float], list[float]]
        The left edges and the scores of the commit, as GAM Changer writes them.

    """
    places, scores = _left_edges(ebm, "age"), _rounded(ebm, "age")
    right = places[stop] if stop < len(places) else float(ebm.feature_bounds_[0][1])
    new_places = np.linspace(places[first], right, len(new_scores) + 1)[:-1].tolist()
    return places[:first] + new_places + places[stop:], scores[:first] + list(new_scores) + scores[stop:]


@pytest.mark.parametrize(("first", "stop"), [(2, 8), (4, None)], ids=["middle", "to the last bin"])
def test_an_equal_bins_commit_becomes_a_rebin_and_the_scores_of_the_commit(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
    first: int,
    stop: int | None,
) -> None:
    bin_count = len(_left_edges(gamchanger_ebm, "age"))
    end = bin_count if stop is None else stop
    places, scores = _equal_bins(gamchanger_ebm, first, end, [0.1, 0.2, 0.3])
    entry = _entry("age", "equal-interpolate", _continuous_points(places, scores))
    (commit,) = load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, entry)), gamchanger_ebm)
    assert commit.edits == (
        Rebin("age", Bins(first, end - 1), 3),
        AssignScores("age", Bins(first, first + 2), (0.1, 0.2, 0.3)),
    )
    session = EditSession(gamchanger_ebm, clock=fixed_clock())
    apply_gamchanger(session, [commit])
    edited = session.model
    assert np.asarray(edited.bins_[0][0], dtype=float) == pytest.approx(places[1:])
    assert np.round(edited.term_scores_[0][1:-1], 4).tolist() == pytest.approx(scores)


def _load_and_apply(
    ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
    entry: Document,
) -> tuple[GamChangerCommit, ExplainableBoostingClassifier | ExplainableBoostingRegressor]:
    (commit,) = load_gamchanger(_write(tmp_path, _document(ebm, entry)), ebm)
    session = EditSession(ebm, clock=fixed_clock())
    apply_gamchanger(session, [commit])
    return commit, session.model


def test_new_bins_of_unequal_width_become_merges_and_splits(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    places, scores = _left_edges(gamchanger_ebm, "age"), _rounded(gamchanger_ebm, "age")
    split_at = (places[5] + places[6]) / 3 + places[5] / 3
    new_places = [*places[:2], *places[4:6], split_at, *places[6:]]
    new_scores = [*scores[:2], scores[2], scores[4], scores[5], scores[5], *scores[6:]]
    new_scores = new_scores[: len(new_places)]
    new_scores[2] = 0.75
    commit, edited = _load_and_apply(
        gamchanger_ebm,
        tmp_path,
        _entry("age", "SplitBin", _continuous_points(new_places, new_scores)),
    )
    assert commit.edits[:2] == (Rebin("age", Bins(1, 3), 1), SplitBin("age", split_at))
    assert np.asarray(edited.bins_[0][0], dtype=float) == pytest.approx(new_places[1:])
    assert np.round(edited.term_scores_[0][1:-1], 4).tolist() == pytest.approx(new_scores)


def test_a_wider_range_becomes_an_extension(gamchanger_ebm: ExplainableBoostingClassifier, tmp_path: Path) -> None:
    places, scores = _left_edges(gamchanger_ebm, "age"), _rounded(gamchanger_ebm, "age")
    old_high = float(gamchanger_ebm.feature_bounds_[0][1])
    points = _continuous_points([places[0] - 1.0, *places, old_high], [scores[0], *scores, 2.0])
    _dict_of(points[str(len(places) + 1)])["maxX"] = old_high + 3.0
    commit, edited = _load_and_apply(gamchanger_ebm, tmp_path, _entry("age", "ExtendRange", points))
    assert commit.edits[0] == ExtendRange("age", places[0] - 1.0, old_high + 3.0)
    assert np.asarray(edited.feature_bounds_[0]).tolist() == [places[0] - 1.0, old_high + 3.0]
    assert float(edited.term_scores_[0][-2]) == pytest.approx(2.0)


def _dict_of(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def test_a_commit_that_narrows_the_range_is_refused(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    places, scores = _left_edges(gamchanger_ebm, "age"), _rounded(gamchanger_ebm, "age")
    places[0] += (places[1] - places[0]) / 2
    path = _write(tmp_path, _document(gamchanger_ebm, _entry("age", "move", _continuous_points(places, scores))))
    with pytest.raises(HistoryError, match=r"The commit 0123456 \(move\) narrows the range of 'age'"):
        load_gamchanger(path, gamchanger_ebm)


def test_a_nominal_commit_with_a_category_that_the_model_lacks_is_refused(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    points = _nominal_points(gamchanger_ebm, "group", _group_scores(gamchanger_ebm))
    points["9"] = {"id": 9, "x": "w", "y": 1.0, "count": 1, "error": 0.0}
    path = _write(tmp_path, _document(gamchanger_ebm, _entry("group", "align", points)))
    with pytest.raises(HistoryError, match=r"names categories of 'group' that the model lacks: \['w'\]"):
        load_gamchanger(path, gamchanger_ebm)


def _feature(document: Document, name: str) -> dict[str, JSONValue]:
    model_data = document["modelData"]
    assert isinstance(model_data, dict)
    features = model_data["features"]
    assert isinstance(features, list)
    found = next(feature for feature in features if isinstance(feature, dict) and feature["name"] == name)
    assert isinstance(found, dict)
    return found


def test_a_file_of_another_model_is_refused_with_the_part_that_differs(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    def refused(change: str, document: Document) -> str:
        with pytest.raises(HistoryError) as caught:
            load_gamchanger(_write(tmp_path, document), gamchanger_ebm)
        return f"{change}: {caught.value}"

    other_names = _document(gamchanger_ebm)
    _feature(other_names, "score")["name"] = "height"
    other_edges = _document(gamchanger_ebm)
    edges = _feature(other_edges, "age")["binEdge"]
    assert isinstance(edges, list)
    _feature(other_edges, "age")["binEdge"] = [*edges[:-1], 99.0]
    other_scores = _document(gamchanger_ebm)
    _feature(other_scores, "score")["additive"] = [0.0] * 3
    other_categories = _document(gamchanger_ebm)
    _feature(other_categories, "group")["binLabel"] = [2, 1, 3]
    unknown_code = _document(gamchanger_ebm)
    _feature(unknown_code, "group")["binLabel"] = [1, 2, 7]
    other_type = _document(gamchanger_ebm)
    _feature(other_type, "score")["type"] = "categorical"
    messages = [
        refused("names", other_names),
        refused("edges", other_edges),
        refused("scores", other_scores),
        refused("categories", other_categories),
        refused("code", unknown_code),
        refused("type", other_type),
    ]
    assert (
        "the file has the features ['age', 'group', 'height'], and the model has ['age', 'group', 'score']"
        in messages[0]
    )
    assert "the bin edges of 'age' differ" in messages[1]
    assert "the scores of 'score' differ" in messages[2]
    assert "the categories of 'group' differ" in messages[3]
    assert "the categories of 'group' differ" in messages[4]
    assert "the feature types of 'score' differ" in messages[5]


def test_a_model_with_an_ordinal_feature_is_refused(
    edit_ebm: ExplainableBoostingClassifier,
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    document = _document(gamchanger_ebm)
    model_data = document["modelData"]
    assert isinstance(model_data, dict)
    model_data["features"] = [
        {"name": name, "type": "continuous" if feature_type == "continuous" else "categorical"}
        for name, feature_type in zip(edit_ebm.feature_names_in_, edit_ebm.feature_types_in_, strict=True)
    ]
    with pytest.raises(
        HistoryError,
        match="GAM Changer edits continuous and nominal features only, and 'grade' is ordinal",
    ):
        load_gamchanger(_write(tmp_path, document), edit_ebm)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "is not JSON, so it is not a .gamchanger document"),
        (b"[1, 2]", "The document of the .gamchanger file is not a JSON object"),
        (b'{"modelData": {}}', "The model data of the .gamchanger file has no field 'features'"),
        (b'{"modelData": {"features": "all"}}', "The features of the .gamchanger file is not a JSON list"),
    ],
)
def test_a_file_that_is_not_a_gamchanger_document_is_refused(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    path = tmp_path / "edit.gamchanger"
    path.write_bytes(content)
    with pytest.raises(HistoryError, match=message):
        load_gamchanger(path, gamchanger_ebm)


def test_a_damaged_document_is_refused(gamchanger_ebm: ExplainableBoostingClassifier, tmp_path: Path) -> None:
    looped = _age_edit(gamchanger_ebm, {1: 0.3})
    state = looped["state"]
    assert isinstance(state, dict)
    points = copy.deepcopy(state["pointData"])
    assert isinstance(points, dict)
    last = points[str(len(points) - 1)]
    assert isinstance(last, dict)
    last["rightPointID"] = 0
    looped["state"] = {"pointData": points}
    with pytest.raises(HistoryError, match=r"The point list of a commit in the \.gamchanger file does not end"):
        load_gamchanger(_write(tmp_path, _document(gamchanger_ebm, looped)), gamchanger_ebm)
    letters = _document(gamchanger_ebm)
    _feature(letters, "score")["additive"] = ["a", "b"]
    with pytest.raises(HistoryError, match=r"The scores of the \.gamchanger file holds a value that is not a number"):
        load_gamchanger(_write(tmp_path, letters), gamchanger_ebm)
