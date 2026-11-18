"""Tests for the writer of .gamchanger files.

The expected fields follow a file that GAM Changer 0.1.13 saved: ``get_model_data`` of its Python package for
``modelData``, and the function of the widget that draws a graph (``Sr`` in ``gamchanger.js``) for the line segments.
"""

import warnings
from pathlib import Path

import numpy as np
import orjson
import pytest

from ebaml.core.custom_types import JSONValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.edit.bin_edits import Rebin
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import Monotonize, RemoveFeature, RemoveTerms, SetScores, Shift
from ebaml.edit.gamchanger_export import save_gamchanger
from ebaml.edit.gamchanger_files import apply_gamchanger, load_gamchanger
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, MissingBin
from tests.unit.ebaml.conftest import PandasFrame, as_frame
from tests.unit.ebaml.edit.conftest import fixed_clock

type Document = dict[str, JSONValue]


def _session(ebm: ExplainableBoostingClassifier) -> EditSession:
    session = EditSession(ebm, clock=fixed_clock())
    session.apply(Monotonize("age", "decreasing", Bins(2, 8)), "Risk must not rise with age")
    session.apply(Shift("score", Bins(1, 3), 0.25), "Low scores were under-estimated")
    session.apply(SetScores("group", Categories(["y"]), 0.5), "Too few rows")
    session.review("current")
    return session


def _read(path: Path) -> Document:
    document = orjson.loads(path.read_bytes())
    assert isinstance(document, dict)
    return document


def _list(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def _dict(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def test_the_file_holds_the_original_model_as_gamchanger_writes_it(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    save_gamchanger(_session(gamchanger_ebm), tmp_path / "edit.gamchanger")
    model_data = _dict(_read(tmp_path / "edit.gamchanger")["modelData"])
    features = {_dict(feature)["name"]: _dict(feature) for feature in _list(model_data["features"])}
    assert set(features) == {"age", "score", "group"}
    age, group = term_view(gamchanger_ebm, "age"), term_view(gamchanger_ebm, "group")
    assert features["age"]["type"] == "continuous"
    assert features["age"]["additive"] == np.round(age.scores[1:-1], 4).tolist()
    assert features["age"]["binEdge"] == [left for left, _ in age.bin_edges()] + [age.bin_edges()[-1][1]]
    assert features["age"]["count"] == age.weights[1:-1].tolist()
    assert features["group"]["type"] == "categorical"
    assert features["group"]["binLabel"] == [1, 2, 3]
    assert (
        _dict(model_data["labelEncoder"])["group"]
        == {"1": "x", "2": "y", "3": "z"}
        == {str(code): name for name, code in dict(gamchanger_ebm.bins_[2][0]).items()}
    )
    assert group.categories() == ["x", "y", "z"]
    assert model_data["isClassifier"] is True
    assert model_data["intercept"] == pytest.approx(float(gamchanger_ebm.intercept_[0]))


def _reference_segments(places: list[float], scores: list[float], upper: float) -> list[Document]:
    """Build the line segments as the function ``Sr`` of the widget does: a step to the right, then a step up or down.

    Returns
    -------
    list[Document]
        The segments, in the order of the widget.

    """
    segments: list[Document] = []
    for number, (place, score) in enumerate(zip(places, scores, strict=True)):
        if number + 1 < len(places):
            right, next_score = places[number + 1], scores[number + 1]
            ends: Document = {"sx": place, "sy": score, "tx": right, "ty": next_score}
            step = {
                "x1": place,
                "y1": score,
                "x2": right,
                "y2": score,
                "id": f"path-{number}-{number + 1}-r",
                "pos": "r",
            }
            rise = {
                "x1": right,
                "y1": score,
                "x2": right,
                "y2": next_score,
                "id": f"path-{number}-{number + 1}-l",
                "pos": "l",
            }
            segments.extend((step | ends, rise | ends))
        else:
            ends = {"sx": place, "sy": score, "tx": upper, "ty": score}
            last: Document = {
                "x1": place,
                "y1": score,
                "x2": upper,
                "y2": score,
                "id": f"path-{number}-{number}-r",
                "pos": "r",
            }
            segments.append(last | ends)
    return segments


def test_each_edit_becomes_a_commit_after_an_original_commit_of_its_feature(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(gamchanger_ebm)
    result = save_gamchanger(session, tmp_path / "edit.gamchanger")
    history = [_dict(entry) for entry in _list(_read(tmp_path / "edit.gamchanger")["historyList"])]
    assert [(entry["featureName"], entry["type"]) for entry in history] == [
        ("age", "original"),
        ("age", "decreasing"),
        ("score", "original"),
        ("score", "move"),
        ("group", "original"),
        ("group", "align"),
    ]
    assert result.commit_count == 3
    assert [entry["reviewed"] for entry in history] == [True, False, True, False, True, True]
    assert history[3]["description"] == session.history_table()[1].description
    assert history[1]["time"] == round(session.history_table()[0].time.timestamp() * 1000)
    assert all(isinstance(entry["hash"], str) and len(entry["hash"]) == 32 for entry in history)
    edited = term_view(session.model_at(session.history_table()[0].id), "age")
    state = _dict(history[1]["state"])
    points = _dict(state["pointData"])
    places = [left for left, _ in edited.bin_edges()]
    scores = np.round(edited.scores[1:-1], 4).tolist()
    assert [_dict(point)["y"] for point in points.values()] == scores
    last = _dict(points[str(len(points) - 1)])
    assert (last["rightPointID"], last["maxX"]) == (None, edited.bin_edges()[-1][1])
    assert state["additiveData"] == _reference_segments(places, scores, edited.bin_edges()[-1][1])
    group_points = _dict(_dict(history[5]["state"])["pointData"])
    assert _dict(group_points["2"])["x"] == "y"
    assert _dict(group_points["2"])["y"] == pytest.approx(0.5)


def test_a_second_edit_of_a_feature_adds_no_second_original_commit(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(gamchanger_ebm)
    session.apply(Shift("age", Bins(0, 1), -0.1), "The youngest were over-estimated")
    save_gamchanger(session, tmp_path / "edit.gamchanger")
    history = [_dict(entry) for entry in _list(_read(tmp_path / "edit.gamchanger")["historyList"])]
    assert [entry["type"] for entry in history if entry["featureName"] == "age"] == ["original", "decreasing", "move"]


def test_a_regressor_writes_its_labels_as_numbers_and_complete_data_leaves_out_no_row(
    edit_data: PandasFrame,
    tmp_path: Path,
) -> None:
    data = edit_data[["score", "group"]]
    target = 2 * data["score"].to_numpy() + (data["group"] == "x").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regressor = ExplainableBoostingRegressor(interactions=0, random_state=0, outer_bags=2).fit(data, target)
    session = EditSession(regressor, clock=fixed_clock())
    session.apply(Shift("score", Bins(0, 2), 0.5), "Low scores were under-estimated")
    result = save_gamchanger(session, tmp_path / "edit.gamchanger", as_frame(data), target)
    document = _read(tmp_path / "edit.gamchanger")
    assert _dict(document["modelData"])["isClassifier"] is False
    assert _list(_dict(document["sampleData"])["labels"]) == pytest.approx(target.tolist())
    assert result.left_out == ()


def test_the_file_reads_back_as_the_edits_of_the_session(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(gamchanger_ebm)
    save_gamchanger(session, tmp_path / "edit.gamchanger")
    reloaded = EditSession(gamchanger_ebm, clock=fixed_clock())
    apply_gamchanger(reloaded, load_gamchanger(tmp_path / "edit.gamchanger", gamchanger_ebm))
    for feature in ("age", "score", "group"):
        assert term_view(reloaded.model, feature).scores[1:-1] == pytest.approx(
            term_view(session.model, feature).scores[1:-1],
            abs=5e-5,
        )


def test_without_data_the_file_holds_one_synthetic_row_for_each_bin_number_with_the_predicted_class(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    result = save_gamchanger(_session(gamchanger_ebm), tmp_path / "edit.gamchanger")
    samples = _dict(_read(tmp_path / "edit.gamchanger")["sampleData"])
    rows = _list(samples["samples"])
    age = term_view(gamchanger_ebm, "age")
    assert result.synthetic_samples
    assert (
        result.sample_count == len(rows) == max(age.real_bin_count, term_view(gamchanger_ebm, "score").real_bin_count)
    )
    first = _list(rows[0])
    assert first[0] == pytest.approx(sum(age.bin_edges()[0]) / 2)
    assert first[2] == 1  # the code of the first category
    assert set(_list(samples["labels"])) <= {0, 1}
    assert samples["featureTypes"] == ["continuous", "continuous", "categorical"]


def test_with_data_the_file_holds_the_complete_rows_and_names_the_rows_that_it_leaves_out(
    gamchanger_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    tmp_path: Path,
) -> None:
    data = edit_data[["age", "score", "group"]].copy()
    data.loc[30, "group"] = "w"
    labels = np.where(np.arange(len(data)) % 2 == 0, 1, 0)
    result = save_gamchanger(_session(gamchanger_ebm), tmp_path / "edit.gamchanger", as_frame(data), labels)
    samples = _dict(_read(tmp_path / "edit.gamchanger")["sampleData"])
    missing = data["age"].isna().sum()
    assert not result.synthetic_samples
    assert result.sample_count == len(data) - missing
    assert f"{missing} rows with a missing value, which GAM Changer cannot read" in result.left_out
    rows = [_list(row) for row in _list(samples["samples"])]
    complete = data[data["age"].notna()]
    assert rows[0][0] == pytest.approx(complete.iloc[0]["age"])
    assert rows[30 - missing][2] == 4  # an unseen category takes the code after the last one
    assert _list(samples["labels"]) == labels[data["age"].notna().to_numpy()].tolist()


def test_the_result_names_what_the_file_cannot_hold(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = EditSession(gamchanger_ebm, clock=fixed_clock())
    session.apply(SetScores("age", MissingBin(), 1.0), "A missing age scores 1")
    session.apply(RemoveTerms([name for name in gamchanger_ebm.term_names_ if " & " in name]), "No pair terms")
    session.apply(Rebin("score", Bins(0, 3), 2), "Smooth the low scores")
    result = save_gamchanger(session, tmp_path / "edit.gamchanger")
    assert result.commit_count == 2
    assert any("which names terms and not one graph" in part for part in result.left_out)
    assert "the edited scores of the missing bin and the unseen bin of age" in result.left_out
    pair = next(name for name in gamchanger_ebm.term_names_ if " & " in name)
    assert any(pair in part and "metrics of the widget" in part for part in result.left_out)
    history = [_dict(entry) for entry in _list(_read(tmp_path / "edit.gamchanger")["historyList"])]
    assert history[-1]["type"] == "Rebin"


def test_a_removed_feature_becomes_a_graph_of_zeros(
    gamchanger_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = EditSession(gamchanger_ebm, clock=fixed_clock())
    session.apply(RemoveFeature("group"), "The model must not use group")
    save_gamchanger(session, tmp_path / "edit.gamchanger")
    history = [_dict(entry) for entry in _list(_read(tmp_path / "edit.gamchanger")["historyList"])]
    assert history[-1]["type"] == "RemoveFeature"
    points = _dict(_dict(history[-1]["state"])["pointData"])
    assert {_dict(point)["y"] for point in points.values()} == {0.0}
    commits = load_gamchanger(tmp_path / "edit.gamchanger", gamchanger_ebm)
    reloaded = EditSession(gamchanger_ebm, clock=fixed_clock())
    apply_gamchanger(reloaded, commits)
    assert term_view(reloaded.model, "group").scores[1:-1].tolist() == [0.0, 0.0, 0.0]


def test_the_writer_refuses_an_ordinal_feature_a_path_that_exists_and_data_that_does_not_fit(
    edit_ebm: ExplainableBoostingClassifier,
    gamchanger_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    tmp_path: Path,
) -> None:
    with pytest.raises(UnsupportedModelError, match="GAM Changer edits continuous and nominal features only"):
        save_gamchanger(EditSession(edit_ebm), tmp_path / "ordinal.gamchanger")
    session = EditSession(gamchanger_ebm)
    (tmp_path / "taken.gamchanger").write_bytes(b"{}")
    with pytest.raises(FileExistsError):
        save_gamchanger(session, tmp_path / "taken.gamchanger")
    with pytest.raises(InvalidInstanceError, match=r"The data has no column for these features: \['group'\]"):
        save_gamchanger(session, tmp_path / "a.gamchanger", as_frame(edit_data[["age", "score"]]))
    with pytest.raises(InvalidInstanceError, match="the data has 600 rows and the labels 2"):
        save_gamchanger(session, tmp_path / "b.gamchanger", as_frame(edit_data), [0, 1])
