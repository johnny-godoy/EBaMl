"""Tests for the records of the edits and the selections."""

import orjson
import pytest

from ebaml.core.custom_types import JSONValue
from ebaml.core.exceptions import HistoryError
from ebaml.edit.edit_records import EDIT_TYPES, SELECTION_TYPES, edit_from_record, to_record
from ebaml.edit.edits import (
    Align,
    Edit,
    FitLine,
    Interpolate,
    Monotonize,
    Recenter,
    RemoveFeature,
    RemoveTerms,
    Scale,
    SetScores,
    Shift,
    ZeroTerm,
)
from ebaml.edit.selections import Bins, Categories, MissingBin, UnseenBin, ValueInterval, WholeGraph

EXAMPLES: list[Edit] = [
    Shift("age", Bins(1, 3), 0.5),
    SetScores("age", MissingBin(), 0.0),
    SetScores("group", Categories(["x", "y"]), -1.0),
    Align("age", UnseenBin(), "mean"),
    Monotonize("score", "auto", ValueInterval(-1.0, None)),
    Monotonize("score", "increasing"),
    Interpolate("score", Bins(2, 9)),
    FitLine("score", ValueInterval()),
    Scale("score", -2.0),
    ZeroTerm("grade"),
    Recenter("grade"),
    RemoveTerms(["age", "score & grade"]),
    RemoveFeature("group"),
]


def test_every_edit_class_and_every_selection_class_is_in_the_registry() -> None:
    assert {type(edit).__name__ for edit in EXAMPLES} == set(EDIT_TYPES)
    assert {"Bins", "Categories", "MissingBin", "UnseenBin", "WholeGraph", "ValueInterval"} == set(SELECTION_TYPES)
    assert WholeGraph().__class__ is SELECTION_TYPES["WholeGraph"]


@pytest.mark.parametrize("edit", EXAMPLES, ids=lambda edit: type(edit).__name__)
def test_an_edit_survives_a_round_trip_through_json_text(edit: Edit) -> None:
    text = orjson.dumps(to_record(edit))
    assert edit_from_record(orjson.loads(text)) == edit


def test_a_record_names_the_class_and_holds_a_selection_as_a_record() -> None:
    assert to_record(Shift("age", Bins(1, 3), 0.5)) == {
        "type": "Shift",
        "feature": "age",
        "selection": {"type": "Bins", "first": 1, "last": 3},
        "delta": 0.5,
    }
    assert to_record(RemoveTerms(["a", "b"]))["terms"] == ("a", "b")
    assert to_record(ValueInterval(None, 2.0)) == {"type": "ValueInterval", "low": None, "high": 2.0}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("Shift", "needs a text under type"),
        ({"feature": "age"}, "needs a text under type"),
        ({"type": 3}, "needs a text under type"),
        ({"type": "Explode"}, "names the edit 'Explode', which this version does not know"),
        ({"type": "Shift", "feature": "age"}, "does not fit its class"),
        (
            {"type": "Align", "feature": "age", "selection": {"type": "WholeGraph"}, "to": "middle"},
            "does not fit its class",
        ),
        (
            {"type": "Shift", "feature": "age", "selection": {"type": "Teleport"}, "delta": 1.0},
            "names the selection 'Teleport'",
        ),
    ],
)
def test_a_record_that_does_not_fit_raises_a_history_error(record: JSONValue, message: str) -> None:
    with pytest.raises(HistoryError, match=message):
        edit_from_record(record)
