"""Tests for the entries, the table and the JSON form of the history."""

import copy
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import attrs
import orjson
import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import HistoryError
from ebaml.edit.edits import Shift
from ebaml.edit.history import (
    FORMAT_VERSION,
    EditEntry,
    SaveEntry,
    decode_history,
    encode_history,
    fingerprint,
    short_ids,
)
from ebaml.edit.selections import Bins
from tests.unit.ebaml.edit.conftest import move_cuts, rename_categories

TIME: datetime = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
EDIT_ID: uuid.UUID = uuid.UUID("018f0000-0000-7000-8000-00000000abcd")
SAVE_ID: uuid.UUID = uuid.UUID("018f0000-0000-7000-8000-00000000ef01")


def _edit_entry(*, reviewed: bool = False) -> EditEntry:
    return EditEntry(
        id=EDIT_ID,
        sequence=0,
        time=TIME,
        message="The risk must not fall with the score",
        edit=Shift("score", Bins(1, 2), 0.5),
        description="Add 0.5 to the scores of 2 bins of score (1 to 2)",
        feature="score",
        reviewed=reviewed,
    )


def _save_entry() -> SaveEntry:
    return SaveEntry(
        id=SAVE_ID,
        sequence=1,
        time=TIME,
        message="Release 1",
        author="Ada",
        edit_ids=(EDIT_ID,),
        reviewed=True,
        revision=str(EDIT_ID),
        file="model.skops",
        model_hash="abc",
    )


def test_the_row_of_an_edit_entry_names_the_edit_the_feature_and_the_reason() -> None:
    row = _edit_entry().row("abcd")
    assert (row.kind, row.edit_type, row.feature, row.reviewed, row.short_id) == (
        "edit",
        "Shift",
        "score",
        False,
        "abcd",
    )
    assert row.reason == "The risk must not fall with the score"
    assert row.description.startswith("Add 0.5")
    assert (row.id, row.sequence, row.time) == (str(EDIT_ID), 0, TIME)


def test_the_row_of_a_save_entry_names_the_revision_and_the_file() -> None:
    row = _save_entry().row("ef01")
    assert (row.kind, row.edit_type, row.feature, row.reason) == ("save", None, None, "Release 1")
    assert row.description == f"Save the revision {EDIT_ID} as model.skops"
    assert row.reviewed


def test_short_ids_take_the_shortest_unique_prefix_of_the_random_tail() -> None:
    first = uuid.UUID("018f0000-0000-7000-8000-000000000001")
    second = uuid.UUID("018f0000-0000-7000-8000-000000000002")
    other = uuid.UUID("018f0000-0000-7abc-8000-000000000003")
    shorts = short_ids([first, second, other])
    assert shorts[other] == "abc8"
    # the tails of the first two ids differ in the last of their 19 characters, so the prefix needs all of them
    assert len(shorts[first]) == len(shorts[second]) == 19
    assert shorts[first] != shorts[second]
    assert short_ids([]) == {}


def test_a_history_survives_a_round_trip_through_json(edit_ebm: ExplainableBoostingClassifier) -> None:
    original = fingerprint(edit_ebm)
    entries = [_edit_entry(reviewed=True), _save_entry()]
    decoded = decode_history(encode_history(original, str(EDIT_ID), entries))
    assert decoded.fingerprint == original
    assert decoded.head == str(EDIT_ID)
    assert decoded.edits == (entries[0],)
    assert decoded.saves == (entries[1],)
    empty = decode_history(encode_history(original, "original", []).decode())
    assert (empty.edits, empty.saves) == ((), ())


def test_a_save_entry_without_an_author_survives_a_round_trip(edit_ebm: ExplainableBoostingClassifier) -> None:
    anonymous = attrs.evolve(_save_entry(), author=None)
    decoded = decode_history(encode_history(fingerprint(edit_ebm), "original", [anonymous]))
    assert decoded.saves == (anonymous,)


def test_the_fingerprint_is_the_same_for_a_copy_and_changes_with_a_score(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    assert fingerprint(copy.deepcopy(edit_ebm)) == fingerprint(edit_ebm)
    edited = copy.deepcopy(edit_ebm)
    edited.term_scores_[1] += 0.001
    changed = fingerprint(edited)
    assert changed.scores != fingerprint(edit_ebm).scores
    assert (changed.features, changed.terms, changed.bins) == (
        fingerprint(edit_ebm).features,
        fingerprint(edit_ebm).terms,
        fingerprint(edit_ebm).bins,
    )


def test_the_fingerprint_changes_with_a_bin_edge_a_category_and_a_feature_bound(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    reference = fingerprint(edit_ebm).bins
    moved = copy.deepcopy(edit_ebm)
    move_cuts(moved, 1, 1e-3)
    assert fingerprint(moved).bins != reference
    renamed = copy.deepcopy(edit_ebm)
    rename_categories(renamed, 2)
    assert fingerprint(renamed).bins != reference
    bounded = copy.deepcopy(edit_ebm)
    bounded.feature_bounds_[1] = (bounded.feature_bounds_[1][0] - 1.0, bounded.feature_bounds_[1][1])
    assert fingerprint(bounded).bins != reference


def test_the_fingerprint_names_the_features_and_the_terms(edit_ebm: ExplainableBoostingClassifier) -> None:
    fingerprint_ = fingerprint(edit_ebm)
    assert fingerprint_.features == ("age", "score", "group", "grade")
    assert fingerprint_.terms == tuple(edit_ebm.term_names_)


def _valid_document(edit_ebm: ExplainableBoostingClassifier) -> dict[str, Any]:
    document: dict[str, Any] = orjson.loads(encode_history(fingerprint(edit_ebm), "original", [_edit_entry()]))
    return document


@pytest.mark.parametrize("text", [b"not json", b"[]"])
def test_a_text_that_is_not_json_or_not_an_object_raises_a_history_error(text: bytes) -> None:
    with pytest.raises(HistoryError, match=r"not JSON|must be an object"):
        decode_history(text)


@pytest.mark.parametrize("text", [b"{}", b'{"format_version": 1}'])
def test_a_document_without_a_field_names_the_field(text: bytes) -> None:
    with pytest.raises(HistoryError, match=r"lacks the field '(format_version|fingerprint)'"):
        decode_history(text)


def test_a_history_with_another_format_version_names_both_versions(edit_ebm: ExplainableBoostingClassifier) -> None:
    document = _valid_document(edit_ebm)
    document["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(
        HistoryError,
        match=rf"format version {FORMAT_VERSION + 1}, and this version reads {FORMAT_VERSION}",
    ):
        decode_history(orjson.dumps(document))


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda entry: entry.pop(key)


def _set(
    key: str,
    value: object,
) -> Callable[[dict[str, Any]], None]:  # nosemgrep: ebaml-no-object-type -- the test sets any JSON value
    return lambda entry: entry.update({key: value})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_drop("id"), "lacks the field 'id'"),
        (_set("id", "not-a-uuid"), "which is not a UUID"),
        (_set("time", "yesterday"), "which is not an ISO 8601 time"),
        (_set("kind", "comment"), "the kind 'comment'"),
        (_drop("description"), "lacks the field 'description'"),
        (_set("message", 3), "must be a text"),
        (_set("reviewed", "yes"), "must be true or false"),
        (_set("sequence", value=True), "must be a whole number"),
        (_set("sequence", "0"), "must be a whole number"),
        (_set("feature", 3), "must be a text"),
    ],
)
def test_an_entry_with_a_bad_field_or_kind_raises_a_history_error(
    edit_ebm: ExplainableBoostingClassifier,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    document = _valid_document(edit_ebm)
    change(document["entries"][0])
    with pytest.raises(HistoryError, match=message):
        decode_history(orjson.dumps(document))


def test_a_list_field_that_is_not_a_list_of_texts_raises_a_history_error(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    document = _valid_document(edit_ebm)
    document["entries"] = [{**_save_entry().to_record(), "edit_ids": "abc"}]
    with pytest.raises(HistoryError, match="must be a list, but it is 'abc'"):
        decode_history(orjson.dumps(document))
    document["entries"] = [{**_save_entry().to_record(), "edit_ids": [1]}]
    with pytest.raises(HistoryError, match="must be a list of texts"):
        decode_history(orjson.dumps(document))
    document["fingerprint"]["features"] = "age"
    with pytest.raises(HistoryError, match="must be a list, but it is 'age'"):
        decode_history(orjson.dumps(document))


def test_an_entry_that_is_not_an_object_raises_a_history_error(edit_ebm: ExplainableBoostingClassifier) -> None:
    document = _valid_document(edit_ebm)
    document["entries"] = ["a text"]
    with pytest.raises(HistoryError, match="must be an object"):
        decode_history(orjson.dumps(document))


def test_an_edit_that_this_version_does_not_know_raises_a_history_error(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    document = _valid_document(edit_ebm)
    document["entries"][0]["edit"]["type"] = "Explode"
    with pytest.raises(HistoryError, match="Explode"):
        decode_history(orjson.dumps(document))
