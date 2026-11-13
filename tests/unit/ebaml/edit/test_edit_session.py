"""Tests for the edit session: apply, undo, redo, review, save, replay and the concurrency rules."""

import copy
import hashlib
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path

import attrs
import numpy as np
import orjson
import pytest
import skops.io
from hypothesis import given, settings
from hypothesis import strategies as st

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import (
    EditError,
    HistoryError,
    StaleRevisionError,
    UnreviewedEditError,
    UnsupportedModelError,
)
from ebaml.edit.edit_session import EditSession, RevisionInfo
from ebaml.edit.edits import Direction, Edit, EditOutcome, Monotonize, Recenter, SetScores, Shift
from ebaml.edit.history import EditEntry, SaveEntry
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, MissingBin, WholeGraph
from tests.unit.ebaml.conftest import PandasFrame
from tests.unit.ebaml.edit.conftest import assert_consistent, fixed_clock, move_cuts

type Model = ExplainableBoostingClassifier | ExplainableBoostingRegressor

DIRECTIONS: tuple[Direction, ...] = ("increasing", "decreasing", "auto")


def _ids() -> Callable[[], uuid.UUID]:
    """Give a function that returns the ids 1, 2, 3 and so on, so that each test gets the same ids.

    Returns
    -------
    Callable[[], uuid.UUID]
        The function.

    """
    counter = iter(range(1, 10_000))
    return lambda: uuid.UUID(int=(0x018F << 112) | (7 << 76) | (2 << 62) | next(counter))


def _session(ebm: Model) -> EditSession:
    return EditSession(ebm, clock=fixed_clock(), new_id=_ids())


def _scores(model: Model, feature: str) -> list[float]:
    return [float(score) for score in term_view(model, feature).scores]


def test_a_new_session_holds_a_copy_of_the_original_model(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    assert session.revision == "original"
    assert session.model is not edit_ebm
    assert session.model.term_scores_[0].tolist() == edit_ebm.term_scores_[0].tolist()
    session.model.term_scores_[0][:] = 99.0
    assert session.model.term_scores_[0].tolist() == edit_ebm.term_scores_[0].tolist()
    assert session.history == ()
    assert session.history_table() == []


def test_a_session_refuses_a_multiclass_model(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(UnsupportedModelError):
        EditSession(multiclass_ebm)


def test_a_session_on_a_monotonized_model_repairs_the_copy_and_leaves_the_model(
    interpret_monotonized: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    session = _session(interpret_monotonized)
    assert_consistent(session.model, edit_data)
    assert interpret_monotonized.bagged_scores_[1] is None


def test_apply_changes_the_model_of_the_session_and_not_the_original(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
) -> None:
    session = _session(edit_ebm)
    report = session.apply(Shift("score", Bins(2, 4), 0.5), "The score must matter more")
    assert report.bins_changed == 3
    assert _scores(session.model, "score") != _scores(edit_ebm, "score")
    assert session.revision == str(session.history[0].id)
    assert session.model_at("original").term_scores_[1].tolist() == edit_ebm.term_scores_[1].tolist()
    assert_consistent(session.model, edit_data)


def test_apply_needs_a_message(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    for message in ("", "   "):
        with pytest.raises(EditError, match="A message is needed"):
            session.apply(Shift("score", Bins(2, 4), 0.5), message)
    assert session.history == ()


def test_a_failed_edit_adds_no_entry(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    with pytest.raises(EditError, match="needs an order"):
        session.apply(Monotonize("group", "increasing"), "Impossible")
    assert session.history == ()
    assert session.revision == "original"


def test_preview_gives_the_report_of_apply_and_changes_nothing(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    edit = Monotonize("score", "increasing", Bins(5, 25))
    preview = session.preview(edit)
    assert session.history == ()
    assert session.model.term_scores_[1].tolist() == edit_ebm.term_scores_[1].tolist()
    assert session.apply(edit, "Make the range monotone") == preview


def test_preview_starts_from_the_revision_that_the_caller_names(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", WholeGraph(), 1.0), "Raise the graph")
    from_original = session.preview(Shift("score", Bins(0, 0), 1.0), revision="original")
    from_current = session.preview(Shift("score", Bins(0, 0), 1.0))
    assert from_original.changes[0].before == pytest.approx(float(term_view(edit_ebm, "score").scores[1]))
    assert from_current.changes[0].before == pytest.approx(float(term_view(edit_ebm, "score").scores[1]) + 1.0)


def test_the_history_table_lists_each_edit_with_its_reason_and_a_short_id(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "First reason")
    session.apply(SetScores("age", MissingBin(), 0.0), "Second reason")
    table = session.history_table()
    assert [(row.sequence, row.kind, row.feature, row.edit_type, row.reason, row.reviewed) for row in table] == [
        (0, "edit", "score", "Shift", "First reason", False),
        (1, "edit", "age", "SetScores", "Second reason", False),
    ]
    assert all(
        str(entry.id).replace("-", "").endswith(row.short_id) or row.short_id in entry.id.hex
        for entry, row in zip(session.history, table, strict=True)
    )
    assert table[1].time > table[0].time


def test_undo_returns_the_original_model_and_redo_returns_the_edited_model(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "Raise three bins")
    edited = session.model
    assert session.undo() == "original"
    assert session.model.term_scores_[1].tolist() == edit_ebm.term_scores_[1].tolist()
    assert session.redo() == session.revision
    assert session.model.term_scores_[1].tolist() == edited.term_scores_[1].tolist()
    assert len(session.history) == 1


def test_undo_with_nothing_to_undo_and_redo_with_nothing_to_redo_fail(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    with pytest.raises(EditError, match="no edit to undo"):
        session.undo()
    with pytest.raises(EditError, match="no edit to redo"):
        session.redo()
    session.apply(Shift("score", Bins(2, 4), 0.5), "Raise three bins")
    with pytest.raises(EditError, match="no edit to redo"):
        session.redo()


def test_an_apply_after_an_undo_drops_the_redo_stack(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    dropped = session.history[1].id
    session.undo()
    session.apply(Shift("score", Bins(10, 12), 0.5), "Three")
    assert [entry.message for entry in session.history] == ["One", "Three"]
    with pytest.raises(EditError, match="no revision"):
        session.model_at(dropped)
    with pytest.raises(EditError, match="no edit to redo"):
        session.redo()


def test_checkout_moves_the_head_and_keeps_the_later_edits_in_the_redo_stack(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    first = session.history[0].id
    assert session.checkout(first) == str(first)
    assert session.revision == str(first)
    assert session.redo() == str(session.history[1].id)
    assert session.checkout("original") == "original"


def test_model_at_reads_any_revision_without_a_checkout(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    first, second = (entry.id for entry in session.history)
    session.undo()
    assert session.revision == str(first)
    assert _scores(session.model_at(second), "score") != _scores(session.model_at(first), "score")
    assert _scores(session.model_at("current"), "score") == _scores(session.model_at(str(first)), "score")
    assert _scores(session.model_at("original"), "score") == _scores(edit_ebm, "score")
    assert session.revision == str(first)
    with pytest.raises(EditError, match=r"no revision 'nothing'. The revisions are \['original', 'current'"):
        session.model_at("nothing")


def test_revision_info_names_the_revision_and_says_what_the_head_holds(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    session = _session(edit_ebm)
    assert session.revision_info() == RevisionInfo("original", "original", 0)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    first, second = (str(entry.id) for entry in session.history)
    assert session.revision_info("current") == RevisionInfo(second, second, 2)
    session.undo()
    assert session.revision_info() == RevisionInfo(first, first, 1)
    assert session.revision_info(second) == RevisionInfo(second, first, 1)
    assert session.revision_info("original") == RevisionInfo("original", first, 1)
    with pytest.raises(EditError, match="no revision 'nothing'"):
        session.revision_info("nothing")


def test_the_pending_table_lists_the_edits_that_the_next_save_records(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    assert session.history_table(pending_only=True) == []
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    assert [row.reason for row in session.history_table(pending_only=True)] == ["One", "Two"]
    assert session.history_table(pending_only=True) == session.history_table()
    session.undo()
    assert [row.reason for row in session.history_table(pending_only=True)] == ["One"]
    session.save(tmp_path / "one.skops", "Release", allow_unreviewed=True)
    assert session.history_table(pending_only=True) == []
    session.apply(Shift("score", Bins(6, 8), 0.5), "Three")
    assert session.history_table(pending_only=True) == session.history_table()[-1:]


def test_a_call_with_a_stale_expected_revision_raises_and_changes_nothing(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    stale = "original"
    calls = [
        lambda: session.apply(Shift("score", Bins(6, 8), 0.5), "Two", expected_revision=stale),
        lambda: session.undo(expected_revision=stale),
        lambda: session.redo(expected_revision=stale),
        lambda: session.checkout("original", expected_revision=stale),
    ]
    for call in calls:
        with pytest.raises(
            StaleRevisionError,
            match=r"The head of the session is '.*', and the call expected 'original'",
        ):
            call()
    assert len(session.history) == 1
    assert session.revision == str(session.history[0].id)


def test_a_call_with_the_current_head_as_expected_revision_goes_on(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One", expected_revision="original")
    head = session.history[0].id
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two", expected_revision=head)
    session.apply(Shift("score", Bins(6, 8), 0.5), "Three", expected_revision="current")
    assert len(session.history) == 3


def test_a_preview_cannot_be_applied_to_a_model_that_changed_after_it(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.preview(Shift("score", Bins(2, 4), 0.5))
    seen = session.revision
    session.apply(Shift("score", Bins(6, 8), 0.5), "Another writer")
    with pytest.raises(StaleRevisionError):
        session.apply(Shift("score", Bins(2, 4), 0.5), "The preview", expected_revision=seen)


def test_review_marks_an_edit_and_the_original_has_none_to_review(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    session.review(session.history[0].id)
    assert [row.reviewed for row in session.history_table()] == [True, False]
    session.review()
    assert [row.reviewed for row in session.history_table()] == [True, True]
    session.checkout("original")
    with pytest.raises(EditError, match="no edit to review"):
        session.review("original")
    with pytest.raises(EditError, match="no edit to review"):
        session.review()


def test_save_writes_the_model_and_the_history_and_adds_a_save_entry(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.review()
    entry = session.save(tmp_path / "model.skops", "Release for the risk rule", author="Ada")
    assert isinstance(entry, SaveEntry)
    assert (entry.author, entry.message, entry.file, entry.reviewed) == (
        "Ada",
        "Release for the risk rule",
        "model.skops",
        True,
    )
    assert entry.edit_ids == (session.history[0].id,)
    assert entry.revision == session.revision
    data = (tmp_path / "model.skops").read_bytes()
    assert entry.model_hash == hashlib.sha256(data).hexdigest()
    loaded = skops.io.loads(data, trusted=skops.io.get_untrusted_types(data=data))
    assert loaded.term_scores_[1].tolist() == session.model.term_scores_[1].tolist()
    assert session.history[-1] == entry
    assert (tmp_path / "model.skops.history.json").read_bytes() == session.to_json()
    row = session.history_table()[-1]
    assert (row.kind, row.reason) == ("save", "Release for the risk rule")


def test_save_refuses_an_edit_that_nobody_reviewed_unless_the_caller_allows_it(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    with pytest.raises(UnreviewedEditError, match="are not reviewed"):
        session.save(tmp_path / "model.skops", "Release")
    assert list(tmp_path.iterdir()) == []
    assert len(session.history) == 1
    entry = session.save(tmp_path / "model.skops", "Release", allow_unreviewed=True)
    assert not entry.reviewed


def test_save_needs_a_message_and_a_session_with_no_edit_may_save(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    with pytest.raises(EditError, match="A message is needed"):
        session.save(tmp_path / "model.skops", " ")
    entry = session.save(tmp_path / "model.skops", "The model without edits")
    assert (entry.edit_ids, entry.revision, entry.reviewed) == ((), "original", True)


def test_a_save_lists_only_the_edits_since_the_previous_save(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.review()
    first = session.save(tmp_path / "one.skops", "First release")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    with pytest.raises(UnreviewedEditError):
        session.save(tmp_path / "two.skops", "Second release")
    session.review()
    second = session.save(tmp_path / "two.skops", "Second release")
    assert first.edit_ids == (session.history[0].id,)
    assert second.edit_ids == (session.history[2].id,)
    assert [row.kind for row in session.history_table()] == ["edit", "save", "edit", "save"]


def test_undo_redo_and_checkout_skip_the_save_entries_and_no_edit_removes_them(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    session.review()
    saved = session.save(tmp_path / "model.skops", "Release")
    session.apply(Shift("score", Bins(6, 8), 0.5), "Two")
    session.undo()
    assert session.revision == saved.revision
    session.undo()
    assert session.revision == "original"
    session.redo()
    session.checkout("original")
    session.apply(Shift("score", Bins(10, 12), 0.5), "Three")
    # the apply dropped the edits "One" and "Two" with the redo stack, and the save keeps the ids of what it saved
    assert [entry.message for entry in session.history] == ["Release", "Three"]
    assert saved in session.history
    assert saved.edit_ids


def test_a_save_never_overwrites_and_a_failed_save_adds_no_entry(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    (tmp_path / "model.skops").write_bytes(b"first")
    with pytest.raises(FileExistsError):
        session.save(tmp_path / "model.skops", "Release")
    assert (tmp_path / "model.skops").read_bytes() == b"first"
    assert session.history == ()


def test_a_save_that_cannot_write_the_history_removes_the_model_file_and_the_entry(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    (tmp_path / "model.skops.history.json").write_bytes(b"taken")
    with pytest.raises(FileExistsError):
        session.save(tmp_path / "model.skops", "Release")
    assert not (tmp_path / "model.skops").exists()
    assert (tmp_path / "model.skops.history.json").read_bytes() == b"taken"
    assert session.history == ()
    session.save(tmp_path / "other.skops", "Release")
    assert session.history[0].sequence == 1  # the failed save used the sequence number 0


def test_a_save_with_a_stale_expected_revision_raises(edit_ebm: ExplainableBoostingClassifier, tmp_path: Path) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    with pytest.raises(StaleRevisionError):
        session.save(tmp_path / "model.skops", "Release", allow_unreviewed=True, expected_revision="original")
    assert list(tmp_path.iterdir()) == []


def test_replay_builds_the_same_session_from_the_original_model_and_the_json(
    edit_ebm: ExplainableBoostingClassifier,
    edit_data: PandasFrame,
    tmp_path: Path,
) -> None:
    session = _session(edit_ebm)
    session.apply(Monotonize("score", "increasing", Bins(5, 25)), "The score must not fall")
    session.apply(SetScores("age", MissingBin(), 0.0), "A missing age scores 0")
    session.review()
    session.save(tmp_path / "model.skops", "Release", allow_unreviewed=True)
    session.apply(Recenter("score"), "Keep the intercept")
    session.undo()
    replayed = EditSession.replay(edit_ebm, session.to_json(), clock=fixed_clock(), new_id=_ids())
    assert replayed.history == session.history
    assert replayed.revision == session.revision
    for feature in ("score", "age"):
        assert _scores(replayed.model, feature) == pytest.approx(_scores(session.model, feature))
    assert np.allclose(replayed.model.predict(edit_data), session.model.predict(edit_data))
    replayed.redo()
    session.redo()
    assert np.allclose(replayed.model.intercept_, session.model.intercept_)
    replayed.apply(Shift("score", Bins(0, 0), 0.1), "The next edit continues the sequence")
    assert replayed.history[-1].sequence == max(entry.sequence for entry in session.history) + 1


def test_replay_of_a_history_with_no_entries_gives_the_original_model(edit_ebm: ExplainableBoostingClassifier) -> None:
    replayed = EditSession.replay(edit_ebm, _session(edit_ebm).to_json())
    assert replayed.revision == "original"
    assert replayed.history == ()
    assert replayed.history_table() == []


def test_replay_refuses_a_model_with_other_bins_or_scores(edit_ebm: ExplainableBoostingClassifier) -> None:
    history = _session(edit_ebm).to_json()
    moved = copy.deepcopy(edit_ebm)
    move_cuts(moved, 1, 0.01)
    with pytest.raises(HistoryError, match="not the model that the history started from"):
        EditSession.replay(moved, history)
    scored = copy.deepcopy(edit_ebm)
    scored.term_scores_[1] *= 1.01
    with pytest.raises(HistoryError, match="not the model that the history started from"):
        EditSession.replay(scored, history)


def test_replay_refuses_a_head_that_is_not_a_revision(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    document = orjson.loads(session.to_json())
    document["head"] = str(uuid.UUID(int=5))
    with pytest.raises(HistoryError, match="is not a revision of its edits"):
        EditSession.replay(edit_ebm, orjson.dumps(document))


def test_replay_refuses_text_that_is_not_a_history(edit_ebm: ExplainableBoostingClassifier) -> None:
    with pytest.raises(HistoryError):
        EditSession.replay(edit_ebm, b"not json")


def test_the_json_of_a_session_names_the_head_and_the_entries(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(2, 4), 0.5), "One")
    document = orjson.loads(session.to_json())
    assert document["head"] == session.revision
    assert [entry["message"] for entry in document["entries"]] == ["One"]
    assert document["entries"][0]["edit"]["type"] == "Shift"


# ---- properties ------------------------------------------------------------------------------------------------------


@settings(max_examples=15, deadline=None)
@given(
    first=st.integers(0, 20),
    width=st.integers(1, 10),
    delta=st.floats(-2, 2, allow_nan=False),
    direction=st.sampled_from(DIRECTIONS),
)
def test_undo_after_apply_gives_back_the_same_model_and_redo_gives_the_edited_model(
    edit_ebm: ExplainableBoostingClassifier,
    first: int,
    width: int,
    delta: float,
    direction: Direction,
) -> None:
    session = _session(edit_ebm)
    session.apply(Shift("score", Bins(first, first + width), delta), "Shift")
    session.apply(Monotonize("score", direction, Bins(first, first + width)), "Monotone")
    edited = session.model
    session.undo()
    session.undo()
    for term, scores in enumerate(edit_ebm.term_scores_):
        assert np.array_equal(session.model.term_scores_[term], scores)
        assert np.array_equal(session.model.bagged_scores_[term], edit_ebm.bagged_scores_[term])
    session.redo()
    session.redo()
    for term, scores in enumerate(edited.term_scores_):
        assert np.array_equal(session.model.term_scores_[term], scores)


@settings(max_examples=10, deadline=None)
@given(deltas=st.lists(st.floats(-2, 2, allow_nan=False), min_size=1, max_size=4))
def test_replaying_the_history_on_the_original_gives_the_same_model(
    edit_ebm: ExplainableBoostingClassifier,
    deltas: list[float],
) -> None:
    session = _session(edit_ebm)
    for position, delta in enumerate(deltas):
        session.apply(Shift("score", Bins(position, position + 3), delta), f"Edit {position}")
    replayed = EditSession.replay(edit_ebm, session.to_json())
    for term, scores in enumerate(session.model.term_scores_):
        assert np.array_equal(replayed.model.term_scores_[term], scores)


# ---- concurrency -----------------------------------------------------------------------------------------------------


class ProbeLock(AbstractContextManager[bool]):
    """A lock that records whether it is held, so that a test sees what runs inside it."""

    def __init__(self) -> None:
        """Make an open lock."""
        self._lock = threading.Lock()
        self.held: bool = False

    def __enter__(self) -> bool:
        """Take the lock, and record that it is held.

        Returns
        -------
        bool
            ``True``, as the lock of ``threading`` returns.

        """
        self._lock.acquire()
        self.held = True
        return True

    def __exit__(self, *_exception: object) -> None:
        """Record that the lock is free, and release it."""
        self.held = False
        self._lock.release()


@attrs.frozen
class _ProbedEdit(Edit):
    """An edit that checks, when it runs, that the session does not hold its lock."""

    probe: ProbeLock
    feature: str = "score"

    def run(self, ebm: Model) -> EditOutcome:
        assert not self.probe.held, "the session ran an edit while it held the lock"
        return Shift(self.feature, WholeGraph(), 0.1).run(ebm)


def test_the_session_never_holds_the_lock_while_it_runs_an_edit_and_holds_it_when_it_assigns_the_id_and_the_time(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    probe = ProbeLock()
    seen: dict[str, list[bool]] = {"id": [], "time": []}
    ids = _ids()
    clock = fixed_clock()

    def new_id() -> uuid.UUID:
        seen["id"].append(probe.held)
        return ids()

    def now():  # ruff: ignore[missing-return-type-private-function] -- the test double returns the time of the fixed clock
        seen["time"].append(probe.held)
        return clock()

    session = EditSession(edit_ebm, clock=now, new_id=new_id)
    session._lock = probe
    session.apply(_ProbedEdit(probe), "Probe")
    session.preview(_ProbedEdit(probe))
    assert seen == {"id": [True], "time": [True]}


def test_n_threads_that_apply_behind_a_barrier_leave_n_edits_and_the_head_equals_the_replay(
    edit_ebm: ExplainableBoostingClassifier,
) -> None:
    thread_count = 8
    session = EditSession(edit_ebm)
    barrier = threading.Barrier(thread_count)

    def apply(position: int) -> None:
        barrier.wait()
        session.apply(Shift("score", Bins(position, position + 2), 0.25 * (position + 1)), f"Edit {position}")

    with ThreadPoolExecutor(max_workers=thread_count) as pool:
        list(pool.map(apply, range(thread_count)))
    assert len(session.history) == thread_count
    assert sorted(entry.sequence for entry in session.history) == list(range(thread_count))
    assert len({entry.id for entry in session.history}) == thread_count
    sequential = copy.deepcopy(edit_ebm)
    for entry in sorted(session.history, key=lambda item: item.sequence):
        assert isinstance(entry, EditEntry)
        entry.edit.run(sequential)
    assert np.allclose(session.model.term_scores_[1], sequential.term_scores_[1])
    assert session.revision == str(max(session.history, key=lambda item: item.sequence).id)


def test_a_reader_during_the_writes_always_gets_a_consistent_revision(edit_ebm: ExplainableBoostingClassifier) -> None:
    session = EditSession(edit_ebm)
    done = threading.Event()
    problems: list[str] = []

    def read() -> None:
        while not done.is_set():
            model = session.model
            for term, scores in enumerate(model.term_scores_):
                if not np.allclose(model.bagged_scores_[term].mean(axis=0), scores):
                    problems.append("the bags do not match the scores")

    reader = threading.Thread(target=read)
    reader.start()
    for position in range(6):
        session.apply(Shift("score", Bins(position, position + 2), 0.25), f"Edit {position}")
    done.set()
    reader.join()
    assert problems == []


def test_two_threads_that_save_at_once_never_take_the_same_file(
    edit_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    session = EditSession(edit_ebm)
    barrier = threading.Barrier(2)

    def save(_: int) -> str:
        barrier.wait()
        try:
            session.save(tmp_path / "model.skops", "Release")
        except FileExistsError:
            return "taken"
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(save, range(2)))
    assert outcomes == ["saved", "taken"]
    assert sum(isinstance(entry, SaveEntry) for entry in session.history) == 1
