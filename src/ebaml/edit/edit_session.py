"""An edit session: apply, undo, redo, review and save the edits of one model, with a history.

The session never changes a model in place. An edit copies the model of the current revision, changes the copy, and
adds it as a new revision. So a reader of a revision never sees a half-written edit, and a slow read needs no lock.
One lock serializes the writers, and it protects the history and the head only. The session never holds it for an edit,
a write of a file, or the confirmation of a person.

A call that changes the head can take ``expected_revision``. If the head differs, the call raises
``StaleRevisionError`` and changes nothing. The session never retries such a call.
"""

import hashlib
import os
import threading
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

import attrs
import skops.io

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EditError, HistoryError, StaleRevisionError, UnreviewedEditError
from ebaml.edit import model_write
from ebaml.edit.edit_report import EditReport, build_report
from ebaml.edit.edits import Edit
from ebaml.edit.history import (
    EditEntry,
    HistoryEntry,
    HistoryRow,
    SaveEntry,
    decode_history,
    encode_history,
    fingerprint,
    short_ids,
)
from ebaml.edit.revision_ids import new_revision_id

ORIGINAL = "original"
CURRENT = "current"
_HISTORY_SUFFIX = ".history.json"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@attrs.frozen
class RevisionInfo:
    """A revision and the state of the session at the moment of the read.

    ``revision`` is ``original`` or the id of an edit, never ``current``. ``head`` is the revision of the head, and
    ``edit_count`` is the number of edits that the head holds.
    """

    revision: str
    head: str
    edit_count: int


@attrs.frozen(eq=False)
class _State:
    """The whole state of a session. A writer replaces it in one step, so a reader sees one consistent state."""

    original: ExplainableBoostingModel
    edits: tuple[EditEntry, ...] = ()
    models: tuple[ExplainableBoostingModel, ...] = ()
    saves: tuple[SaveEntry, ...] = ()
    position: int = 0
    next_sequence: int = 0

    @property
    def head_model(self) -> ExplainableBoostingModel:
        return self.original if self.position == 0 else self.models[self.position - 1]

    @property
    def head_revision(self) -> str:
        return ORIGINAL if self.position == 0 else str(self.edits[self.position - 1].id)

    def last_save_sequence(self) -> int:
        return max((save.sequence for save in self.saves), default=-1)

    def pending_edits(self) -> list[EditEntry]:
        """List the applied edits that no save has recorded yet, in the order of their sequence numbers.

        Returns
        -------
        list[EditEntry]
            The edits that the next save records.

        """
        return [entry for entry in self.edits[: self.position] if entry.sequence > self.last_save_sequence()]

    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(sorted((*self.edits, *self.saves), key=lambda entry: entry.sequence))

    def position_of(self, revision: str | uuid.UUID) -> int:
        """Find the number of applied edits at a revision: 0 for the original model.

        Returns
        -------
        int
            The position.

        Raises
        ------
        EditError
            If the session has no such revision.

        """
        text = str(revision)
        if text == ORIGINAL:
            return 0
        if text == CURRENT:
            return self.position
        for index, entry in enumerate(self.edits):
            if str(entry.id) == text:
                return index + 1
        known = [ORIGINAL, CURRENT, *(str(entry.id) for entry in self.edits)]
        msg = f"The session has no revision {text!r}. The revisions are {known}."
        raise EditError(msg)

    def check_expected(self, expected_revision: str | uuid.UUID | None) -> None:
        if expected_revision is None or str(expected_revision) == CURRENT:
            return
        if str(expected_revision) != self.head_revision:
            msg = (
                f"The head of the session is {self.head_revision!r}, and the call expected {str(expected_revision)!r}."
                " Read the session again, and decide if the call still fits."
            )
            raise StaleRevisionError(msg)


class EditSession:
    """The edits of one model, with undo, redo, a history and a save.

    The session works on a copy of the model, so the model that you pass does not change. It supports binary
    classifiers and regressors. A model that ``interpret`` monotonized has no standard deviations, and the session
    fills them with 0 in the copy.

    Several threads can share one session.

    Examples
    --------
    >>> import numpy as np
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import EditSession, Monotonize
    >>> random_generator = np.random.default_rng(0)
    >>> ages = random_generator.integers(1, 9, size=(300, 1)) * 10.0
    >>> target = (ages[:, 0] + random_generator.normal(0, 20, 300) > 40).astype(int)
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=2).fit(ages, target)
    >>> session = EditSession(ebm)
    >>> report = session.apply(Monotonize("feature_0000", "increasing"), "The risk must not fall with age")
    >>> session.history_table()[0].reason
    'The risk must not fall with age'
    >>> session.undo()
    'original'

    """

    def __init__(
        self,
        ebm: ExplainableBoostingModel,
        *,
        clock: Callable[[], datetime] = _utc_now,
        new_id: Callable[[], uuid.UUID] = new_revision_id,
    ) -> None:
        """Start a session on a copy of a model.

        Parameters
        ----------
        ebm : ExplainableBoostingModel
            The fitted EBM.
        clock : Callable[[], datetime]
            Gives the time of an edit. Pass a fixed function in a test, so that the history is deterministic.
        new_id : Callable[[], uuid.UUID]
            Gives the id of an entry. The default makes UUIDs of version 7.

        """
        model_write.check_editable(ebm)
        original = ebm.copy()
        model_write.repair_uncertainty(original)
        self._fingerprint = fingerprint(original)
        self._clock = clock
        self._new_id = new_id
        self._lock: AbstractContextManager[bool] = threading.Lock()
        self._state = _State(original)

    @property
    def revision(self) -> str:
        """The head: ``original``, or the id of the last applied edit."""
        return self._state.head_revision

    def revision_info(self, revision: str | uuid.UUID = CURRENT) -> "RevisionInfo":
        """Name a revision by its fixed name, and say what the head holds.

        Parameters
        ----------
        revision : str | uuid.UUID
            The revision to name. ``current`` names the head.

        Returns
        -------
        RevisionInfo
            The fixed name of the revision, the head, and the number of edits that the head holds.

        """
        state = self._state
        position = state.position_of(revision)
        name = ORIGINAL if position == 0 else str(state.edits[position - 1].id)
        return RevisionInfo(revision=name, head=state.head_revision, edit_count=state.position)

    @property
    def model(self) -> ExplainableBoostingModel:
        """A copy of the model at the head. Changing it does not change the session."""
        return self._state.head_model.copy()

    def model_at(self, revision: str | uuid.UUID = CURRENT) -> ExplainableBoostingModel:
        """Read a copy of the model at any revision, with no checkout.

        Parameters
        ----------
        revision : str | uuid.UUID
            ``original``, ``current`` or the id of an edit, also an edit that an undo moved to the redo stack.

        Returns
        -------
        ExplainableBoostingModel
            A copy that you can change without effect on the session.

        """
        state = self._state
        position = state.position_of(revision)
        return (state.original if position == 0 else state.models[position - 1]).copy()

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        """The entries in the order of their sequence numbers: the edits, and the saves."""
        return self._state.entries()

    def history_table(self, *, pending_only: bool = False) -> list[HistoryRow]:
        """List the entries as rows: id, time, kind, feature, edit type, description, reason and review.

        Parameters
        ----------
        pending_only : bool
            If ``True``, list only the applied edits that no save has recorded yet. These are the edits that the next
            ``save`` records. The short ids stay the same as in the whole table.

        Returns
        -------
        list[HistoryRow]
            One row for each entry. ``short_id`` is a short unique prefix of the random tail of the id.

        """
        state = self._state
        entries = state.entries()
        shorts = short_ids([entry.id for entry in entries])
        chosen = state.pending_edits() if pending_only else entries
        return [entry.row(shorts[entry.id]) for entry in chosen]

    def to_json(self) -> bytes:
        """Write the history as JSON that ``replay`` reads.

        Returns
        -------
        bytes
            The format version, the fingerprint of the original model, the head and the entries.

        """
        state = self._state
        return encode_history(self._fingerprint, state.head_revision, state.entries())

    def preview(self, edit: Edit, *, revision: str | uuid.UUID = CURRENT) -> EditReport:
        """Compute the report of an edit and change nothing.

        Parameters
        ----------
        edit : Edit
            The edit.
        revision : str | uuid.UUID
            The revision to start from.

        Returns
        -------
        EditReport
            The report that ``apply`` gives for the same edit on the same revision.

        """
        before = self.model_at(revision)
        after = before.copy()
        outcome = edit.run(after)
        return build_report(before, after, type(edit).__name__, outcome)

    def apply(self, edit: Edit, message: str, *, expected_revision: str | uuid.UUID | None = None) -> EditReport:
        """Apply an edit as a new revision.

        The session drops the redo stack, as GAM Changer does. A ``save`` entry keeps the ids of the edits that it
        saved, also after an apply drops those edits, so the history still shows which edits became which file.

        Parameters
        ----------
        edit : Edit
            The edit.
        message : str
            The reason for the edit. The history keeps it as documentation of the model.
        expected_revision : str | uuid.UUID, optional
            The head that the caller read. If the head differs, the call raises ``StaleRevisionError``.

        Returns
        -------
        EditReport
            What the edit changed.

        """
        _require_message(message)
        while True:
            state = self._state
            state.check_expected(expected_revision)
            before = state.head_model
            after = before.copy()
            outcome = edit.run(after)
            report = build_report(before, after, type(edit).__name__, outcome)
            with self._lock:
                if self._state is state:
                    entry = EditEntry(
                        id=self._new_id(),
                        sequence=state.next_sequence,
                        time=self._clock(),
                        message=message,
                        edit=edit,
                        description=outcome.description,
                        feature=outcome.feature,
                    )
                    self._state = attrs.evolve(
                        state,
                        edits=(*state.edits[: state.position], entry),
                        models=(*state.models[: state.position], after),
                        position=state.position + 1,
                        next_sequence=state.next_sequence + 1,
                    )
                    return report
            # another writer changed the state while this call computed, so compute again on the new head

    def _move(self, target: Callable[[_State], int], expected_revision: str | uuid.UUID | None) -> str:
        with self._lock:
            state = self._state
            state.check_expected(expected_revision)
            self._state = attrs.evolve(state, position=target(state))
            return self._state.head_revision

    def undo(self, *, expected_revision: str | uuid.UUID | None = None) -> str:
        """Move the head to the previous revision. The undone edit stays in the redo stack.

        An undo with no edit to undo raises an ``EditError``, and a stale ``expected_revision`` raises a
        ``StaleRevisionError``.

        Parameters
        ----------
        expected_revision : str | uuid.UUID, optional
            The head that the caller read.

        Returns
        -------
        str
            The head after the undo.

        """
        return self._move(lambda state: _relative(state, -1), expected_revision)

    def redo(self, *, expected_revision: str | uuid.UUID | None = None) -> str:
        """Move the head to the next revision of the redo stack.

        A redo with no undone edit raises an ``EditError``, and a stale ``expected_revision`` raises a
        ``StaleRevisionError``.

        Parameters
        ----------
        expected_revision : str | uuid.UUID, optional
            The revision that the caller expects at the head.

        Returns
        -------
        str
            The head after the redo.

        """
        return self._move(lambda state: _relative(state, 1), expected_revision)

    def checkout(self, revision: str | uuid.UUID, *, expected_revision: str | uuid.UUID | None = None) -> str:
        """Move the head to a revision. The later edits stay in the redo stack.

        Parameters
        ----------
        revision : str | uuid.UUID
            ``original`` or the id of an edit.
        expected_revision : str | uuid.UUID, optional
            The head that the caller read.

        Returns
        -------
        str
            The new head.

        """
        return self._move(lambda state: state.position_of(revision), expected_revision)

    def review(self, revision: str | uuid.UUID = CURRENT) -> None:
        """Mark the edit of a revision as reviewed by a person.

        Parameters
        ----------
        revision : str | uuid.UUID
            The id of an edit, or ``current`` for the edit at the head.

        Raises
        ------
        EditError
            If the revision is the original model, which no edit made.

        """
        with self._lock:
            state = self._state
            position = state.position_of(revision)
            if position == 0:
                msg = "The original model has no edit to review."
                raise EditError(msg)
            reviewed = attrs.evolve(state.edits[position - 1], reviewed=True)
            edits = (*state.edits[: position - 1], reviewed, *state.edits[position:])
            self._state = attrs.evolve(state, edits=edits)

    def save(
        self,
        path: str | os.PathLike[str],
        message: str,
        *,
        author: str | None = None,
        allow_unreviewed: bool = False,
        expected_revision: str | uuid.UUID | None = None,
    ) -> SaveEntry:
        """Save the model at the head, and add a ``save`` entry to the history.

        The save writes the model with ``skops.io`` and the history next to it, in a file with the suffix
        ``.history.json``. Neither file may exist: a save never overwrites. A save that fails adds no entry.

        Parameters
        ----------
        path : str | os.PathLike[str]
            The file of the model.
        message : str
            The description of the change that the saved model holds, for example the business requirement that the
            edits serve.
        author : str, optional
            The name of the person who saves.
        allow_unreviewed : bool
            If ``False``, the save refuses edits since the previous save that nobody reviewed.
        expected_revision : str | uuid.UUID, optional
            The head that the caller read.

        Returns
        -------
        SaveEntry
            The entry: its id, the revision, the edits since the previous save, and the hash of the saved model.

        Raises
        ------
        UnreviewedEditError
            If an edit since the previous save is not reviewed, and ``allow_unreviewed`` is ``False``.
        OSError
            If a file exists or cannot be written. The save then adds no entry.

        """
        _require_message(message)
        state = self._state
        state.check_expected(expected_revision)
        pending = state.pending_edits()
        unreviewed = [entry for entry in pending if not entry.reviewed]
        if unreviewed and not allow_unreviewed:
            ids = [str(entry.id) for entry in unreviewed]
            msg = f"The edits {ids} are not reviewed. Call review for each one, or pass allow_unreviewed=True."
            raise UnreviewedEditError(msg)
        target = Path(path)
        data = skops.io.dumps(state.head_model)
        with self._lock:
            latest = self._state
            entry = SaveEntry(
                id=self._new_id(),
                sequence=latest.next_sequence,
                time=self._clock(),
                message=message,
                author=author,
                edit_ids=tuple(pending_entry.id for pending_entry in pending),
                reviewed=not unreviewed,
                revision=state.head_revision,
                file=target.name,
                model_hash=hashlib.sha256(data).hexdigest(),
            )
            self._state = attrs.evolve(latest, saves=(*latest.saves, entry), next_sequence=latest.next_sequence + 1)
            snapshot = self._state
        created: list[Path] = []
        try:
            history_text = encode_history(self._fingerprint, snapshot.head_revision, snapshot.entries())
            for file, content in ((target, data), (target.with_name(target.name + _HISTORY_SUFFIX), history_text)):
                with file.open("xb") as handle:
                    created.append(file)
                    handle.write(content)
        except OSError:
            with self._lock:
                current = self._state
                self._state = attrs.evolve(current, saves=tuple(save for save in current.saves if save.id != entry.id))
            for file in created:
                file.unlink(missing_ok=True)
            raise
        return entry

    @classmethod
    def replay(
        cls,
        ebm: ExplainableBoostingModel,
        history: bytes | str,
        *,
        clock: Callable[[], datetime] = _utc_now,
        new_id: Callable[[], uuid.UUID] = new_revision_id,
    ) -> "EditSession":
        """Build a session from the JSON of ``to_json`` and the model that the history started from.

        The replay runs each edit again, and keeps the ids, the times, the reasons and the reviews of the history.
        A model with another fingerprint is refused, because bins that moved make a selection mean other rows.

        Parameters
        ----------
        ebm : ExplainableBoostingModel
            The original model.
        history : bytes | str
            The JSON text.
        clock : Callable[[], datetime]
            Gives the time of the next entry.
        new_id : Callable[[], uuid.UUID]
            Gives the id of the next entry.

        Returns
        -------
        EditSession
            The session, with the head of the history.

        Raises
        ------
        HistoryError
            If the history is not readable, or the model has another fingerprint.

        """
        decoded = decode_history(history)
        session = cls(ebm, clock=clock, new_id=new_id)
        if decoded.fingerprint != session._fingerprint:
            msg = "The model is not the model that the history started from: its features, bins or scores differ."
            raise HistoryError(msg)
        state = session._state
        edits = sorted(decoded.edits, key=lambda entry: entry.sequence)
        models: list[ExplainableBoostingModel] = []
        for entry in edits:
            model = (models[-1] if models else state.original).copy()
            entry.edit.run(model)
            models.append(model)
        head_ids = [ORIGINAL, *(str(entry.id) for entry in edits)]
        if decoded.head not in head_ids:
            msg = f"The head {decoded.head!r} of the history is not a revision of its edits."
            raise HistoryError(msg)
        sequences = [entry.sequence for entry in (*decoded.edits, *decoded.saves)]
        session._state = attrs.evolve(
            state,
            edits=tuple(edits),
            models=tuple(models),
            saves=tuple(sorted(decoded.saves, key=lambda entry: entry.sequence)),
            position=head_ids.index(decoded.head),
            next_sequence=max(sequences, default=-1) + 1,
        )
        return session


_NO_STEP = {-1: "undo, because the head is the original model", 1: "redo, because no edit was undone"}


def _relative(state: _State, step: int) -> int:
    """Find the position that is one edit before or after the head.

    Returns
    -------
    int
        The position.

    Raises
    ------
    EditError
        If the head is at the end of the history in that direction.

    """
    target = state.position + step
    if not 0 <= target <= len(state.edits):
        msg = f"There is no edit to {_NO_STEP[step]}."
        raise EditError(msg)
    return target


def _require_message(message: str) -> None:
    if not message.strip():
        msg = "A message is needed: write the reason for the change, so that a reviewer can read it later."
        raise EditError(msg)
