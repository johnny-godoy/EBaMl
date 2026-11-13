"""The history of an edit session: its entries, its table, and its JSON form.

Every entry is a commit with an id, a time, a position (its sequence number) and the reason that a person wrote. An
entry of kind ``edit`` holds one edit. An entry of kind ``save`` records which revision became which file. The sequence
number gives the order, and the id only names the entry.
"""

import abc
import hashlib
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import override

import attrs
import numpy as np
import orjson

from ebaml.core.custom_types import ExplainableBoostingModel, JSONObject, JSONValue
from ebaml.core.exceptions import HistoryError
from ebaml.edit.edit_records import edit_from_record, to_record
from ebaml.edit.edits import Edit

FORMAT_VERSION = 1
_MINIMUM_SHORT_ID = 4
# a UUID of version 7 has 12 hex digits of time, then the version digit, then the random tail
_TAIL_START = 13


@attrs.frozen
class HistoryRow:
    """One row of the table of the history."""

    id: str
    short_id: str
    sequence: int
    time: datetime
    kind: str
    feature: str | None
    edit_type: str | None
    description: str
    reason: str
    reviewed: bool


@attrs.frozen
class HistoryEntry(abc.ABC):
    """A commit of the history. The classes of its kinds hold the fields of the kind."""

    id: uuid.UUID
    sequence: int
    time: datetime
    message: str

    @abc.abstractmethod
    def row(self, short_id: str) -> HistoryRow:
        """Write the entry as a row of the table.

        Parameters
        ----------
        short_id : str
            The short unique prefix of the random tail of the id.

        Returns
        -------
        HistoryRow
            The row.

        """

    @abc.abstractmethod
    # nosemgrep: ebaml-return-attrs-not-container -- the record is a JSON fragment that the history merges and writes
    def to_record(self) -> JSONObject:
        """Write the entry as a record of JSON values."""


@attrs.frozen
class EditEntry(HistoryEntry):
    """The commit of one edit, with the reason that a person gave for it."""

    edit: Edit
    description: str
    feature: str | None
    reviewed: bool = False

    @override
    def row(self, short_id: str) -> HistoryRow:
        return HistoryRow(
            id=str(self.id),
            short_id=short_id,
            sequence=self.sequence,
            time=self.time,
            kind="edit",
            feature=self.feature,
            edit_type=type(self.edit).__name__,
            description=self.description,
            reason=self.message,
            reviewed=self.reviewed,
        )

    @override
    # nosemgrep: ebaml-return-attrs-not-container -- the record is a JSON fragment that the history merges and writes
    def to_record(self) -> JSONObject:
        return {
            "kind": "edit",
            "id": str(self.id),
            "sequence": self.sequence,
            "time": self.time.isoformat(),
            "message": self.message,
            "edit": to_record(self.edit),
            "description": self.description,
            "feature": self.feature,
            "reviewed": self.reviewed,
        }


@attrs.frozen
class SaveEntry(HistoryEntry):
    """The commit of a save: the revision, the file that holds it, and the edits since the previous save."""

    author: str | None
    edit_ids: tuple[uuid.UUID, ...]
    reviewed: bool
    revision: str
    file: str
    model_hash: str

    @override
    def row(self, short_id: str) -> HistoryRow:
        return HistoryRow(
            id=str(self.id),
            short_id=short_id,
            sequence=self.sequence,
            time=self.time,
            kind="save",
            feature=None,
            edit_type=None,
            description=f"Save the revision {self.revision} as {self.file}",
            reason=self.message,
            reviewed=self.reviewed,
        )

    @override
    # nosemgrep: ebaml-return-attrs-not-container -- the record is a JSON fragment that the history merges and writes
    def to_record(self) -> JSONObject:
        return {
            "kind": "save",
            "id": str(self.id),
            "sequence": self.sequence,
            "time": self.time.isoformat(),
            "message": self.message,
            "author": self.author,
            "edit_ids": [str(edit_id) for edit_id in self.edit_ids],
            "reviewed": self.reviewed,
            "revision": self.revision,
            "file": self.file,
            "model_hash": self.model_hash,
        }


# nosemgrep: ebaml-return-attrs-not-container -- the keys are the ids of the entries, so they change from call to call
def short_ids(ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
    """Find the shortest unique prefix of the random tail of each id, with at least four characters.

    Parameters
    ----------
    ids : Sequence[uuid.UUID]
        The ids of the entries.

    Returns
    -------
    dict[uuid.UUID, str]
        The short form of each id.

    """
    tails = {entry_id: entry_id.hex[_TAIL_START:] for entry_id in ids}
    shorts: dict[uuid.UUID, str] = {}
    length = _MINIMUM_SHORT_ID
    while len(shorts) < len(tails):
        prefixes = [tail[:length] for tail in tails.values()]
        for entry_id, tail in tails.items():
            if entry_id not in shorts and prefixes.count(tail[:length]) == 1:
                shorts[entry_id] = tail[:length]
        length += 1
    return shorts


def _hash_arrays(
    arrays: Sequence[np.ndarray],
) -> str:  # nosemgrep: ebaml-numpy-loop-needs-reason -- the loop feeds a hash, and the arrays have different shapes
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
    return digest.hexdigest()


@attrs.frozen
class Fingerprint:
    """What identifies the model that a history started from: its features, terms, bins and scores."""

    features: tuple[str, ...]
    terms: tuple[str, ...]
    bins: str
    scores: str

    # nosemgrep: ebaml-return-attrs-not-container -- the record is a JSON fragment that the history merges and writes
    def to_record(self) -> JSONObject:
        """Write the fingerprint as a record of JSON values.

        Returns
        -------
        JSONObject
            The features, the terms and the two hashes.

        """
        return {
            "features": list(self.features),
            "terms": list(self.terms),
            "bins": self.bins,
            "scores": self.scores,
        }


def fingerprint(ebm: ExplainableBoostingModel) -> Fingerprint:
    """Fingerprint a model.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The fitted EBM.

    Returns
    -------
    Fingerprint
        The names of the features and terms, a hash of the bins, and a hash of the scores.

    """
    bin_digest = hashlib.sha256()
    for feature_bins in ebm.bins_:
        first = feature_bins[0]
        if isinstance(first, Mapping):
            bin_digest.update(orjson.dumps(sorted(first.items())))
        else:
            bin_digest.update(_hash_arrays([np.asarray(first)]).encode())
    bin_digest.update(_hash_arrays([np.asarray(bounds, dtype=float) for bounds in ebm.feature_bounds_]).encode())
    return Fingerprint(
        tuple(ebm.feature_names_in_),
        tuple(ebm.term_names_),
        bin_digest.hexdigest(),
        _hash_arrays([np.asarray(scores) for scores in ebm.term_scores_]),
    )


@attrs.frozen
class DecodedHistory:
    """The content of a history that ``decode_history`` read."""

    fingerprint: Fingerprint
    head: str
    edits: tuple[EditEntry, ...]
    saves: tuple[SaveEntry, ...]


def encode_history(model_fingerprint: Fingerprint, head: str, entries: Sequence[HistoryEntry]) -> bytes:
    """Write a history as JSON.

    Parameters
    ----------
    model_fingerprint : Fingerprint
        The fingerprint of the original model.
    head : str
        The revision that the session holds: ``original`` or the id of an edit.
    entries : Sequence[HistoryEntry]
        The entries, in the order of their sequence numbers.

    Returns
    -------
    bytes
        The JSON text.

    """
    document: JSONObject = {
        "format_version": FORMAT_VERSION,
        "fingerprint": model_fingerprint.to_record(),
        "head": head,
        "entries": [entry.to_record() for entry in entries],
    }
    return orjson.dumps(document, option=orjson.OPT_INDENT_2)


def _object(value: JSONValue, what: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        msg = f"{what} must be an object, but it is {value!r}."
        raise HistoryError(msg)
    return value


def _field(record: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key not in record:
        msg = f"A history record lacks the field {key!r}: {dict(record)!r}"
        raise HistoryError(msg)
    return record[key]


def _text(record: Mapping[str, JSONValue], key: str) -> str:
    value = _field(record, key)
    if not isinstance(value, str):
        msg = f"The field {key!r} of a history record must be a text, but it is {value!r}."
        raise HistoryError(msg)
    return value


def _optional_text(record: Mapping[str, JSONValue], key: str) -> str | None:
    return None if _field(record, key) is None else _text(record, key)


def _flag(record: Mapping[str, JSONValue], key: str) -> bool:
    value = _field(record, key)
    if not isinstance(value, bool):
        msg = f"The field {key!r} of a history record must be true or false, but it is {value!r}."
        raise HistoryError(msg)
    return value


def _count(record: Mapping[str, JSONValue], key: str) -> int:
    value = _field(record, key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"The field {key!r} of a history record must be a whole number, but it is {value!r}."
        raise HistoryError(msg)
    return value


def _list(record: Mapping[str, JSONValue], key: str) -> Sequence[JSONValue]:
    value = _field(record, key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"The field {key!r} of a history record must be a list, but it is {value!r}."
        raise HistoryError(msg)
    return value


def _texts(record: Mapping[str, JSONValue], key: str) -> tuple[str, ...]:
    items = _list(record, key)
    if not all(isinstance(item, str) for item in items):
        msg = f"The field {key!r} of a history record must be a list of texts, but it is {items!r}."
        raise HistoryError(msg)
    return tuple(str(item) for item in items)


def _uuid(text: str) -> uuid.UUID:
    try:
        return uuid.UUID(text)
    except ValueError as error:
        msg = f"A history record holds the id {text!r}, which is not a UUID."
        raise HistoryError(msg) from error


def _time(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError as error:
        msg = f"A history record holds the time {text!r}, which is not an ISO 8601 time."
        raise HistoryError(msg) from error


def _edit_entry(record: Mapping[str, JSONValue]) -> EditEntry:
    return EditEntry(
        id=_uuid(_text(record, "id")),
        sequence=_count(record, "sequence"),
        time=_time(_text(record, "time")),
        message=_text(record, "message"),
        edit=edit_from_record(_field(record, "edit")),
        description=_text(record, "description"),
        feature=_optional_text(record, "feature"),
        reviewed=_flag(record, "reviewed"),
    )


def _save_entry(record: Mapping[str, JSONValue]) -> SaveEntry:
    return SaveEntry(
        id=_uuid(_text(record, "id")),
        sequence=_count(record, "sequence"),
        time=_time(_text(record, "time")),
        message=_text(record, "message"),
        author=_optional_text(record, "author"),
        edit_ids=tuple(_uuid(edit_id) for edit_id in _texts(record, "edit_ids")),
        reviewed=_flag(record, "reviewed"),
        revision=_text(record, "revision"),
        file=_text(record, "file"),
        model_hash=_text(record, "model_hash"),
    )


def decode_history(history: bytes | str) -> DecodedHistory:
    """Read the JSON that ``encode_history`` wrote.

    Parameters
    ----------
    history : bytes | str
        The JSON text.

    Returns
    -------
    DecodedHistory
        The fingerprint, the head, and the edit entries and the save entries in the order of their sequence numbers.

    Raises
    ------
    HistoryError
        If the text is not JSON, has another format version, or lacks a field.

    """
    try:
        document = _object(orjson.loads(history), "A history")
    except orjson.JSONDecodeError as error:
        msg = f"The text is not JSON ({error})."
        raise HistoryError(msg) from error
    version = _field(document, "format_version")
    if version != FORMAT_VERSION:
        msg = f"The history has the format version {version!r}, and this version reads {FORMAT_VERSION}."
        raise HistoryError(msg)
    saved = _object(_field(document, "fingerprint"), "The fingerprint")
    model_fingerprint = Fingerprint(
        _texts(saved, "features"),
        _texts(saved, "terms"),
        _text(saved, "bins"),
        _text(saved, "scores"),
    )
    records = [_object(record, "A history entry") for record in _list(document, "entries")]
    kinds = {kind: [record for record in records if _text(record, "kind") == kind] for kind in ("edit", "save")}
    if len(kinds["edit"]) + len(kinds["save"]) != len(records):
        known = [_text(record, "kind") for record in records if _text(record, "kind") not in kinds]
        msg = f"A history entry has the kind {known[0]!r}, and the kinds are 'edit' and 'save'."
        raise HistoryError(msg)
    return DecodedHistory(
        model_fingerprint,
        _text(document, "head"),
        tuple(_edit_entry(record) for record in kinds["edit"]),
        tuple(_save_entry(record) for record in kinds["save"]),
    )
