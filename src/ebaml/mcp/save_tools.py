"""The tool that saves the edited model, after a person confirms the edits.

GAM Changer refuses a save until a person has reviewed every edit (Wang et al., KDD 2022). An agent edits faster than a
person, so the save asks the person through MCP elicitation, with the list of the edits and their reasons. If the client
cannot ask, the tool saves the model and the history marks each edit as not reviewed by a person.

The tool writes only into the directory that the user gave at start, and it never overwrites a file. It holds no lock
while the person decides. The head is fixed before the question, and the save checks it again after the answer, so a
confirmation of an older revision never saves a newer one.
"""

import enum
from pathlib import Path

import attrs
from pydantic import BaseModel, Field

from ebaml.core.exceptions import EditError, StaleRevisionError
from ebaml.edit.edit_session import CURRENT, EditSession, RevisionInfo
from ebaml.edit.history import HistoryRow, SaveEntry
from ebaml.mcp.model_store import ModelStore


class Verdict(enum.Enum):
    """Who decided on the save. The value is the text that the result shows."""

    CONFIRMED = "person_confirmed"
    NO_EDITS = "no_edits_to_confirm"
    CANNOT_ASK = "client_cannot_ask"


class Confirmation(BaseModel):
    """The answer of the person. The client shows one yes-or-no field."""

    confirm: bool = Field(
        title="Save the model with these edits",
        description="Choose yes only after you read each edit and its reason.",
    )


@attrs.frozen
class SaveResult:
    """What a save wrote and who confirmed it.

    ``confirmation`` is ``person_confirmed`` if a person answered yes, ``no_edits_to_confirm`` if no edit came after the
    previous save, and ``client_cannot_ask`` if the client has no elicitation. In the last case the history marks each
    edit of the save as not reviewed by a person. The file names are inside the directory that the user gave at start.
    """

    session: RevisionInfo
    model_file: str
    history_file: str
    confirmation: str
    commit: HistoryRow
    edits_saved: list[HistoryRow]


def _check_head(head: str, expected_revision: str | None) -> None:
    if expected_revision is not None and expected_revision not in {CURRENT, head}:
        msg = (
            f"The head of the session is {head!r}, and the call expected {expected_revision!r}."
            " Read the session again, and decide if the call still fits."
        )
        raise StaleRevisionError(msg)


def _output_directory(store: ModelStore, message: str) -> Path:
    if store.output_directory is None:
        msg = "The server has no output directory. Start it with --output-dir to save."
        raise EditError(msg)
    if not message.strip():
        msg = "A message is needed: write the change that the saved model holds, so that a reviewer can read it later."
        raise EditError(msg)
    return store.output_directory


def current_head(store: ModelStore) -> str:
    """Name the head of the session. The save fixes it before it asks the person.

    Returns
    -------
    str
        ``original`` or the id of the last applied edit.

    """
    return store.editing().revision_info().head


def question(store: ModelStore, message: str, head: str, expected_revision: str | None) -> str | None:
    """Check that the save can go on, and write the question for the person.

    The checks come before the question, so the person never answers a call that must fail.

    Parameters
    ----------
    store: ModelStore
        The session. It needs an output directory.
    message: str
        The change that the saved model holds.
    head: str
        The head that the question describes.
    expected_revision: str, optional
        The head that the caller read.

    Returns
    -------
    str | None
        The message of the save and the edits with their reasons, or ``None`` if no edit came after the previous save.
        An empty message, a server with no output directory, and a stale head raise an ``EBaMlError`` with the fix.

    """
    _output_directory(store, message)
    session = store.editing()
    _check_head(head, expected_revision)
    _check_head(session.revision_info().head, head)
    pending = session.history_table(pending_only=True)
    if not pending:
        return None
    edits = "\n".join(
        f"{number}. {row.description}. Reason: {row.reason}" for number, row in enumerate(pending, start=1)
    )
    return (
        f"Save the edited model at revision {head}? The change that the saved model holds: {message}\n"
        f"Edits since the last save ({len(pending)}):\n{edits}\n"
        "The saved model changes real predictions. Confirm only after you read each edit and its reason."
    )


def _save_in_directory(session: EditSession, directory: Path, message: str, head: str, *, reviewed: bool) -> SaveEntry:
    """Save the model under the first name that no file uses, and never overwrite a file.

    Returns
    -------
    SaveEntry
        The entry of the save. ``file`` is the name of the model file.

    """
    number = len(list(directory.glob("*.skops"))) + 1
    while True:
        target = directory / f"edited-model-{number:03d}.skops"
        try:
            return session.save(target, message, allow_unreviewed=not reviewed, expected_revision=head)
        except FileExistsError:
            # a file already uses the name, or another save took it between the check and the write
            number += 1


def save_model(store: ModelStore, message: str, head: str, *, confirmed: bool | None) -> SaveResult:
    """Save the edited model and its history, after the answer of the person.

    The function checks that the head is still the one that the question described. If a person confirmed, it marks
    the edits of the question as reviewed. It runs file operations, so an async caller runs it in a thread.

    Parameters
    ----------
    store: ModelStore
        The session. It needs an output directory.
    message: str
        The change that the saved model holds.
    head: str
        The head that the question described.
    confirmed: bool | None
        The answer of the person. ``False`` saves nothing. ``None`` means that nobody was asked: either no edit needs a
        review, or the client cannot ask, and then the history marks each edit as not reviewed.

    Returns
    -------
    SaveResult
        The file names, who confirmed the save, and the commits. An empty message, a server with no output directory,
        and a model that changed while the person decided raise an ``EBaMlError`` and save nothing.

    Raises
    ------
    EditError
        If the person refused.

    """
    if confirmed is False:
        msg = "The person did not confirm the save, so nothing was saved. Ask the person before you try again."
        raise EditError(msg)
    directory = _output_directory(store, message)
    session = store.editing()
    # the model can change while the person decides, and then the person confirmed a model that no longer exists
    _check_head(session.revision_info().head, head)
    pending_ids = {row.id for row in session.history_table(pending_only=True)}
    if confirmed is None:
        verdict = Verdict.CANNOT_ASK if pending_ids else Verdict.NO_EDITS
    else:
        verdict = Verdict.CONFIRMED
        for edit_id in pending_ids:
            session.review(edit_id)
    entry = _save_in_directory(session, directory, message, head, reviewed=verdict is not Verdict.CANNOT_ASK)
    rows = session.history_table()
    commit = next(row for row in rows if row.id == str(entry.id))
    saved = [row for row in rows if row.id in pending_ids]
    return SaveResult(session.revision_info(), entry.file, entry.file + ".history.json", verdict.value, commit, saved)
