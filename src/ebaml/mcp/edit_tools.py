"""The tools that edit the model: list the bins, preview and apply an edit, move in the history and list the edits.

Each function calls the edit API of ``ebaml.edit`` and returns an attrs class. No function calls an LLM. The agent is
the LLM, so the tools stay exact and free. The server turns each class into JSON.
"""

import inspect
from collections.abc import Mapping
from typing import cast

import attrs

from ebaml.core.custom_types import ExplainableBoostingModel, JSONMapping, JSONValue
from ebaml.core.exceptions import EditError, InvalidInstanceError
from ebaml.edit.data_effect import RowChanges, row_changes
from ebaml.edit.edit_records import EDIT_TYPES, SELECTION_FIELD, SELECTION_TYPES, edit_from_record
from ebaml.edit.edit_report import EditReport
from ebaml.edit.edit_session import CURRENT, RevisionInfo
from ebaml.edit.edits import Edit
from ebaml.edit.history import HistoryRow
from ebaml.edit.model_view import interaction_names, term_view
from ebaml.edit.selections import Selection
from ebaml.mcp.model_store import ModelStore

_KIND = "kind"
_RECORD_TYPE = "type"
_NO_DATA = "The server has no data (--data), so the effect of the edit on rows is not checked."


@attrs.frozen
class BinInfo:
    """One bin of a graph, as the edit tools address it.

    ``number`` is the number that ``Bins`` uses, counted from 0 among the real bins. It is ``None`` for the missing bin
    and the unseen bin. ``left`` and ``right`` are the edges of a continuous bin, and ``None`` for any other bin.
    """

    number: int | None
    label: str
    left: float | None
    right: float | None
    score: float
    weight: float
    weight_share: float


@attrs.frozen
class BinList:
    """The bins of one feature at one revision, with the training weight and the score of each bin."""

    session: RevisionInfo
    feature: str
    feature_type: str
    bins: list[BinInfo]
    missing: BinInfo
    unseen: BinInfo
    pair_terms_using_feature: list[str]


@attrs.frozen
class EditResult:
    """The report of an edit, and what the edit does to the rows of ``--data``.

    ``applied`` is ``False`` for a preview. ``effect_on_data`` is ``None`` when the server has no data or the data does
    not fit the model, and ``effect_on_data_note`` says which.
    """

    session: RevisionInfo
    applied: bool
    report: EditReport
    report_text: str
    effect_on_data: RowChanges | None
    effect_on_data_note: str | None

    @classmethod
    def of(
        cls,
        session: RevisionInfo,
        *,
        report: EditReport,
        applied: bool,
        effect: "_Effect",
    ) -> "EditResult":
        """Build the result of a preview or of an apply.

        Returns
        -------
        EditResult
            The result, with the text of the report.

        """
        return cls(session, applied, report, report.to_text(), effect.counts, effect.note)


@attrs.frozen
class HeadChange:
    """The head of the session after an undo, a redo or a checkout."""

    action: str
    session: RevisionInfo


@attrs.frozen
class EditList:
    """The history of the session as a table of commits.

    ``undone`` holds the ids of the edits that an undo moved to the redo stack. A redo or a checkout brings them back,
    and a new edit drops them. ``replay_json`` is the JSON that ``--edits`` reads, and it is ``None`` unless the caller
    asks for it.
    """

    session: RevisionInfo
    commits: list[HistoryRow]
    undone: list[str]
    replay_json: str | None


def list_bins(store: ModelStore, feature: str, revision: str = CURRENT) -> BinList:
    """List the bins of a feature with their training weight and their score.

    Parameters
    ----------
    store: ModelStore
        The session.
    feature: str
        The name of the feature, as ``list_features`` gives it.
    revision: str, default="current"
        The revision to read. See ``model_summary``.

    Returns
    -------
    BinList
        The real bins in the order of the graph, then the missing bin and the unseen bin.

    """
    store.editing()
    read = store.read(revision)
    view = term_view(read.model, feature)
    scores, weights = view.scores, view.weights
    total = float(weights.sum())
    edges = view.bin_edges() if view.feature_type == "continuous" else []

    def info(position: int) -> BinInfo:
        real = view.is_real([position])
        left, right = edges[position - 1] if edges and real else (None, None)
        return BinInfo(
            number=position - 1 if real else None,
            label=view.label(position),
            left=left,
            right=right,
            score=float(scores[position]),
            weight=float(weights[position]),
            weight_share=float(weights[position]) / total if total > 0 else 0.0,
        )

    return BinList(
        session=read.info,
        feature=feature,
        feature_type=view.feature_type,
        bins=[info(position) for position in range(1, view.real_bin_count + 1)],
        missing=info(0),
        unseen=info(view.unseen_position),
        pair_terms_using_feature=list(interaction_names(read.model, feature)),
    )


def _hint_text(hint: object) -> str:  # nosemgrep: ebaml-no-object-type -- the type hint of any field of an edit class
    if isinstance(hint, type):
        return "selection" if issubclass(hint, Selection) else hint.__name__
    return str(hint).replace("typing.", "")


def _signature(item: type) -> str:
    """Write a class as ``Name(field: type, field: type = default)`` with the first sentence of its docstring.

    Returns
    -------
    str
        One line for the description of a tool.

    """
    fields = ", ".join(
        f"{field.name}: {_hint_text(field.type)}" + (" = whole graph" if field.default is not attrs.NOTHING else "")
        for field in attrs.fields(item)
    )
    first_paragraph = " ".join((inspect.getdoc(item) or "").split("\n\n")[0].split())
    return f"- {_KIND} {item.__name__}({fields}): {first_paragraph.split('. ')[0].rstrip('.')}."


def edit_kinds_help() -> str:
    """List every edit and every selection with its fields, for the description of a tool.

    A new class in ``EDIT_TYPES`` or ``SELECTION_TYPES`` appears here with no other change.

    Returns
    -------
    str
        The text.

    """
    edit_lines = "\n".join(_signature(item) for item in EDIT_TYPES.values())
    selection_lines = "\n".join(_signature(item) for item in SELECTION_TYPES.values())
    return (
        f"The edit is a JSON object. Its field {_KIND!r} names the edit, and the other fields are the fields of that"
        f" edit. The field {SELECTION_FIELD!r} is a JSON object with a {_KIND!r} too.\n\nThe edits:\n{edit_lines}"
        f'\n\nThe selections:\n{selection_lines}\n\nExample: {{"{_KIND}": "Shift", "feature": "age",'
        f' "{SELECTION_FIELD}": {{"{_KIND}": "Bins", "first": 3, "last": 5}}, "delta": -0.2}}'
    )


def _record(value: JSONValue) -> JSONValue:
    """Rename the key ``kind`` to ``type`` in each record, and turn a whole number into an int.

    The MCP SDK reads each JSON number as a float, so ``Bins(first=0)`` arrives as ``first=0.0``. A field that holds a
    bin number needs an int, and a field that holds a score accepts an int.

    Returns
    -------
    JSONValue
        The record that ``edit_from_record`` reads. A value that is not a record stays as it is.

    """
    # nosemgrep: ebaml-polymorphism-not-type-branches -- a parsed JSON record at the edge of the tool
    if isinstance(value, Mapping):
        return {(_RECORD_TYPE if key == _KIND else key): _record(item) for key, item in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _check_kind(record: JSONMapping, registry: Mapping[str, type], what: str) -> None:
    kind = record.get(_KIND)
    if kind not in registry:
        msg = f"The {what} needs a field {_KIND!r} with one of {sorted(registry)}, but it has {kind!r}."
        raise EditError(msg)


def parse_edit(arguments: JSONMapping) -> Edit:
    """Build an edit from the JSON object that a tool takes.

    Parameters
    ----------
    arguments: JSONMapping
        The object. The field ``kind`` names the edit, and the field ``selection`` is an object with a ``kind`` too.

    Returns
    -------
    Edit
        The edit. An object with no known ``kind``, or with fields that do not fit the edit, raises an ``EditError``
        that lists what is known.

    """
    _check_kind(arguments, EDIT_TYPES, "edit")
    selection = arguments.get(SELECTION_FIELD)
    if isinstance(selection, Mapping):
        _check_kind(selection, SELECTION_TYPES, "selection")
    return edit_from_record(_record(arguments))


@attrs.frozen
class _Effect:
    """The counts of the rows that an edit changes, or the reason why the tool did not count them."""

    counts: RowChanges | None
    note: str | None


def _effect(store: ModelStore, before: ExplainableBoostingModel, after: ExplainableBoostingModel) -> _Effect:
    """Count the rows of ``--data`` that an edit changes.

    Returns
    -------
    _Effect
        The counts and no note, or no counts and the reason for it.

    """
    if store.data is None:
        return _Effect(None, _NO_DATA)
    try:
        return _Effect(row_changes(before, after, store.data), None)
    except InvalidInstanceError as error:
        return _Effect(None, f"The effect of the edit on rows is not checked. {error}")


def preview_edit(store: ModelStore, edit: Edit, revision: str = CURRENT) -> EditResult:
    """Compute the report of an edit and its effect on the data. The model of the session stays as it is.

    Parameters
    ----------
    store: ModelStore
        The session.
    edit: Edit
        The edit.
    revision: str, default="current"
        The revision to start from.

    Returns
    -------
    EditResult
        The report, the effect on ``--data`` if the server has data, and the state of the session.

    """
    session = store.editing()
    report = session.preview(edit, revision=revision)
    before = session.model_at(report.base_revision)
    after = before.copy()
    edit.run(after)
    effect = _effect(store, before, after)
    return EditResult.of(session.revision_info(), report=report, applied=False, effect=effect)


def apply_edit(
    store: ModelStore,
    edit: Edit,
    reason: str,
    expected_revision: str | None = None,
) -> EditResult:
    """Apply an edit to the session copy of the model, with the reason for it.

    Parameters
    ----------
    store: ModelStore
        The session.
    edit: Edit
        The edit.
    reason: str
        The business requirement that the edit serves. The history keeps it.
    expected_revision: str, optional
        The head that the caller read. If the head differs, the call fails and changes nothing.

    Returns
    -------
    EditResult
        The report, the effect on ``--data`` if the server has data, and the state of the session.

    """
    session = store.editing()
    report = session.apply(edit, reason, expected_revision=expected_revision)
    applied = cast(str, report.revision)  # apply always sets the revision that it made
    effect = _effect(store, session.model_at(report.base_revision), session.model_at(applied))
    return EditResult.of(session.revision_info(applied), report=report, applied=True, effect=effect)


def undo(store: ModelStore, expected_revision: str | None = None) -> HeadChange:
    """Undo the last applied edit. The edit stays in the redo stack.

    Parameters
    ----------
    store: ModelStore
        The session.
    expected_revision: str, optional
        The head that the caller read. A different head fails the call with no change.

    Returns
    -------
    HeadChange
        The session after the undo.

    """
    session = store.editing()
    return HeadChange("undo", session.revision_info(session.undo(expected_revision=expected_revision)))


def redo(store: ModelStore, expected_revision: str | None = None) -> HeadChange:
    """Apply the edit that the last undo moved to the redo stack.

    Parameters
    ----------
    store: ModelStore
        The session.
    expected_revision: str, optional
        The head that the caller read. A different head makes the call fail, and the model stays as it is.

    Returns
    -------
    HeadChange
        The session after the redo.

    """
    session = store.editing()
    return HeadChange("redo", session.revision_info(session.redo(expected_revision=expected_revision)))


def checkout(store: ModelStore, revision: str, expected_revision: str | None = None) -> HeadChange:
    """Set the model to its state after one commit of the history. The later edits stay in the redo stack.

    Parameters
    ----------
    store: ModelStore
        The session.
    revision: str
        ``original`` or the id of an edit.
    expected_revision: str, optional
        The head that the caller read. If the head is another one, the call raises a ``StaleRevisionError``.

    Returns
    -------
    HeadChange
        The session after the checkout.

    """
    session = store.editing()
    return HeadChange(
        "checkout",
        session.revision_info(session.checkout(revision, expected_revision=expected_revision)),
    )


def list_edits(store: ModelStore, *, include_replay_json: bool = False) -> EditList:
    """List the commits of the session: the edits with their reasons, and the saves.

    Parameters
    ----------
    store: ModelStore
        The session.
    include_replay_json: bool, default=False
        Add the JSON that ``--edits`` reads to continue the work in a later session.

    Returns
    -------
    EditList
        The table, the ids of the undone edits, and the JSON on request.

    """
    session = store.editing()
    info = session.revision_info()
    rows = session.history_table()
    edit_ids = [row.id for row in rows if row.kind == "edit"]
    return EditList(
        session=info,
        commits=rows,
        undone=edit_ids[info.edit_count :],
        replay_json=session.to_json().decode() if include_replay_json else None,
    )
