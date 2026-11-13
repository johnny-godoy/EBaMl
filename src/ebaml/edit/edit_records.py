"""Write an edit and its selection as a record of JSON values, and read it back.

The history of a session stores each edit as a record, so that a replay builds the same edit. A record holds the name
of the class under ``type`` and the fields of the class. A selection inside an edit is a record too.
"""

from collections.abc import Mapping
from typing import cast

import attrs
from beartype.roar import BeartypeCallHintViolation

from ebaml.core.custom_types import JSONMapping, JSONObject, JSONValue
from ebaml.core.exceptions import HistoryError
from ebaml.edit import edits, selections
from ebaml.edit.edits import Edit
from ebaml.edit.selections import Selection

EDIT_TYPES: dict[str, type[Edit]] = {
    edit_class.__name__: edit_class
    for edit_class in (
        edits.Shift,
        edits.SetScores,
        edits.Align,
        edits.Monotonize,
        edits.Interpolate,
        edits.FitLine,
        edits.Scale,
        edits.ZeroTerm,
        edits.Recenter,
        edits.RemoveTerms,
        edits.RemoveFeature,
    )
}
# the name of the field of an edit that holds a selection
SELECTION_FIELD = "selection"
SELECTION_TYPES: dict[str, type[Selection]] = {
    selection_class.__name__: selection_class
    for selection_class in (
        selections.Bins,
        selections.Categories,
        selections.MissingBin,
        selections.UnseenBin,
        selections.WholeGraph,
        selections.ValueInterval,
    )
}


# nosemgrep: ebaml-return-attrs-not-container -- the record is a JSON fragment that the history merges and writes
def to_record(item: Edit | Selection) -> JSONObject:
    """Write an edit or a selection as a record.

    Parameters
    ----------
    item : Edit | Selection
        An attrs class of ``ebaml.edit``. The field ``selection`` of an edit holds a selection, and it becomes a nested
        record.

    Returns
    -------
    JSONObject
        The name of the class under ``type``, and each field.

    """
    fields = attrs.asdict(item, recurse=False)
    # the field "selection" of an edit holds a selection, and a tuple of texts is a valid JSON value for orjson
    nested = {name: to_record(value) for name, value in fields.items() if name == SELECTION_FIELD}
    return {"type": type(item).__name__} | fields | nested


def _arguments(
    record: JSONMapping,
) -> dict[
    str,
    object,
]:  # nosemgrep: ebaml-no-object-type -- the arguments hold the fields of any edit or selection class
    """Read the fields of a record: a nested record is a selection, and a list is a tuple.

    Returns
    -------
    dict[str, object]
        The arguments for the constructor of the class.

    """
    arguments: dict[str, object] = {}  # nosemgrep: ebaml-no-object-type -- same reason
    for field_name, value in record.items():
        match value:
            case {}:
                arguments[field_name] = _decode(value, SELECTION_TYPES, "selection")
            case [*_] if not isinstance(value, str):
                arguments[field_name] = tuple(value)
            case _:
                arguments[field_name] = value
    return arguments


def _decode(
    record: JSONValue,
    registry: Mapping[str, type],
    kind: str,
) -> object:  # nosemgrep: ebaml-no-object-type -- the registry gives the class, and the callers narrow the result
    match record:
        case {"type": str() as name, **fields}:
            pass
        case _:
            msg = f"A {kind} record needs a text under type, but it is {record!r}."
            raise HistoryError(msg)
    if name not in registry:
        msg = (
            f"The history names the {kind} {name!r}, which this version does not know."
            f" The known ones are {sorted(registry)}."
        )
        raise HistoryError(msg)
    try:
        return registry[name](**_arguments(fields))
    except (TypeError, ValueError, BeartypeCallHintViolation) as error:
        msg = f"The record of the {kind} {name!r} does not fit its class: {error}"
        raise HistoryError(msg) from error


def edit_from_record(record: JSONValue) -> Edit:
    """Build an edit from its record.

    Parameters
    ----------
    record : JSONValue
        A record that ``to_record`` wrote.

    Returns
    -------
    Edit
        The edit. A record that names an unknown class, or has fields that do not fit it, raises a ``HistoryError``.

    """
    return cast(Edit, _decode(record, EDIT_TYPES, "edit"))  # the registry holds the classes of edits only
