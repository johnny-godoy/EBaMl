"""The tools that show the effect of the edits on the data of ``--data``: the metrics in each scope, and the subgroups.

Each function calls ``edit_effect`` or ``affected_profile`` of ``ebaml.edit`` and returns an attrs class. No function
calls an LLM. The server turns each class into JSON.
"""

import attrs

from ebaml.core.custom_types import Frame
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.edit_effect import AffectedProfile, EditEffect, Subgroup, affected_profile, edit_effect
from ebaml.edit.edit_session import CURRENT, RevisionInfo
from ebaml.mcp.model_store import ModelStore

_NO_LABELS = (
    "The server has no label column (--labels), so the result holds the counts of rows and no metric. Tell the person"
    " that nobody checked the accuracy of the edited model."
)


@attrs.frozen
class EffectResult:
    """The effect of the edits on the data, and the state of the session.

    ``note`` says what the result leaves out, for example the metrics when the server has no label column.
    """

    session: RevisionInfo
    effect: EditEffect
    note: str | None


@attrs.frozen
class ProfileResult:
    """The profile of the rows that the last edit reaches, and the state of the session."""

    session: RevisionInfo
    profile: AffectedProfile


def _data(store: ModelStore) -> Frame:
    if store.data is None:
        msg = "The server has no data. Start it with --data to see the effect of the edits on rows."
        raise InvalidInstanceError(msg)
    return store.data


def subgroup_of(feature: str | None, value: str | float | None) -> Subgroup | None:
    """Build the subgroup of the arguments of a tool.

    Parameters
    ----------
    feature: str, optional
        The feature of the subgroup. Without it, the tool has no subgroup.
    value: str | float, optional
        The value of the feature that the rows of the subgroup have.

    Returns
    -------
    Subgroup | None
        The subgroup, or ``None`` if the tool names no feature.

    Raises
    ------
    InvalidInstanceError
        If the tool names one of the two arguments and not the other.

    """
    if feature is None and value is None:
        return None
    if feature is None or value is None:
        msg = "A subgroup needs both subgroup_feature and subgroup_value, for example 'sex' and 'female'."
        raise InvalidInstanceError(msg)
    return Subgroup(feature, value)


def get_edit_effect(
    store: ModelStore,
    subgroup: Subgroup | None = None,
    revision: str = CURRENT,
) -> EffectResult:
    """Measure the effect of the edits on the rows of ``--data``: counts, and metrics if the server has labels.

    Parameters
    ----------
    store: ModelStore
        The session.
    subgroup: Subgroup, optional
        The rows of the third scope.
    revision: str, default="current"
        The revision to describe.

    Returns
    -------
    EffectResult
        The effect, and a note when the server has no label column. Without ``--data``, the call raises an
        ``InvalidInstanceError``.

    """
    session = store.editing()
    labels = store.labels()
    effect = edit_effect(session, _data(store), labels, subgroup, revision=revision)
    return EffectResult(session.revision_info(effect.revision), effect, None if labels is not None else _NO_LABELS)


def get_affected_profile(store: ModelStore, revision: str = CURRENT) -> ProfileResult:
    """Compare the rows that the last edit reaches with all rows, feature by feature.

    Parameters
    ----------
    store: ModelStore
        The session.
    revision: str, default="current"
        The revision whose last edit to describe.

    Returns
    -------
    ProfileResult
        The profile. Without ``--data``, the call raises an ``InvalidInstanceError``.

    """
    session = store.editing()
    profile = affected_profile(session, _data(store), revision=revision)
    return ProfileResult(session.revision_info(profile.revision), profile)
