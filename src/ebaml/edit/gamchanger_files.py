"""Read a ``.gamchanger`` file of GAM Changer, and turn its commits into edits.

GAM Changer (Wang et al., KDD 2022, MIT licence) saves its work as JSON with three keys: ``modelData`` (the model that
it edited), ``sampleData`` (rows of data) and ``historyList`` (the commits). This module follows ``get_edited_model`` of
the GAM Changer package: it reads the newest commit of each feature whose type is not ``original``, and it ignores the
commit that the person checked out. It differs from ``get_edited_model`` in three ways:

- It emits an edit only for a bin whose score differs from the score of the model rounded to 4 decimals, because GAM
  Changer writes rounded scores. The other bins keep their full precision, so the result differs from
  ``get_edited_model`` by up to 0.00005 on those bins.
- It refuses a file that GAM Changer made from another model: other features, other bin edges, or other scores.
- It loads a commit that changes the bin edges (``equal-interpolate`` and ``equal-regression``) as a ``Rebin`` of the
  range that the commit replaced, followed by the scores of the commit. It refuses new edges that are not of equal
  width, because no GAM Changer tool makes them and no ``Rebin`` repeats them.

GAM Changer cannot edit the missing bin or the unseen bin, so the edits leave both as they are.
"""

import abc
import copy
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, override

import attrs
import numpy as np
import numpy.typing as npt
import orjson

from ebaml.core.custom_types import ExplainableBoostingModel, JSONMapping, JSONValue
from ebaml.core.exceptions import HistoryError
from ebaml.edit.bin_edits import Rebin
from ebaml.edit.edit_report import EditReport
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import AssignScores, Edit
from ebaml.edit.model_view import TermView, term_view
from ebaml.edit.selections import Bins, Categories
from ebaml.graphs.terms import main_effect_features

# GAM Changer rounds every score that it writes to this number of decimals
GAMCHANGER_DECIMALS = 4
# a score that JSON carries keeps its value exactly, so this tolerance only absorbs the rounding of numpy
_SCORE_TOLERANCE = 1e-9
_EDGE_TOLERANCE = 1e-9
_ORIGINAL = "original"
_INTERACTION = "interaction"
_MILLISECONDS = 1000


@attrs.frozen
class GamChangerCommit:
    """The edits that carry the newest GAM Changer commit of one feature.

    Attributes
    ----------
    feature: str
        The feature that the commit changed.
    commit_type: str
        The type of the commit in GAM Changer, for example ``increasing`` or ``align``.
    description: str
        The description that GAM Changer wrote for the commit.
    reviewed: bool
        Whether a person confirmed the commit in GAM Changer.
    time: datetime
        The time of the commit, in UTC.
    commit_hash: str
        The hash of the commit in GAM Changer.
    edits: tuple[Edit, ...]
        The edits that set the scores of the commit. Each edit covers one range of neighbour bins that the commit
        changed, or every changed category of a nominal feature.

    """

    feature: str
    commit_type: str
    description: str
    reviewed: bool
    time: datetime
    commit_hash: str
    edits: tuple[Edit, ...]


def _field(record: JSONMapping, key: str, what: str) -> JSONValue:
    if key not in record:
        msg = f"The {what} of the .gamchanger file has no field {key!r}. The file is not a GAM Changer document."
        raise HistoryError(msg)
    return record[key]


def _mapping(value: JSONValue, what: str) -> JSONMapping:
    # a parsed JSON value at the edge of ebaml: narrow the union once
    if not isinstance(value, Mapping):
        msg = f"The {what} of the .gamchanger file is not a JSON object. The file is not a GAM Changer document."
        raise HistoryError(msg)
    return value


def _sequence(value: JSONValue, what: str) -> Sequence[JSONValue]:
    # a parsed JSON value at the edge of ebaml: narrow the union once, and a text is not a list
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"The {what} of the .gamchanger file is not a JSON list. The file is not a GAM Changer document."
        raise HistoryError(msg)
    return value


def _numbers(value: JSONValue, what: str) -> npt.NDArray[np.float64]:
    try:
        return np.asarray(_sequence(value, what), dtype=float)
    except (TypeError, ValueError) as error:
        msg = f"The {what} of the .gamchanger file holds a value that is not a number."
        raise HistoryError(msg) from error


def _rounded_scores(view: TermView) -> npt.NDArray[np.float64]:
    """Give the scores of the real bins as GAM Changer writes them.

    Returns
    -------
    npt.NDArray[np.float64]
        The score of each real bin, rounded to 4 decimals.

    """
    return np.round(view.scores[1:-1], GAMCHANGER_DECIMALS)


def _model_mismatch(feature: str, part: str) -> HistoryError:
    return HistoryError(
        f"The .gamchanger file does not fit the model: the {part} of {feature!r} differ. GAM Changer made the file from"
        " another model. Load the model that you edited in GAM Changer.",
    )


@attrs.frozen
class _Points:
    """The bins of one feature in one commit: the place or the category of each bin, and its score."""

    places: tuple[float | str, ...]
    scores: npt.NDArray[np.float64]


def _changed(current: npt.NDArray[np.float64], scores: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return ~np.isclose(current, scores, rtol=0.0, atol=_SCORE_TOLERANCE)


def _check_scores(view: TermView, feature_data: JSONMapping) -> None:
    file_scores = _numbers(_field(feature_data, "additive", "model data"), "scores")
    current = _rounded_scores(view)
    if len(file_scores) != len(current) or _changed(current, file_scores).any():
        raise _model_mismatch(view.feature_name, "scores")


class _FeatureFormat(abc.ABC):
    """How a .gamchanger file writes the bins of one type of feature."""

    @abc.abstractmethod
    def check_model(self, view: TermView, feature_data: JSONMapping, label_encoder: JSONMapping) -> None:
        """Check that the model data of the file holds the bins and the scores of the model.

        Raises
        ------
        HistoryError
            If the bins or the scores differ.

        """

    @abc.abstractmethod
    def points(self, point_data: JSONMapping) -> _Points:
        """Read the bins of a commit in their order.

        Returns
        -------
        _Points
            The place or the category of each bin, and its score.

        """

    @abc.abstractmethod
    def edits(self, view: TermView, points: _Points, commit: str) -> tuple[Edit, ...]:
        """Build the edits that set the scores of a commit.

        Returns
        -------
        tuple[Edit, ...]
            The edits, or no edit if the commit changes no score.

        """


def _shared_count(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> int:
    """Count the leading values that two lists of edges share.

    Returns
    -------
    int
        The number of leading positions where the edges are equal, within the tolerance.

    """
    length = min(len(first), len(second))
    equal = np.isclose(first[:length], second[:length], rtol=_EDGE_TOLERANCE, atol=0.0)
    return int(np.argmin(equal)) if not equal.all() else length


def _rebin_of(view: TermView, places: npt.NDArray[np.float64], commit: str) -> Rebin:
    """Find the ``Rebin`` that turns the bins of the model into the bins of a commit.

    GAM Changer keeps the left edge of the first selected bin and the bins after the selection, and replaces the bins
    between by bins of equal width. So the edges that both lists share at the start and at the end stay, and the bins
    between them are the ones that the commit replaced.

    Returns
    -------
    Rebin
        The edit.

    Raises
    ------
    HistoryError
        If the new bins between the shared edges are not of equal width.

    """
    old_edges = view.left_edges()
    prefix = _shared_count(old_edges, places)
    suffix = _shared_count(old_edges[prefix:][::-1], places[prefix:][::-1])
    first, old_last, new_last = prefix - 1, len(old_edges) - suffix - 1, len(places) - suffix - 1
    right = old_edges[old_last + 1] if suffix else view.bin_edges()[-1][1]
    new_count = new_last - first + 1
    if first >= 0 and new_count >= 1:
        expected = np.linspace(old_edges[first], right, new_count + 1)[:-1]
        if np.allclose(places[first : new_last + 1], expected, rtol=_EDGE_TOLERANCE, atol=_EDGE_TOLERANCE * abs(right)):
            return Rebin(view.feature_name, Bins(first, old_last), new_count)
    msg = (
        f"The commit {commit} changes the bin edges of {view.feature_name!r} ({len(old_edges)} bins become"
        f" {len(places)}) in a way that no Rebin repeats: the new bins are not of equal width. GAM Changer makes such"
        " bins with no tool, so the file may be damaged."
    )
    raise HistoryError(msg)


class _ContinuousFormat(_FeatureFormat):
    """A continuous feature: ``binEdge`` in the model data, and a linked list of points that starts at key ``0``."""

    @override
    def check_model(self, view: TermView, feature_data: JSONMapping, label_encoder: JSONMapping) -> None:
        edges = [left for left, _ in view.bin_edges()] + [view.bin_edges()[-1][1]]
        file_edges = _numbers(_field(feature_data, "binEdge", "model data"), "bin edges")
        if len(file_edges) != len(edges) or not np.allclose(file_edges, edges, rtol=_EDGE_TOLERANCE, atol=0.0):
            raise _model_mismatch(view.feature_name, "bin edges")
        _check_scores(view, feature_data)

    @override
    def points(self, point_data: JSONMapping) -> _Points:
        places: list[float | str] = []
        scores: list[float] = []
        key: JSONValue = 0
        # the list has one point for each bin, so a longer walk means a cycle
        for _ in point_data:
            point = _mapping(_field(point_data, str(key), "point list"), "point")
            places.append(float(cast(float, _field(point, "x", "point"))))
            scores.append(float(cast(float, _field(point, "y", "point"))))
            key = _field(point, "rightPointID", "point")
            if key is None:
                return _Points(tuple(places), np.asarray(scores, dtype=float))
        msg = "The point list of a commit in the .gamchanger file does not end. The file is damaged."
        raise HistoryError(msg)

    @override
    def edits(self, view: TermView, points: _Points, commit: str) -> tuple[Edit, ...]:
        left_edges = view.left_edges()
        places = np.asarray(points.places, dtype=float)
        bin_edits: tuple[Edit, ...] = ()
        if len(places) != len(left_edges) or not np.allclose(places, left_edges, rtol=_EDGE_TOLERANCE, atol=0.0):
            rebin = _rebin_of(view, places, commit)
            rebinned = copy.deepcopy(view.ebm)
            rebin.run(rebinned)
            view = term_view(rebinned, view.feature_name)
            bin_edits = (rebin,)
        changed = np.flatnonzero(_changed(_rounded_scores(view), points.scores))
        # one edit for each range of neighbour bins, so that no unchanged bin loses its precision
        runs = np.split(changed, np.flatnonzero(np.diff(changed) > 1) + 1) if len(changed) else []
        return bin_edits + tuple(
            AssignScores(
                view.feature_name,
                Bins(int(run[0]), int(run[-1])),
                tuple(points.scores[run].tolist()),
            )
            for run in runs
        )


class _NominalFormat(_FeatureFormat):
    """A nominal feature: level codes in the model data, and one point for each level, with the level name as ``x``."""

    @override
    def check_model(self, view: TermView, feature_data: JSONMapping, label_encoder: JSONMapping) -> None:
        names = _mapping(_field(label_encoder, view.feature_name, "label encoder"), "label encoder")
        codes = _sequence(_field(feature_data, "binLabel", "model data"), "bin labels")
        categories = view.categories()
        try:
            file_categories = [names[str(code)] for code in codes]
        except KeyError as error:
            raise _model_mismatch(view.feature_name, "categories") from error
        if file_categories != categories:
            raise _model_mismatch(view.feature_name, "categories")
        _check_scores(view, feature_data)

    @override
    def points(self, point_data: JSONMapping) -> _Points:
        points = [_mapping(point, "point") for point in point_data.values()]
        return _Points(
            tuple(str(_field(point, "x", "point")) for point in points),
            np.asarray([cast(float, _field(point, "y", "point")) for point in points], dtype=float),
        )

    @override
    def edits(self, view: TermView, points: _Points, commit: str) -> tuple[Edit, ...]:
        categories = view.categories()
        unknown = [name for name in points.places if name not in categories]
        if unknown:
            msg = f"The commit {commit} names categories of {view.feature_name!r} that the model lacks: {unknown}."
            raise HistoryError(msg)
        file_scores = dict(zip(points.places, points.scores.tolist(), strict=True))
        current = _rounded_scores(view)
        changed = [
            (name, file_scores[name])
            for name, score in zip(categories, current.tolist(), strict=True)
            if name in file_scores and abs(file_scores[name] - score) > _SCORE_TOLERANCE
        ]
        if not changed:
            return ()
        names, values = zip(*changed, strict=True)
        return (AssignScores(view.feature_name, Categories(names), values),)


# the feature types of GAM Changer, which writes "categorical" for a nominal feature
_FORMATS: dict[str, _FeatureFormat] = {"continuous": _ContinuousFormat(), "categorical": _NominalFormat()}
_TYPE_NAMES = {"continuous": "continuous", "nominal": "categorical"}


def _format_of(view: TermView) -> _FeatureFormat:
    if view.feature_type not in _TYPE_NAMES:
        msg = (
            f"GAM Changer edits continuous and nominal features only, and {view.feature_name!r} is"
            f" {view.feature_type}. GAM Changer cannot have made this file from this model."
        )
        raise HistoryError(msg)
    return _FORMATS[_TYPE_NAMES[view.feature_type]]


def _check_model(ebm: ExplainableBoostingModel, model_data: JSONMapping) -> None:
    features = [
        _mapping(feature, "feature") for feature in _sequence(_field(model_data, "features", "model data"), "features")
    ]
    by_name = {
        str(_field(feature, "name", "feature")): feature
        for feature in features
        if _field(feature, "type", "feature") != _INTERACTION
    }
    names = [ebm.feature_names_in_[index] for index in main_effect_features(ebm)]
    if sorted(by_name) != sorted(names):
        msg = (
            f"The .gamchanger file does not fit the model: the file has the features {sorted(by_name)}, and the model"
            f" has {sorted(names)}. Load the model that you edited in GAM Changer."
        )
        raise HistoryError(msg)
    label_encoder = _mapping(_field(model_data, "labelEncoder", "model data"), "label encoder")
    # check every type first, so that a feature that GAM Changer cannot edit gets its own message
    views = {name: term_view(ebm, name) for name in names}
    formats = {name: _format_of(view) for name, view in views.items()}
    for name, feature_format in formats.items():
        if _FORMATS.get(str(_field(by_name[name], "type", "feature"))) is not feature_format:
            raise _model_mismatch(name, "feature types")
    for name, feature_format in formats.items():
        feature_format.check_model(views[name], by_name[name], label_encoder)


def _commit(ebm: ExplainableBoostingModel, entry: JSONMapping) -> GamChangerCommit:
    feature = str(_field(entry, "featureName", "commit"))
    commit_hash = str(_field(entry, "hash", "commit"))
    commit_type = str(_field(entry, "type", "commit"))
    view = term_view(ebm, feature)
    feature_format = _format_of(view)
    state = _mapping(_field(entry, "state", "commit"), "state")
    points = feature_format.points(_mapping(_field(state, "pointData", "state"), "point list"))
    milliseconds = float(cast(float, _field(entry, "time", "commit")))
    return GamChangerCommit(
        feature=feature,
        commit_type=commit_type,
        description=str(_field(entry, "description", "commit")),
        reviewed=_field(entry, "reviewed", "commit") is True,
        time=datetime.fromtimestamp(milliseconds / _MILLISECONDS, UTC),
        commit_hash=commit_hash,
        edits=feature_format.edits(view, points, f"{commit_hash[:7]} ({commit_type})"),
    )


def load_gamchanger(path: str | os.PathLike[str], ebm: ExplainableBoostingModel) -> list[GamChangerCommit]:
    """Read a ``.gamchanger`` file, and build the edits of the newest commit of each feature.

    The function reads the file and changes no model. Apply the result with ``apply_gamchanger``.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The file that GAM Changer saved.
    ebm : ExplainableBoostingModel
        The model that the person edited in GAM Changer, before the edits.

    Returns
    -------
    list[GamChangerCommit]
        One commit for each feature that a commit changed, in the order of the history. A commit that changes no score
        has no edit, and the list leaves it out.

    Raises
    ------
    HistoryError
        If the file is not a GAM Changer document, if it does not fit the model, or if a commit changes the bin edges
        in a way that no ``Rebin`` repeats.

    """
    try:
        document = _mapping(orjson.loads(Path(path).read_bytes()), "document")
    except orjson.JSONDecodeError as error:
        msg = f"The file {os.fspath(path)!r} is not JSON, so it is not a .gamchanger document: {error}"
        raise HistoryError(msg) from error
    _check_model(ebm, _mapping(_field(document, "modelData", "document"), "model data"))
    entries = [_mapping(entry, "commit") for entry in _sequence(_field(document, "historyList", "document"), "history")]
    newest: dict[str, JSONMapping] = {}
    for entry in reversed(entries):
        if _field(entry, "type", "commit") != _ORIGINAL:
            newest.setdefault(str(_field(entry, "featureName", "commit")), entry)
    commits = [_commit(ebm, entry) for entry in entries if any(entry is chosen for chosen in newest.values())]
    return [commit for commit in commits if commit.edits]


def apply_gamchanger(session: EditSession, commits: Sequence[GamChangerCommit]) -> list[EditReport]:
    """Apply the edits of GAM Changer commits to a session, one revision for each edit.

    The reason of each edit names the GAM Changer commit and holds its description. An edit of a commit that a person
    confirmed in GAM Changer is marked as reviewed. Preview the edits, and check their effect on data before you save.

    Parameters
    ----------
    session : EditSession
        The session of the model that the commits fit.
    commits : Sequence[GamChangerCommit]
        The result of ``load_gamchanger``.

    Returns
    -------
    list[EditReport]
        The report of each edit, in the order of the commits.

    """
    reports: list[EditReport] = []
    for commit in commits:
        for edit in commit.edits:
            report = session.apply(edit, f"GAM Changer commit {commit.commit_hash[:7]}: {commit.description}")
            if commit.reviewed:
                session.review(cast(str, report.revision))  # apply always sets the revision that it made
            reports.append(report)
    return reports
