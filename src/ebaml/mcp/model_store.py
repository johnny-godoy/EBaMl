"""The state of one server session: the EBM, the optional data and the optional LLM options.

The user gives the paths at start. No tool argument names a path, so an agent cannot make the server load a file.
"""

import codecs
import os
import zipfile
from pathlib import Path

import attrs
import joblib  # ruff: ignore[banned-api] -- ebaml-mcp also loads the joblib files that users saved, and the user trusts the file by naming its path
import narwhals as nw
import polars as pl
import skops.io
import skops.io.exceptions

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue, Frame
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import DataFileError, EditError, InvalidInstanceError, UnsupportedModelError
from ebaml.edit.edit_session import CURRENT, ORIGINAL, EditSession, RevisionInfo


@attrs.frozen(eq=False)
class RevisionRead:
    """A copy of the model at one revision, and the state of the session when the tool read it.

    Attributes
    ----------
    model: ExplainableBoostingModel
        A copy of the model at ``info.revision``. A reader can use it with no lock, because an edit never changes it.
    info: RevisionInfo
        The fixed name of the revision, the head, and the number of edits that the head holds.

    """

    model: ExplainableBoostingModel
    info: RevisionInfo


@attrs.frozen
class _Described:
    """The text of ``describe_ebm`` and the revision that it describes."""

    revision: str
    text: str


def _start_session(ebm: ExplainableBoostingModel) -> EditSession | None:
    """Start an edit session on a model, or give ``None`` if the edits do not support the model.

    Returns
    -------
    EditSession | None
        The session. The reads of the server work with no session, and the edit tools raise the error of the session.

    """
    try:
        return EditSession(ebm)
    except UnsupportedModelError:
        return None


@attrs.define
class ModelStore:
    """The EBM of the session and what the tools need besides it.

    The store never changes the loaded model. The edit session works on a copy, so each read of a revision gives the
    model with the edits of that revision.

    Attributes
    ----------
    original: ExplainableBoostingModel
        The fitted EBM that the user loaded. The constructor takes it as ``ebm``.
    data: Frame | None
        The rows that the ``row`` argument of the explain tool indexes. Wrap a pandas or polars frame with
        ``narwhals.from_native``.
    session: EditSession | None
        The edits of the session. The default is a new session on ``original``, or ``None`` for a model that the
        edits do not support (a multiclass classifier). Pass a replayed session to continue the work of an earlier
        session.

    """

    original: ExplainableBoostingModel = attrs.field(alias="ebm")
    data: Frame | None = None
    session: EditSession | None = attrs.field(
        default=attrs.Factory(lambda self: _start_session(self.original), takes_self=True),
    )
    _described: _Described | None = attrs.field(default=None, init=False)

    def read(self, revision: str = CURRENT) -> RevisionRead:
        """Read the model at a revision. The read never changes the model of the session.

        Parameters
        ----------
        revision: str
            ``original``, ``current`` or the id of a revision of the history.

        Returns
        -------
        RevisionRead
            A copy of the model at the revision, and the state of the session. ``current`` becomes the id of the head.

        Raises
        ------
        EditError
            If the model has no edit session and the revision is not ``original`` or ``current``. The session raises
            the same error for a revision that it does not hold.

        """
        if self.session is None:
            if revision not in {ORIGINAL, CURRENT}:
                msg = (
                    f"The model has no edits, so its only revisions are {ORIGINAL!r} and {CURRENT!r}, not {revision!r}."
                )
                raise EditError(msg)
            return RevisionRead(self.original, RevisionInfo(ORIGINAL, ORIGINAL, 0))
        info = self.session.revision_info(revision)
        return RevisionRead(self.session.model_at(info.revision), info)

    def editing(self) -> EditSession:
        """Give the edit session.

        Returns
        -------
        EditSession
            The session that the edit tools use.

        Raises
        ------
        UnsupportedModelError
            If the model is one that the edits do not support. The message of ``EditSession`` says which.

        """
        if self.session is None:
            msg = "Editing supports binary classifiers and regressors. A multiclass classifier has no edit session."
            raise UnsupportedModelError(msg)
        return self.session

    def described(self, revision: str) -> str | None:
        """Give the kept description of the model if it describes this revision.

        Parameters
        ----------
        revision: str
            The fixed name of a revision, as ``RevisionRead.info.revision`` gives it.

        Returns
        -------
        str | None
            The output of ``describe_ebm``, or ``None`` if the store keeps none for this revision.

        """
        kept = self._described
        return kept.text if kept is not None and kept.revision == revision else None

    def keep_description(self, revision: str, text: str) -> None:
        """Keep the description of a revision, but only if the head is still this revision.

        An edit during the slow LLM call moves the head. The description then describes a model that is no longer the
        current one, so the store drops it.

        Parameters
        ----------
        revision: str
            The fixed name of the revision that the text describes.
        text: str
            The output of ``describe_ebm``.

        """
        head = ORIGINAL if self.session is None else self.session.revision
        if head == revision:
            self._described = _Described(revision, text)

    def feature_index(self, feature: str) -> int:
        """Find the index of a feature by its name.

        Parameters
        ----------
        feature: str
            The name of the feature.

        Returns
        -------
        int
            The index of the feature in ``feature_names_in_``. Edits never change the features of the model.

        Raises
        ------
        InvalidInstanceError
            If the model has no feature with this name.

        """
        names = list(self.original.feature_names_in_)
        if feature not in names:
            msg = f"The model has no feature {feature!r}. The features are: {names}."
            raise InvalidInstanceError(msg)
        return names.index(feature)

    # nosemgrep: ebaml-return-attrs-not-container -- the keys are the feature names of the model, so they are data
    def row_instance(self, row: int) -> dict[str, FeatureValue]:
        """Read one row of the data as an instance of the model.

        Parameters
        ----------
        row: int
            The position of the row in the data of ``--data``.

        Returns
        -------
        dict[str, FeatureValue]
            The value of each feature of the model.

        Raises
        ------
        InvalidInstanceError
            If the server has no data, the row does not exist, or the data lacks a feature.

        """
        if self.data is None:
            msg = "The server has no data. Start it with --data to use the row argument, or pass an instance."
            raise InvalidInstanceError(msg)
        if not 0 <= row < len(self.data):
            msg = f"The row {row} does not exist. The data has {len(self.data)} rows, numbered from 0."
            raise InvalidInstanceError(msg)
        names = list(self.original.feature_names_in_)
        missing = [name for name in names if name not in self.data.columns]
        if missing:
            msg = f"The data has no column for these features: {missing}."
            raise InvalidInstanceError(msg)
        values = dict(zip(self.data.columns, self.data.row(row), strict=True))
        return {name: values[name] for name in names}


_EBM_CLASSES = (ExplainableBoostingClassifier, ExplainableBoostingRegressor)
# skops loads only the types that the caller names. The EBM classes are the only types that the file may need.
_TRUSTED_TYPES = [f"{cls.__module__}.{cls.__qualname__}" for cls in _EBM_CLASSES]


def _load_skops(
    path: str | os.PathLike[str],
) -> object:  # nosemgrep: ebaml-no-object-type -- a file of any content, which the caller checks
    """Load a skops file and trust only the EBM classes.

    Parameters
    ----------
    path: str | os.PathLike[str]
        The skops file.

    Returns
    -------
    object
        The object in the file.

    Raises
    ------
    UnsupportedModelError
        If the file holds a type other than an EBM class.

    """
    try:
        return skops.io.load(os.fspath(path), trusted=_TRUSTED_TYPES)
    except skops.io.exceptions.UntrustedTypesFoundException as error:
        msg = f"The file {os.fspath(path)!r} holds types that the server does not trust: {error}"
        raise UnsupportedModelError(msg) from error


def load_model(path: str | os.PathLike[str]) -> ExplainableBoostingModel:
    """Load an EBM from a skops file, or from a pickled or joblib file, of the path that the user gave.

    A skops file is a zip archive. The loader trusts only the EBM classes, so a skops file runs no code. A pickled or
    joblib file runs code from the file. Call this function only with a path from the command line of the user, never
    with a path from a tool argument.

    Parameters
    ----------
    path: str | os.PathLike[str]
        The file of the fitted EBM.

    Returns
    -------
    ExplainableBoostingModel
        The EBM.

    Raises
    ------
    UnsupportedModelError
        If the file does not hold an EBM, or a skops file holds an untrusted type.

    """
    model = _load_skops(path) if zipfile.is_zipfile(path) else joblib.load(path)
    if not isinstance(model, _EBM_CLASSES):
        msg = f"The file {os.fspath(path)!r} does not hold an EBM. It holds a {type(model).__name__}."
        raise UnsupportedModelError(msg)
    return model


_PARQUET_MAGIC = b"PAR1"
_SAMPLE_SIZE = 8192
_FORMAT_FIX = "Save the rows with DataFrame.write_parquet (best) or DataFrame.write_csv, as UTF-8 text."


@attrs.frozen
class _FileProbe:
    """What the first and the last bytes of a file say about its format.

    Attributes
    ----------
    size: int
        The size of the file in bytes.
    starts_with_mark: bool
        Whether the file starts with the Parquet mark ``PAR1``.
    ends_with_mark: bool
        Whether the file ends with the Parquet mark ``PAR1``.
    is_text: bool
        Whether the first bytes are UTF-8 text with no NUL byte.

    """

    size: int
    starts_with_mark: bool
    ends_with_mark: bool
    is_text: bool

    @property
    def is_parquet(self) -> bool:
        """Whether the file starts and ends with the Parquet mark."""
        return self.starts_with_mark and self.ends_with_mark

    def problem(self) -> str | None:
        """Say why the file is neither Parquet nor CSV.

        Returns
        -------
        str | None
            The reason, or ``None`` if the file is Parquet or looks like CSV text. Text that starts with ``PAR1`` is a
            CSV file, for example one whose first column is ``PAR1_id``, because a Parquet body is binary.

        """
        if self.size == 0:
            return "The file is empty."
        if self.is_parquet or self.is_text:
            return None
        if self.starts_with_mark:
            return "The file starts with the Parquet mark but does not end with it, so it is cut off."
        return "The file is neither Parquet nor UTF-8 CSV: it has no Parquet mark, and it is not UTF-8 text."


def _probe(path: str | os.PathLike[str]) -> _FileProbe:
    """Read the first and the last bytes of a file.

    Returns
    -------
    _FileProbe
        The facts that the bytes show.

    """
    magic_size = len(_PARQUET_MAGIC)
    with Path(path).open("rb") as file:
        sample = file.read(_SAMPLE_SIZE)
        size = file.seek(0, os.SEEK_END)
        file.seek(max(size - magic_size, 0))
        tail = file.read(magic_size)
    # The incremental decoder accepts a multibyte character that the end of the sample cuts in two.
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(sample)
    except UnicodeDecodeError:
        is_text = False
    else:
        is_text = b"\x00" not in sample
    return _FileProbe(
        size=size,
        starts_with_mark=sample.startswith(_PARQUET_MAGIC),
        ends_with_mark=tail == _PARQUET_MAGIC,
        is_text=is_text,
    )


def load_data(path: str | os.PathLike[str]) -> Frame:
    """Read the rows from a Parquet file, or from a CSV file.

    The function reads the first and the last bytes of the file, not its extension. A file that starts and ends with
    the Parquet mark ``PAR1`` is a Parquet file, and Parquet keeps the type of each column. A file of UTF-8 text is a
    CSV file, and polars infers the type of each column from the text.

    Parameters
    ----------
    path: str | os.PathLike[str]
        The Parquet file, or the CSV file.

    Returns
    -------
    Frame
        The rows.

    Raises
    ------
    DataFileError
        If the file is empty, is cut off, is neither Parquet nor UTF-8 CSV, or the reader fails. The message names the
        file and says how to save it. Polars reads a binary file as a CSV file of no rows, so the check of the bytes
        comes first.

    """
    probe = _probe(path)
    problem = probe.problem()
    if problem is not None:
        msg = f"Cannot read the data file {os.fspath(path)!r}. {problem} {_FORMAT_FIX}"
        raise DataFileError(msg)
    try:
        frame = pl.read_parquet(os.fspath(path)) if probe.is_parquet else pl.read_csv(os.fspath(path))
    except pl.exceptions.PolarsError as error:
        format_name = "Parquet" if probe.is_parquet else "CSV"
        msg = f"Cannot read the data file {os.fspath(path)!r} as {format_name}: {error} {_FORMAT_FIX}"
        raise DataFileError(msg) from error
    return nw.from_native(frame)  # pyrefly: ignore[no-matching-overload] -- pyrefly rejects every overload of narwhals.from_native for a polars frame
