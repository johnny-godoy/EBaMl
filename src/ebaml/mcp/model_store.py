"""The state of one server session: the EBM, the optional data and the optional LLM options.

The user gives the paths at start. No tool argument names a path, so an agent cannot make the server load a file.
"""

import os
import zipfile

import attrs
import joblib
import narwhals as nw
import polars as pl
import skops.io
import skops.io.exceptions
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue, Frame
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError


@attrs.define
class ModelStore:
    """The EBM of the session and what the tools need besides it.

    Attributes
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    data: Frame | None
        The rows that the ``row`` argument of the explain tool indexes. Wrap a pandas or polars frame with
        ``narwhals.from_native``.
    model_description: str | None
        The output of ``describe_ebm``. The first call computes it and the next calls reuse it.

    """

    ebm: ExplainableBoostingModel
    data: Frame | None = None
    model_description: str | None = None

    def feature_index(self, feature: str) -> int:
        """Find the index of a feature by its name.

        Parameters
        ----------
        feature: str
            The name of the feature.

        Returns
        -------
        int
            The index of the feature in ``feature_names_in_``.

        Raises
        ------
        InvalidInstanceError
            If the model has no feature with this name.

        """
        names = list(self.ebm.feature_names_in_)
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
        names = list(self.ebm.feature_names_in_)
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


def load_data(path: str | os.PathLike[str]) -> Frame:
    """Read the CSV file of the data.

    Parameters
    ----------
    path: str | os.PathLike[str]
        The CSV file.

    Returns
    -------
    Frame
        The rows.

    """
    return nw.from_native(pl.read_csv(os.fspath(path)))  # pyrefly: ignore[no-matching-overload] -- pyrefly rejects every overload of narwhals.from_native for a polars frame
