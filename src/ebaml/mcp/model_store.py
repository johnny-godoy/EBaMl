"""The state of one server session: the EBM, the optional data and the optional LLM options.

The user gives the paths at start. No tool argument names a path, so an agent cannot make the server load a file.
"""

import os

import attrs
import joblib
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError


@attrs.define
class ModelStore:
    """The EBM of the session and what the tools need besides it.

    Attributes
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    data: pd.DataFrame | None
        The rows that the ``row`` argument of the explain tool indexes.
    model_description: str | None
        The output of ``describe_ebm``. The first call computes it and the next calls reuse it.

    """

    ebm: ExplainableBoostingModel
    data: pd.DataFrame | None = None
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
        return {name: self.data.iloc[row][name] for name in names}


def load_model(path: str | os.PathLike[str]) -> ExplainableBoostingModel:
    """Load a pickled or joblib EBM from the path that the user gave.

    Loading a pickle runs code from the file. Call this function only with a path from the command line of the user,
    never with a path from a tool argument.

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
        If the file does not hold an EBM.

    """
    model = joblib.load(path)
    if not isinstance(model, ExplainableBoostingClassifier | ExplainableBoostingRegressor):
        msg = f"The file {os.fspath(path)!r} does not hold an EBM. It holds a {type(model).__name__}."
        raise UnsupportedModelError(msg)
    return model


def load_data(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read the CSV file of the data.

    Parameters
    ----------
    path: str | os.PathLike[str]
        The CSV file.

    Returns
    -------
    pd.DataFrame
        The rows.

    """
    return pd.read_csv(path)
