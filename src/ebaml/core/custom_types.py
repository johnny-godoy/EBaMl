"""Types for EBM models, and the coercion of options to them."""

import enum
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from decimal import Decimal

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
    ExplainableBoostingRegressor,  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
)
from pandas.api.typing import NAType

from ebaml.core.exceptions import InvalidOptionError

ExplainableBoostingModel = ExplainableBoostingClassifier | ExplainableBoostingRegressor

# A label of a class, as NumPy's tolist() and item() return it. The numeric tower lets float cover int and bool.
type ClassLabel = str | float
# A value of one feature of an instance. A row of a pandas DataFrame gives NumPy scalars, a nullable column gives pd.NA,
# and None, NaN and pd.NA are missing values.
type FeatureValue = str | float | Decimal | np.generic | NAType | None
# A JSON value of the texts for the prompts. The abstract containers keep a list[str] or a dict[str, float] assignable,
# so the hint also accepts a Sequence or a Mapping that orjson cannot write (a range, a MappingProxyType). orjson raises
# for those when it writes the text.
type JSONValue = str | float | Sequence[JSONValue] | Mapping[str, JSONValue] | None


class Normalization(enum.StrEnum):
    """Normalization options for EBM graphs."""

    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


class FeatureType(enum.StrEnum):
    """Feature types for EBM graphs."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    BOOLEAN = "boolean"


def normalize_feature_type(feature_type: str) -> FeatureType:
    """Normalize a feature type to the FeatureType enum.

    Parameters
    ----------
    feature_type : str
        The feature type to normalize.

    Returns
    -------
    FeatureType
        The normalized feature type.

    Raises
    ------
    InvalidOptionError.from_enum
        If an invalid feature type is provided.

    """
    if feature_type.upper() in FeatureType._member_names_:
        return FeatureType[feature_type.upper()]
    raise InvalidOptionError.from_enum(feature_type, FeatureType)


def coerce_enum[E: enum.Enum](enum_class: type[E], value: E | str) -> E:
    """Convert a value (a member or its string value) to a member of an enum.

    Parameters
    ----------
    enum_class : type[E]
        The enum class.
    value : E | str
        The member or the value to convert.

    Returns
    -------
    E
        The enum member.

    Raises
    ------
    InvalidOptionError.from_enum
        If the value is not a member of the enum.

    """
    try:
        return enum_class(value)
    except ValueError:
        raise InvalidOptionError.from_enum(str(value), enum_class) from None


def normalize_feature_types(
    feature_types: Iterable[str],
    reject_set: AbstractSet[FeatureType] | None = None,
) -> list[FeatureType]:
    """Normalize a list of feature types to the FeatureType enum.

    Parameters
    ----------
    feature_types: Iterable[str]
        The feature types.
    reject_set: AbstractSet[FeatureType], optional
        Set of feature types to reject.
        If a feature type is in this set, the function raises an InvalidOptionError.

    Returns
    -------
        list[FeatureType]: List of normalized feature types.

    Raises
    ------
    InvalidOptionError
        If an invalid feature type is provided or if a feature type is in the reject set.

    """
    normalized_feature_types = []
    reject_set = reject_set if reject_set is not None else set()
    accept_set = {f for f in FeatureType if f not in reject_set}
    for feature_type in feature_types:
        if feature_type.upper() in FeatureType._member_names_:
            value = FeatureType[feature_type.upper()]
            if value in reject_set:
                raise InvalidOptionError(feature_type, accept_set)
            normalized_feature_types.append(value)
        else:
            raise InvalidOptionError(feature_type, accept_set)
    return normalized_feature_types
