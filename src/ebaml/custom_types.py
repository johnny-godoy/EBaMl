"""Types for EBM models, and the coercion of options to them."""

import enum
from collections.abc import Iterable
from collections.abc import Set as AbstractSet

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

from ebaml.exceptions import InvalidOptionError

ExplainableBoostingModel = ExplainableBoostingClassifier | ExplainableBoostingRegressor


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


class Audience(enum.StrEnum):
    """Who reads a local explanation."""

    LAYPERSON = "layperson"
    DATA_SCIENTIST = "data_scientist"


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


def coerce_enum[E: enum.Enum](enum_class: type[E], value: object) -> E:
    """Convert a value (a member or its string value) to a member of an enum.

    Parameters
    ----------
    enum_class : type[E]
        The enum class.
    value : object
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
        If a feature is in this set, we raise a ValueError.

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
