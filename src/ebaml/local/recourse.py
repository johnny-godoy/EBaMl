"""The limits of a recourse plan: the features that must not change, and the values that each feature may take.

A counterfactual is a plan that a person carries out. The person may be unable to change some features (an age, a
past event), and may reach only some values of the others. ``RecourseOptions`` states these limits, and the
counterfactual keeps to them. The sufficient reason and the completions describe what the world can be, not what the
person can do, so they ignore the limits.

The section "Allowed bins of a recourse plan" of ``proofs/papers/local-explanations/paper.typ`` proves that the
counterfactual stays the fewest changes under the limits, that it never changes an immutable feature, and that fewer
allowed values never make a flip possible.
"""

import enum
import math
import types
import typing
from collections.abc import Callable, Collection, Mapping, Sequence

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import FeatureType, Interval
from ebaml.core.exceptions import InvalidOptionError, RecourseOptionError
from ebaml.local.costs import CostStrategy, UnitCost
from ebaml.local.fact_features import Feature, bin_label


class RangeMatch(enum.StrEnum):
    """Which bins of a continuous feature a ``ValueRange`` allows.

    An EBM groups the values of a continuous feature in bins, so a range of values and a bin can overlap in part.

    Attributes
    ----------
    OVERLAP
        A bin is allowed when some of its values are in the range. Use it when the person can reach any value in the
        range, so a bin with one reachable value is a real option.
    INSIDE
        A bin is allowed only when all of its values are in the range. Use it when every value of the chosen bin must
        be reachable, for example when the plan must hold for the whole bin.

    """

    OVERLAP = "overlap"
    INSIDE = "inside"


def _sorted_names(names: Collection[str]) -> tuple[str, ...]:
    return tuple(sorted(set(names)))


def _overlaps(low: float, high: float, bin_range: Interval) -> bool:
    bin_low, bin_high = bin_range
    return bin_low <= high and bin_high > low


def _inside(low: float, high: float, bin_range: Interval) -> bool:
    bin_low, bin_high = bin_range
    return bin_low >= low and bin_high <= high


# the test of each way to match a bin with a range: a new way adds a row (solid.md, O)
_MATCHES: Mapping[RangeMatch, Callable[[float, float, Interval], bool]] = types.MappingProxyType(
    {RangeMatch.OVERLAP: _overlaps, RangeMatch.INSIDE: _inside},
)


def _to_match(match: RangeMatch | str) -> RangeMatch:
    try:
        return RangeMatch(match)
    except ValueError as error:
        raise InvalidOptionError.from_enum(str(match), RangeMatch) from error


def _require_type(feature: Feature, feature_type: FeatureType, values: str) -> None:
    if feature.graph.feature_type is not feature_type:
        msg = f"Feature {feature.name!r} is {feature.graph.feature_type.value}, so it does not accept {values}."
        raise RecourseOptionError(msg)


@attrs.frozen
class ValueRange:
    """The values of a continuous feature that a plan may move it to: from ``low`` to ``high``, both included.

    Attributes
    ----------
    low: float
        The lowest allowed value. The default has no lower limit.
    high: float
        The highest allowed value. The default has no upper limit.
    match: RangeMatch
        Which bins the range allows. ``RangeMatch.OVERLAP`` (the default) allows each bin with some values in the
        range, and ``RangeMatch.INSIDE`` only each bin with all of its values in the range. The text of a member,
        such as ``"inside"``, is accepted too.

    Examples
    --------
    >>> from ebaml import RangeMatch, ValueRange
    >>> ValueRange(high=40, match="inside").match is RangeMatch.INSIDE
    True

    """

    low: float = -math.inf
    high: float = math.inf
    match: RangeMatch = attrs.field(default=RangeMatch.OVERLAP, converter=_to_match)

    def __attrs_post_init__(self) -> None:
        """Reject a range whose low end is above its high end.

        Raises
        ------
        RecourseOptionError
            If ``low > high``.

        """
        if self.low > self.high:
            msg = f"A value range needs low <= high. Got low={self.low} and high={self.high}."
            raise RecourseOptionError(msg)

    def bins(self, feature: Feature) -> npt.NDArray[np.bool_]:
        """Mark the bins of a feature that the range allows.

        Parameters
        ----------
        feature: Feature
            A continuous feature.

        Returns
        -------
        npt.NDArray[np.bool_]
            For each bin, whether the range allows it. A feature that is not continuous raises a
            ``RecourseOptionError``.

        """
        _require_type(feature, FeatureType.CONTINUOUS, "a value range")
        allows = _MATCHES[self.match]
        # a continuous feature has an interval for each bin, so the type check above narrows the values
        bin_ranges = typing.cast("Sequence[Interval]", feature.graph.x_vals)
        # the graph holds numpy floats, and a comparison of numpy floats gives a numpy bool
        return np.array(
            [allows(self.low, self.high, (float(bin_low), float(bin_high))) for bin_low, bin_high in bin_ranges],
            dtype=bool,
        )

    def describe(self) -> str:
        """Put the range in words, for the facts.

        Returns
        -------
        str
            For example ``"from 18 to 40, bins that overlap the range"``.

        """
        return f"from {self.low:.4g} to {self.high:.4g}, bins that {_MATCH_WORDS[self.match]}"


_MATCH_WORDS: Mapping[RangeMatch, str] = types.MappingProxyType(
    {RangeMatch.OVERLAP: "overlap the range", RangeMatch.INSIDE: "lie fully inside the range"},
)


@attrs.frozen
class Levels:
    """The categories of a nominal or ordinal feature that a plan may move it to.

    Attributes
    ----------
    levels: tuple[str, ...]
        The allowed categories, as the graph of the feature names them, in alphabetical order. Any collection of
        texts is accepted.

    Examples
    --------
    >>> from ebaml import Levels
    >>> Levels(["rent", "own"]).levels
    ('own', 'rent')

    """

    levels: tuple[str, ...] = attrs.field(converter=_sorted_names)

    def bins(self, feature: Feature) -> npt.NDArray[np.bool_]:
        """Mark the bins of a feature that the levels allow.

        Parameters
        ----------
        feature: Feature
            A nominal or ordinal feature.

        Returns
        -------
        npt.NDArray[np.bool_]
            For each bin, whether its category is one of the levels.

        Raises
        ------
        RecourseOptionError
            If the feature is continuous, or a level is not a category of the feature.

        """
        if feature.graph.feature_type is FeatureType.CONTINUOUS:
            _require_type(feature, FeatureType.NOMINAL, "levels")
        categories = [bin_label(x_val) for x_val in feature.graph.x_vals]
        unknown = sorted(set(self.levels).difference(categories))
        if unknown:
            msg = f"Feature {feature.name!r} has no categories {unknown}. Its categories are {categories}."
            raise RecourseOptionError(msg)
        return np.isin(categories, list(self.levels))

    def describe(self) -> str:
        """Put the levels in words, for the facts.

        Returns
        -------
        str
            The categories, in alphabetical order.

        """
        return "one of " + ", ".join(self.levels)


type AllowedValues = ValueRange | Levels


@attrs.frozen
class RecourseOptions:
    """The limits that the counterfactual keeps to.

    Attributes
    ----------
    immutable: tuple[str, ...]
        The features that the counterfactual must not change, in alphabetical order. Any collection of names is
        accepted.
    allowed: Mapping[str, AllowedValues]
        For some features, the values that the counterfactual may move them to: a ``ValueRange`` for a continuous
        feature, and ``Levels`` for a nominal or ordinal one. An allowed value must also be plausible
        (``LocalOptions.min_support``). A feature that is in neither field may take every plausible value.
    costs: CostStrategy
        How hard each change is. ``UnitCost()`` (the default) makes the counterfactual the fewest changes. Another
        strategy (``DataCost``, ``UserCost``, ``ScaledCost``) makes it the cheapest plan, which the search of
        ``LocalOptions.search`` finds, because sorting is exact for equal costs only.
    max_changes: int | None
        The most features that the counterfactual may change, or ``None`` for no cap. A cheapest plan without a cap can
        change many cheap features.
    plans: int
        How many diverse plans to return, cheapest first. As in GAM Coach, each plan after the first uses no pair of a
        feature and a bin that an earlier plan uses, so it may move a feature of an earlier plan to another bin. The
        list is shorter when the search proves that no further plan exists.

    Examples
    --------
    >>> from ebaml import Levels, RecourseOptions, ValueRange
    >>> options = RecourseOptions(immutable=["age"], allowed={"income": ValueRange(high=60_000)})
    >>> options.immutable
    ('age',)

    """

    immutable: tuple[str, ...] = attrs.field(factory=tuple, converter=_sorted_names)
    # a copy, so that a change of the caller's mapping does not change the options; a dict, so that the options
    # serialize to JSON
    allowed: Mapping[str, AllowedValues] = attrs.field(factory=dict, converter=dict)
    costs: CostStrategy = attrs.field(factory=UnitCost)
    max_changes: int | None = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.ge(1)))
    plans: int = attrs.field(default=1, validator=attrs.validators.ge(1))

    def restrict(self, features: Sequence[Feature]) -> list[Feature]:
        """Limit the bins that each feature can end in.

        Parameters
        ----------
        features: Sequence[Feature]
            The features of the instance.

        Returns
        -------
        list[Feature]
            The same features. The field ``reachable`` of each one holds its allowed plausible bins and the bin of the
            instance. A ``RecourseOptionError`` tells that a name is not a feature of the model, that a feature is
            both immutable and in ``allowed``, that the values of a feature do not fit its type, or that they leave no
            plausible bin.

        """
        names = [feature.name for feature in features]
        self._check_names(names)
        return [self._restrict_one(feature) for feature in features]

    def _check_names(self, names: Collection[str]) -> None:
        unknown = sorted((set(self.immutable) | self.allowed.keys()) - set(names))
        if unknown:
            msg = f"The recourse options name features that the model does not have: {unknown}. Its features: {names}."
            raise RecourseOptionError(msg)
        both = sorted(set(self.immutable) & self.allowed.keys())
        if both:
            msg = f"These features are both immutable and in allowed, so the options disagree: {both}."
            raise RecourseOptionError(msg)

    def _restrict_one(self, feature: Feature) -> Feature:
        if feature.name in self.immutable:
            allowed = np.zeros_like(feature.plausible, dtype=bool)
        elif feature.name in self.allowed:
            allowed = self.allowed[feature.name].bins(feature) & feature.plausible
            if not allowed.any():
                msg = (
                    f"The allowed values of feature {feature.name!r} leave no plausible bin. Widen them, lower "
                    "min_support, or make the feature immutable."
                )
                raise RecourseOptionError(msg)
        else:
            return feature
        allowed[feature.bin_index] = True
        return attrs.evolve(feature, reachable=allowed)

    def describe(self) -> dict[str, str]:  # nosemgrep: ebaml-return-attrs-not-container -- the keys are feature names
        """Put the allowed values in words, for the facts.

        Returns
        -------
        dict[str, str]
            The words of the allowed values of each feature in ``allowed``, by name.

        """
        return {name: values.describe() for name, values in sorted(self.allowed.items())}
