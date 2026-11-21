"""Business constraints on the graphs of an EBM, and the check that finds the constraints that a model breaks.

A constraint states a business requirement, for example "the risk must not fall when the age goes up" or "a missing
value scores 0". Each constraint is a frozen class with two methods: ``violation`` reads the model and returns what
breaks the constraint, and ``repair`` names the edit that makes the constraint hold. A new constraint is a new class.

A constraint reads the scores of the main effect of a feature. A pair term that uses the feature can still change the
effect of the feature on a prediction, so only ``Unused`` makes a claim about the pair terms. A score difference at or
below the tolerance is not a violation, so that the rounding of a repair does not count.

A multiclass model has one score per class in each bin. A constraint on a graph then takes ``class_label``: the class
whose scores must meet it, or ``None`` for every class. The repair changes the same classes.
"""

import abc
import functools
from collections.abc import Iterable, Sequence
from typing import override

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ClassLabel, ExplainableBoostingModel, MonotoneDirection, coerce_enum
from ebaml.core.exceptions import EditError, OptionRangeError, SelectionError
from ebaml.edit import model_write
from ebaml.edit.edits import Clip, Edit, LimitSteps, Monotonize, RemoveFeature, SetScores
from ebaml.edit.model_view import TermView, class_views, term_view
from ebaml.edit.selections import EveryBin, Selection, WholeGraph

DEFAULT_TOLERANCE = 1e-9


def _number(value: float) -> str:
    return format(value, ".4g")


class Constraint(abc.ABC):
    """A business requirement on the graph of one feature."""

    @abc.abstractmethod
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> "Violation | None":
        """Find what breaks the constraint in a model.

        Parameters
        ----------
        ebm : ExplainableBoostingModel
            The model to check. The check reads the model and changes nothing.
        tolerance : float
            The largest difference that does not count as a violation.

        Returns
        -------
        Violation | None
            The violation, or ``None`` if the model meets the constraint.

        Raises
        ------
        EditError
            If the constraint does not fit the model.
        SelectionError
            If the feature or the bins of the constraint do not exist.

        """

    @abc.abstractmethod
    def repair(self) -> Edit:
        """Name the edit that makes the constraint hold.

        Returns
        -------
        Edit
            The edit.

        """


@attrs.frozen
class Violation:
    """What breaks one constraint in one model.

    Attributes
    ----------
    constraint: Constraint
        The constraint that the model breaks.
    description: str
        What breaks the constraint, in one sentence.
    bins: tuple[str, ...]
        The bins that break the constraint, as the report of an edit names them: ``missing``, ``unseen``, the edges of a
        continuous bin, or a category.
    terms: tuple[str, ...]
        The terms that break the constraint. Only ``Unused`` fills it.
    excess: float
        The largest amount by which one score, or one step between two scores, breaks the constraint.
    repair: Edit
        The edit that makes the constraint hold. Preview it before you apply it, because it changes predictions.

    """

    constraint: Constraint
    description: str
    bins: tuple[str, ...]
    excess: float
    repair: Edit
    terms: tuple[str, ...] = ()


@attrs.frozen(eq=False)
class _Scores:
    """The selected scores of the main effect of one feature, in each class column that the constraint reads."""

    views: tuple[TermView, ...]
    positions: tuple[int, ...]

    @property
    def view(self) -> TermView:
        """The view of the first column, which has the bins and the feature type of every column."""
        return self.views[0]

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The selected scores: one row for each class column, and one value for each selected bin."""
        return np.array([view.scores[list(self.positions)] for view in self.views], dtype=float)

    def labels(self, mask: npt.NDArray[np.bool_]) -> tuple[str, ...]:
        """Name the bins where a mask is true in one class column or more.

        Returns
        -------
        tuple[str, ...]
            The name of each such bin, in the order of the bins.

        """
        in_any = np.asarray(mask).reshape(-1, len(self.positions)).any(axis=0)
        return tuple(
            self.view.label(position) for position, chosen in zip(self.positions, in_any, strict=True) if chosen
        )

    def pair_labels(self, mask: npt.NDArray[np.bool_]) -> tuple[str, ...]:
        """Name both bins of each pair of neighbour bins where a mask is true.

        Returns
        -------
        tuple[str, ...]
            The name of each such bin, in the order of the bins, with no repeat.

        """
        pairs = np.asarray(mask).reshape(-1, len(self.positions) - 1).any(axis=0)
        in_pair = np.zeros(len(self.positions), dtype=bool)
        in_pair[:-1] |= pairs
        in_pair[1:] |= pairs
        return self.labels(in_pair)

    def describe(self) -> str:
        """Describe the selected bins and the classes.

        Returns
        -------
        str
            For example ``6 bins of age (95.5 to 105)``, or the same text with `` for the class 'low'``.

        """
        return self.view.describe_positions(self.positions) + self.class_text()

    def class_text(self) -> str:
        """Name the classes that the constraint reads.

        Returns
        -------
        str
            `` for each class``, `` for the class 'low'``, or an empty text for a model with one score per bin.

        """
        return " for each class" if len(self.views) > 1 else self.view.class_text()


def _scores(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    class_label: ClassLabel | None,
) -> _Scores:
    views = class_views(term_view(ebm, feature), class_label)
    return _Scores(views, selection.resolve(views[0]))


def _ordered_scores(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    class_label: ClassLabel | None,
    name: str,
) -> _Scores:
    """Read the selected scores for a constraint on the order of the bins.

    Returns
    -------
    _Scores
        The real bins of a continuous or ordinal feature.

    Raises
    ------
    EditError
        If the feature has no order, or the selection holds the missing bin or the unseen bin.

    """
    scores = _scores(ebm, feature, selection, class_label)
    if not scores.view.ordered:
        msg = f"{name} needs an order, but {feature!r} is {scores.view.feature_type}. Use Bounded or FixedScore."
        raise EditError(msg)
    if not scores.view.is_real(scores.positions):
        msg = f"{name} needs real bins, and the selection holds the missing bin or the unseen bin of {feature!r}."
        raise EditError(msg)
    return scores


@attrs.frozen
class Monotone(Constraint):
    """The selected scores must not fall (``increasing``) or must not rise (``decreasing``) from bin to bin.

    Select a range for a graph that must rise on one side and fall on the other, such as a U-shape. The repair is
    ``Monotonize`` on the same bins.
    """

    feature: str
    direction: MonotoneDirection = attrs.field(converter=functools.partial(coerce_enum, MonotoneDirection))
    selection: Selection = attrs.field(factory=WholeGraph)
    class_label: ClassLabel | None = None

    @override
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> Violation | None:
        scores = _ordered_scores(ebm, self.feature, self.selection, self.class_label, "Monotone")
        steps = np.diff(scores.values, axis=1)
        wrong_way = -steps if self.direction is MonotoneDirection.INCREASING else steps
        breaks = wrong_way > tolerance
        if not breaks.any():
            return None
        verb = "falls" if self.direction is MonotoneDirection.INCREASING else "rises"
        excess = float(wrong_way.max())
        description = (
            f"The graph of {self.feature} {verb} between {int(breaks.sum())} pairs of neighbour bins"
            f" in {scores.describe()}, by up to {_number(excess)}"
        )
        return Violation(self, description, scores.pair_labels(breaks), excess, self.repair())

    @override
    def repair(self) -> Edit:
        return Monotonize(self.feature, self.direction, self.selection, self.class_label)


@attrs.frozen
class MaxStep(Constraint):
    """No step between the scores of two neighbour bins in the selection may be larger than ``step``.

    The rule stops a jump of the graph that the data does not explain. The repair is ``LimitSteps`` on the same bins.
    """

    feature: str
    step: float
    selection: Selection = attrs.field(factory=WholeGraph)
    class_label: ClassLabel | None = None

    @override
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> Violation | None:
        if not 0 <= self.step < np.inf:
            msg = f"MaxStep needs a step that is 0 or more and finite, but the step is {self.step}."
            raise EditError(msg)
        scores = _ordered_scores(ebm, self.feature, self.selection, self.class_label, "MaxStep")
        over = np.abs(np.diff(scores.values, axis=1)) - self.step
        breaks = over > tolerance
        if not breaks.any():
            return None
        excess = float(over.max())
        description = (
            f"{int(breaks.sum())} steps of the graph of {self.feature} in {scores.describe()} are larger than"
            f" {_number(self.step)}, by up to {_number(excess)}"
        )
        return Violation(self, description, scores.pair_labels(breaks), excess, self.repair())

    @override
    def repair(self) -> Edit:
        return LimitSteps(self.feature, self.step, self.selection, self.class_label)


@attrs.frozen
class Bounded(Constraint):
    """Each selected score must be between ``low`` and ``high``, both included.

    The default selection is every bin, the missing bin and the unseen bin included, because a missing value is scored
    too. The repair is ``Clip`` on the same bins. A score of the graph is not a prediction: the intercept and the other
    terms add to it.
    """

    feature: str
    low: float
    high: float
    selection: Selection = attrs.field(factory=EveryBin)
    class_label: ClassLabel | None = None

    @override
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> Violation | None:
        if not self.low <= self.high:
            msg = f"Bounded needs numbers with low not above high, but low is {self.low} and high is {self.high}."
            raise EditError(msg)
        scores = _scores(ebm, self.feature, self.selection, self.class_label)
        values = scores.values
        outside = np.maximum(self.low - values, values - self.high)
        breaks = outside > tolerance
        if not breaks.any():
            return None
        excess = float(outside.max())
        description = (
            f"{int(breaks.sum())} scores of {self.feature}{scores.class_text()} are outside the range"
            f" {_number(self.low)} to {_number(self.high)}, by up to {_number(excess)}"
        )
        return Violation(self, description, scores.labels(breaks), excess, self.repair())

    @override
    def repair(self) -> Edit:
        return Clip(self.feature, self.selection, self.low, self.high, self.class_label)


@attrs.frozen
class FixedScore(Constraint):
    """Each selected score must equal ``value``, for example "a missing value scores 0". The repair is ``SetScores``."""

    feature: str
    selection: Selection
    value: float
    class_label: ClassLabel | None = None

    @override
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> Violation | None:
        scores = _scores(ebm, self.feature, self.selection, self.class_label)
        difference = np.abs(scores.values - self.value)
        breaks = difference > tolerance
        if not breaks.any():
            return None
        excess = float(difference.max())
        count = int(breaks.sum())
        description = (
            f"{count} scores of {self.feature}{scores.class_text()} differ from {_number(self.value)},"
            f" by up to {_number(excess)}"
        )
        return Violation(self, description, scores.labels(breaks), excess, self.repair())

    @override
    def repair(self) -> Edit:
        return SetScores(self.feature, self.selection, self.value, self.class_label)


@attrs.frozen
class Unused(Constraint):
    """No term of the model may use the feature with a score other than 0, the pair terms included.

    A term whose scores are all 0 meets the constraint, so ``ZeroTerm`` meets it for a feature with no pair term. The
    repair is ``RemoveFeature``, which removes every term that uses the feature. The model keeps its input columns.
    """

    feature: str

    def _terms(self, ebm: ExplainableBoostingModel) -> Sequence[int]:
        names = list(ebm.feature_names_in_)
        if self.feature not in names:
            msg = f"The model has no feature {self.feature!r}. The features are {names}."
            raise SelectionError(msg)
        feature_index = names.index(self.feature)
        return [term for term, features in enumerate(ebm.term_features_) if feature_index in features]

    @override
    def violation(self, ebm: ExplainableBoostingModel, tolerance: float) -> Violation | None:
        largest = {
            ebm.term_names_[term]: float(np.abs(np.asarray(ebm.term_scores_[term], dtype=float)).max())
            for term in self._terms(ebm)
        }
        breaking = {name: score for name, score in largest.items() if score > tolerance}
        if not breaking:
            return None
        terms = tuple(breaking)
        excess = max(breaking.values())
        description = (
            f"{len(terms)} terms of the model use {self.feature} with a score other than 0, by up to {_number(excess)}"
        )
        return Violation(self, description, (), excess, self.repair(), terms)

    @override
    def repair(self) -> Edit:
        return RemoveFeature(self.feature)


def check_constraints(
    ebm: ExplainableBoostingModel,
    constraints: Iterable[Constraint],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Violation]:
    """Find the constraints that a model breaks.

    The check reads the model and changes nothing. Apply the ``repair`` of a violation in an ``EditSession`` to make the
    constraint hold, and check again: a repair of one constraint can break another one.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The model to check, for example ``session.model``.
    constraints : Iterable[Constraint]
        The business requirements.
    tolerance : float
        The largest difference that does not count as a violation. It is 0 or more.

    Returns
    -------
    list[Violation]
        One violation for each constraint that the model breaks, in the order of the constraints.

    Raises
    ------
    OptionRangeError
        If the tolerance is negative or not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import Bounded, check_constraints
    >>> random_generator = np.random.default_rng(0)
    >>> ages = random_generator.integers(1, 9, size=(300, 1)) * 10.0
    >>> target = (ages[:, 0] + random_generator.normal(0, 20, 300) > 40).astype(int)
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=2, feature_names=["age"]).fit(ages, target)
    >>> check_constraints(ebm, [Bounded("age", -100.0, 100.0)])
    []
    >>> (violation,) = check_constraints(ebm, [Bounded("age", 0.0, 0.0)])
    >>> violation.repair
    Clip(feature='age', selection=EveryBin(), low=0.0, high=0.0, class_label=None)

    """
    if not 0 <= tolerance < np.inf:
        msg = f"The tolerance must be 0 or more and finite, but it is {tolerance}."
        raise OptionRangeError(msg)
    model_write.check_editable(ebm)
    return [violation for constraint in constraints if (violation := constraint.violation(ebm, tolerance)) is not None]
