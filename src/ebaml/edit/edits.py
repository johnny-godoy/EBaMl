"""The edits of a graph.

Each edit is a frozen class with one method, ``run``, that changes a copy of the model through ``model_write``. A new
edit is a new class. The docstring of each class names the tool of GAM Changer (Wang et al., KDD 2022, MIT licence)
that it follows.

No edit re-centres a graph, except ``Recenter``. A person who sets a score to 0.38 reads 0.38. The standard deviation
of an edited bin is 0, because a score that a person sets has no fitted uncertainty.

A multiclass classifier has one score per class in each bin. An edit of a graph then takes ``class_label``: the class
whose scores it changes, or ``None`` to change the scores of every class in the same way, one class at a time. Softmax
reads only the differences between the classes, so an edit of one class also changes the probabilities of the other
classes, and the same shift of every class changes no prediction. A binary classifier and a regressor have one score
in each bin, so their edits take no class.
"""

import abc
import enum
import functools
from collections.abc import Callable, Sequence
from typing import override

import attrs
import numpy as np
import numpy.typing as npt
from scipy.optimize import lsq_linear
from sklearn.isotonic import IsotonicRegression

from ebaml.core.custom_types import ClassLabel, Direction, ExplainableBoostingModel, coerce_enum
from ebaml.core.exceptions import EditError, SelectionError
from ebaml.edit import model_write
from ebaml.edit.model_view import TermView, class_views, term_view
from ebaml.edit.selections import Selection, WholeGraph


class AlignTarget(enum.StrEnum):
    """The value that ``Align`` gives to the selected scores."""

    LEFT = "left"
    RIGHT = "right"
    MEAN = "mean"


# a bin with no training weight still gets a tiny weight in a fit, so that the fit has no 0/0
_TINY_WEIGHT_SHARE = 1e-9
_LINE_MINIMUM_BINS = 2


def _number(value: float) -> str:
    return format(value, ".4g")


@attrs.frozen
class EditOutcome:
    """What an edit did to the copy of the model.

    ``feature`` is ``None`` for an edit that names terms and not one feature. ``positions`` holds the positions in
    ``term_scores_`` that the edit set, in increasing order. ``removed_terms`` names the terms that the edit removed.
    """

    feature: str | None
    positions: tuple[int, ...]
    description: str
    removed_terms: tuple[str, ...] = ()


class Edit(abc.ABC):
    """A change of one model, that a session can apply, undo, record and replay."""

    @abc.abstractmethod
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        """Change the model. The session passes a copy, and never the model of an earlier revision.

        Parameters
        ----------
        ebm : ExplainableBoostingModel
            The copy of the model to change.

        Returns
        -------
        EditOutcome
            The bins that the edit changed, and a description of the edit.

        Raises
        ------
        EditError
            If the edit does not fit the model.
        SelectionError
            If the bins of the edit do not exist.

        """


@attrs.frozen(eq=False)
class _Target:
    """The term of an edit, in one class column, and the positions that the edit changes."""

    view: TermView
    positions: tuple[int, ...]

    def write(
        self,
        ebm: ExplainableBoostingModel,
        values: npt.ArrayLike,
        positions: tuple[int, ...] | None = None,
    ) -> None:
        """Set the scores of the selected bins, or of ``positions``, in the column of the view."""
        chosen = self.positions if positions is None else positions
        model_write.set_scores(ebm, self.view.term, chosen, values, column=self.view.column)

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """The scores of the selected bins."""
        return self.view.scores[list(self.positions)]

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """The training weights of the selected bins."""
        return self.view.weights[list(self.positions)]

    @property
    def places(self) -> npt.NDArray[np.float64]:
        """The places of the selected real bins on the x axis."""
        return self.view.left_edges()[[position - 1 for position in self.positions]]


def _class_text(views: Sequence[TermView]) -> str:
    """Name the classes of an edit for its text: one class, every class, or none for a model with one score per bin.

    Returns
    -------
    str
        For example `` for the class 'low'`` or `` for each class``.

    """
    return " for each class" if len(views) > 1 else views[0].class_text()


@attrs.frozen(eq=False)
class _Targets:
    """The selected bins of an edit, in each class column that the edit changes."""

    columns: tuple[_Target, ...]

    @property
    def view(self) -> TermView:
        """The view of the first column, which has the bins and the feature type of every column."""
        return self.columns[0].view

    @property
    def positions(self) -> tuple[int, ...]:
        """The positions of the selected bins, which are the same in every column."""
        return self.columns[0].positions

    def write_each(self, ebm: ExplainableBoostingModel, new_scores: Callable[[_Target], npt.ArrayLike]) -> None:
        """Set the selected scores of each class column to the scores that a function gives for that column."""
        for column in self.columns:
            column.write(ebm, new_scores(column))

    def describe(self) -> str:
        """Describe the selected bins and the classes for the text of an edit.

        Returns
        -------
        str
            For example ``6 bins of age (95.5 to 105)``, or ``6 bins of age (95.5 to 105) for the class 'low'``.

        """
        selected = self.view.describe_positions(self.positions)
        return selected + _class_text([column.view for column in self.columns])


def _views(ebm: ExplainableBoostingModel, feature: str, class_label: ClassLabel | None) -> tuple[TermView, ...]:
    model_write.check_editable(ebm)
    return class_views(term_view(ebm, feature), class_label)


def _resolve(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    class_label: ClassLabel | None,
) -> _Targets:
    views = _views(ebm, feature, class_label)
    positions = selection.resolve(views[0])
    return _Targets(tuple(_Target(view, positions) for view in views))


def _resolve_ordered(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    class_label: ClassLabel | None,
    *,
    minimum_bins: int,
) -> _Targets:
    """Resolve a selection for an edit that needs an order: real bins of a continuous or ordinal feature.

    Returns
    -------
    _Targets
        The view of the term in each class column, and the positions of the bins.

    Raises
    ------
    EditError
        If the feature has no order, the selection holds the missing bin or the unseen bin, or it is too small.

    """
    target = _resolve(ebm, feature, selection, class_label)
    if not target.view.ordered:
        msg = f"This edit needs an order, but {feature!r} is {target.view.feature_type}. Use Shift, SetScores or Align."
        raise EditError(msg)
    if not target.view.is_real(target.positions):
        msg = f"This edit needs real bins, and the selection holds the missing bin or the unseen bin of {feature!r}."
        raise EditError(msg)
    if len(target.positions) < minimum_bins:
        msg = (
            f"This edit needs at least {minimum_bins} bins of {feature!r},"
            f" and the selection holds {len(target.positions)}."
        )
        raise EditError(msg)
    return target


def _resolve_line(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    class_label: ClassLabel | None,
) -> _Targets:
    return _resolve_ordered(ebm, feature, selection, class_label, minimum_bins=_LINE_MINIMUM_BINS)


def _fit_weights(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Give each bin a positive weight for a fit, and the same weight to every bin if no bin has weight.

    Returns
    -------
    npt.NDArray[np.float64]
        One weight for each bin.

    """
    largest = weights.max(initial=0.0)
    if largest <= 0:
        return np.ones_like(weights)
    return np.maximum(weights, largest * _TINY_WEIGHT_SHARE)


@attrs.frozen
class Shift(Edit):
    """Add one value to each selected score. GAM Changer: ``move``."""

    feature: str
    selection: Selection
    delta: float
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection, self.class_label)
        target.write_each(ebm, lambda column: column.scores + self.delta)
        text = f"Add {_number(self.delta)} to the scores of {target.describe()}"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class SetScores(Edit):
    """Set each selected score to one value. GAM Changer: ``align`` to a typed value, and ``delete`` with value 0."""

    feature: str
    selection: Selection
    value: float
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection, self.class_label)
        text = f"Set {target.describe()} to {_number(self.value)}"
        target.write_each(ebm, lambda _: self.value)
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class AssignScores(Edit):
    """Set each selected score to its own value. The values follow the order of the bins.

    GAM Changer has no such tool. ``load_gamchanger`` uses it to carry the scores of one GAM Changer commit, such as a
    monotone fit, into one edit.
    """

    feature: str
    selection: Selection
    values: tuple[float, ...] = attrs.field(converter=tuple)
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection, self.class_label)
        if len(self.values) != len(target.positions):
            msg = (
                f"AssignScores needs one value for each selected bin: the selection holds {len(target.positions)}"
                f" bins of {self.feature!r}, and the edit has {len(self.values)} values."
            )
            raise EditError(msg)
        for column in target.columns:
            column.write(ebm, np.asarray(self.values, dtype=float))
        return EditOutcome(self.feature, target.positions, f"Assign new scores to {target.describe()}")


@attrs.frozen
class Align(Edit):
    """Set each selected score to the score of the left bin, the score of the right bin, or the weighted mean.

    GAM Changer: ``align``. The bin on the left is the real bin before the first selected bin, and the bin on the
    right is the real bin after the last one. They need a feature with an order. The mean uses the training weight of
    the selected bins, or counts each bin once if no selected bin has weight. In a multiclass model, each class gets
    the value of its own scores.
    """

    feature: str
    selection: Selection
    to: AlignTarget = attrs.field(converter=functools.partial(coerce_enum, AlignTarget))
    class_label: ClassLabel | None = None

    def _value(self, target: _Target) -> float:
        view, positions = target.view, target.positions
        if self.to is AlignTarget.MEAN:
            weights = target.weights
            return float(np.average(target.scores, weights=weights if weights.sum() > 0 else None))
        if not view.ordered or not view.is_real(positions):
            msg = (
                f"Align to the {self.to} needs real bins of a feature with an order,"
                f" but {view.feature_name!r} has none in the selection."
            )
            raise EditError(msg)
        neighbour = {AlignTarget.LEFT: positions[0] - 1, AlignTarget.RIGHT: positions[-1] + 1}[self.to]
        if not view.is_real([neighbour]):
            msg = (
                f"The selection has no bin on the {self.to} in {view.feature_name!r}."
                " Select other bins, or use SetScores."
            )
            raise EditError(msg)
        return float(view.scores[neighbour])

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection, self.class_label)
        values = [self._value(column) for column in target.columns]
        for column, value in zip(target.columns, values, strict=True):
            column.write(ebm, value)
        text = f"Align {target.describe()} to the {self.to} ({', '.join(_number(value) for value in values)})"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen(eq=False)
class _MonotoneFit:
    """The direction of a monotone fit of one class column, and the fitted scores."""

    direction: Direction
    scores: npt.NDArray[np.float64]


@attrs.frozen
class Monotonize(Edit):
    """Make the selected scores monotone with an isotonic regression, weighted by the training weight of the bins.

    GAM Changer: ``increasing`` and ``decreasing``. The scores outside the selection stay as they are. The weighted
    isotonic fit keeps the weighted mean of the selected scores, so the weighted mean of the term does not change, and
    the edit adds no constant. ``interpret``'s ``monotonize`` gives the same scores on the whole graph.

    With ``direction="auto"``, the edit fits both directions and keeps the one with the smaller weighted sum of
    squared changes, and ``increasing`` if they tie. A bin with no training weight counts with a tiny weight. In a
    multiclass model, the edit fits each class on its own, and ``auto`` picks the direction of each class on its own.
    ``interpret``'s ``monotonize`` refuses a multiclass model, so this edit is the only way to monotonize one.
    """

    feature: str
    direction: Direction = attrs.field(converter=functools.partial(coerce_enum, Direction))
    selection: Selection = attrs.field(factory=WholeGraph)
    class_label: ClassLabel | None = None

    def _fit(self, target: _Target) -> "_MonotoneFit":
        """Fit the selected scores of one class column.

        Returns
        -------
        _MonotoneFit
            The direction that the fit took, and the fitted scores.

        """
        scores = target.scores
        weights = _fit_weights(target.weights)
        order = np.arange(len(target.positions), dtype=float)
        candidates = {
            direction: IsotonicRegression(increasing=direction is Direction.INCREASING).fit_transform(
                order,
                scores,
                sample_weight=weights,
            )
            for direction in (
                (Direction.INCREASING, Direction.DECREASING) if self.direction is Direction.AUTO else (self.direction,)
            )
        }
        direction = min(candidates, key=lambda name: float(np.sum(weights * (candidates[name] - scores) ** 2)))
        return _MonotoneFit(direction, candidates[direction])

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve_ordered(ebm, self.feature, self.selection, self.class_label, minimum_bins=1)
        fits = [self._fit(column) for column in target.columns]
        for column, fit in zip(target.columns, fits, strict=True):
            column.write(ebm, fit.scores)
        directions = list(dict.fromkeys(str(fit.direction) for fit in fits))
        text = f"Make {target.describe()} {' or '.join(directions)}"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class _LineEdit(Edit):
    """An edit that replaces the selected scores of each class by a line through them."""

    feature: str
    selection: Selection
    class_label: ClassLabel | None = None

    @abc.abstractmethod
    def line(self, column: _Target) -> npt.NDArray[np.float64]:
        """Give the new scores of the selected bins of one class column.

        Returns
        -------
        npt.NDArray[np.float64]
            One score for each selected bin.

        """

    @abc.abstractmethod
    def text(self, selected: str) -> str:
        """Describe the edit of the selected bins.

        Returns
        -------
        str
            The description of the edit.

        """

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve_line(ebm, self.feature, self.selection, self.class_label)
        target.write_each(ebm, self.line)
        return EditOutcome(self.feature, target.positions, self.text(target.describe()))


@attrs.frozen
class Interpolate(_LineEdit):
    """Replace the selected scores by a straight line between the first and the last selected score.

    GAM Changer: ``inplace-interpolate``. The place of a bin on the x axis is its left edge, and the place of an
    ordinal bin is its bin number. The bins stay as they are. The first and the last score do not change.
    """

    @override
    def line(self, column: _Target) -> npt.NDArray[np.float64]:
        places, scores = column.places, column.scores
        slope = (scores[-1] - scores[0]) / (places[-1] - places[0])
        line: npt.NDArray[np.float64] = scores[0] + slope * (places - places[0])
        return line

    @override
    def text(self, selected: str) -> str:
        return f"Interpolate {selected} with a straight line"


@attrs.frozen
class FitLine(_LineEdit):
    """Replace the selected scores by the weighted least-squares line through them.

    GAM Changer: ``inplace-regression``. The weights are the training weights of the bins, and the places on the x axis
    are the left edges of the bins, or the bin numbers of an ordinal feature. The bins stay as they are.
    """

    @override
    def line(self, column: _Target) -> npt.NDArray[np.float64]:
        places, scores = column.places, column.scores
        weights = _fit_weights(column.weights)
        mean_place = np.average(places, weights=weights)
        mean_score = np.average(scores, weights=weights)
        spread = float(np.sum(weights * (places - mean_place) ** 2))
        slope = float(np.sum(weights * (places - mean_place) * (scores - mean_score))) / spread
        return mean_score + slope * (places - mean_place)

    @override
    def text(self, selected: str) -> str:
        return f"Replace {selected} by a weighted least-squares line"


@attrs.frozen
class Clip(Edit):
    """Move each selected score that is below ``low`` or above ``high`` to the nearer bound.

    GAM Changer has no such tool. The constraint ``Bounded`` names it as its repair. A score in the range does not
    change, and its bin keeps its fitted uncertainty.
    """

    feature: str
    selection: Selection
    low: float
    high: float
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        if not self.low <= self.high:
            msg = f"Clip needs numbers with low not above high, but low is {self.low} and high is {self.high}."
            raise EditError(msg)
        target = _resolve(ebm, self.feature, self.selection, self.class_label)
        changed: set[int] = set()
        for column in target.columns:
            scores = column.scores
            outside = (scores < self.low) | (scores > self.high)
            positions = tuple(np.asarray(column.positions)[outside].tolist())
            if positions:
                column.write(ebm, np.clip(scores[outside], self.low, self.high), positions)
            changed.update(positions)
        positions = tuple(sorted(changed))
        text = f"Clip {target.describe()} to the range {_number(self.low)} to {_number(self.high)}"
        return EditOutcome(self.feature, positions, text)


@attrs.frozen
class LimitSteps(Edit):
    """Replace the selected scores by a weighted least-squares fit with no step above ``step``.

    GAM Changer has no such tool. The constraint ``MaxStep`` names it as its repair. A step is the difference between
    the scores of two neighbour bins. The weights are the training weights of the bins, as in ``Monotonize``, and a bin
    with no training weight counts with a tiny weight. The fit uses the bounded least-squares solver of SciPy
    (``lsq_linear`` with ``bvls``), so it holds to the precision of that solver. The scores outside the selection stay.
    """

    feature: str
    step: float
    selection: Selection = attrs.field(factory=WholeGraph)
    class_label: ClassLabel | None = None

    def _fit(self, scores: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.step == 0:
            # the solver needs a low bound below the high bound, and a fit with no step is the weighted mean
            return np.full_like(scores, np.average(scores, weights=weights))
        root_weights = np.sqrt(weights)
        bin_count = len(scores)
        # the unknowns are the first score and each step, and the score of a bin is the first score plus the steps
        # before it, so the score of every bin is a row of a lower triangle of ones times the unknowns
        cumulative = np.tril(np.ones((bin_count, bin_count)))
        lower_bounds = np.full(bin_count, -self.step)
        upper_bounds = np.full(bin_count, self.step)
        lower_bounds[0], upper_bounds[0] = -np.inf, np.inf
        fit = lsq_linear(
            cumulative * root_weights[:, None],
            scores * root_weights,
            bounds=(lower_bounds, upper_bounds),
            method="bvls",
        )
        solution = np.asarray(fit.x, dtype=float)
        steps = np.clip(solution[1:], -self.step, self.step)
        offsets: npt.NDArray[np.float64] = np.concatenate(([0.0], np.cumsum(steps)))
        return offsets + float(solution[0])

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        if not 0 <= self.step < np.inf:
            msg = f"LimitSteps needs a step that is 0 or more and finite, but the step is {self.step}."
            raise EditError(msg)
        target = _resolve_line(ebm, self.feature, self.selection, self.class_label)
        for column in target.columns:
            column.write(ebm, self._fit(column.scores, _fit_weights(column.weights)))
        text = f"Limit each step of {target.describe()} to {_number(self.step)}"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class Scale(Edit):
    """Multiply every score of a graph by a factor. The intercept stays. ``interpret`` has the same method."""

    feature: str
    factor: float
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        views = _views(ebm, self.feature, self.class_label)
        for view in views:
            model_write.scale_term(ebm, view.term, self.factor, column=view.column)
        every_position = tuple(range(views[0].real_bin_count + 2))
        text = f"Multiply the graph of {self.feature}{_class_text(views)} by {_number(self.factor)}"
        return EditOutcome(self.feature, every_position, text)


@attrs.frozen
class ZeroTerm(Edit):
    """Set every score of a graph to 0, the missing bin and the unseen bin too. The term stays in the model.

    GAM Changer: ``delete`` on all bins. Use it for "the model must not use this feature". ``RemoveFeature`` removes
    the terms instead. In a multiclass model, a graph whose scores are 0 for every class gives every class the same
    score, so the feature changes no probability.
    """

    feature: str
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        views = _views(ebm, self.feature, self.class_label)
        every_position = tuple(range(views[0].real_bin_count + 2))
        for view in views:
            model_write.set_scores(ebm, view.term, every_position, 0.0, column=view.column)
        text = f"Set every score of {self.feature}{_class_text(views)} to 0"
        return EditOutcome(self.feature, every_position, text)


@attrs.frozen
class Recenter(Edit):
    """Move the weighted mean of a graph into the intercept. No prediction changes, except by rounding.

    The weights are the training weights of the bins, the missing bin and the unseen bin included. After the edit the
    weighted mean of the graph is 0. An edit that sets scores moves this mean, and ``Recenter`` puts it back where
    the fitted model had it. In a multiclass model, the mean of each class moves into the intercept of that class.
    """

    feature: str
    class_label: ClassLabel | None = None

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        views = _views(ebm, self.feature, self.class_label)
        weights = views[0].weights
        if weights.sum() <= 0:
            msg = f"No bin of {self.feature!r} has training weight, so the weighted mean of its graph is not defined."
            raise EditError(msg)
        means = [float(np.average(view.scores, weights=weights)) for view in views]
        for view, mean in zip(views, means, strict=True):
            model_write.shift_term(ebm, view.term, -mean, intercept_shift=mean, column=view.column)
        every_position = tuple(range(views[0].real_bin_count + 2))
        mean_text = ", ".join(_number(mean) for mean in means)
        text = f"Move the weighted mean of {self.feature}{_class_text(views)} ({mean_text}) into the intercept"
        return EditOutcome(self.feature, every_position, text)


def _terms_text(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


@attrs.frozen
class RemoveTerms(Edit):
    """Remove terms from the model, by term name. The model keeps its input columns.

    A pair term has a name such as ``age & sex``. ``interpret`` removes each term, and the other terms stay.
    """

    terms: tuple[str, ...] = attrs.field(converter=tuple)

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        model_write.check_editable(ebm)
        known = list(ebm.term_names_)
        if not self.terms:
            msg = "RemoveTerms needs at least one term."
            raise SelectionError(msg)
        for name in self.terms:
            if name not in known:
                msg = f"The model has no term {name!r}. The terms are {known}."
                raise SelectionError(msg)
        model_write.remove_terms(ebm, self.terms)
        return EditOutcome(None, (), f"Remove the terms {_terms_text(self.terms)}", self.terms)


@attrs.frozen
class RemoveFeature(Edit):
    """Remove every term that uses a feature, the pair terms included. The model keeps its input columns."""

    feature: str

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        model_write.check_editable(ebm)
        names = list(ebm.feature_names_in_)
        if self.feature not in names:
            msg = f"The model has no feature {self.feature!r}. The features are {names}."
            raise SelectionError(msg)
        feature_index = names.index(self.feature)
        terms = tuple(
            name
            for name, features in zip(ebm.term_names_, ebm.term_features_, strict=True)
            if feature_index in features
        )
        if not terms:
            msg = f"No term of the model uses the feature {self.feature!r}."
            raise SelectionError(msg)
        model_write.remove_terms(ebm, terms)
        return EditOutcome(self.feature, (), f"Remove every term of {self.feature}: {_terms_text(terms)}", terms)
