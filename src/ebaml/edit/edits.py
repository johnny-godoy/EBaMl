"""The edits of a graph.

Each edit is a frozen class with one method, ``run``, that changes a copy of the model through ``model_write``. A new
edit is a new class. The docstring of each class names the tool of GAM Changer (Wang et al., KDD 2022, MIT licence)
that it follows.

No edit re-centres a graph, except ``Recenter``. A person who sets a score to 0.38 reads 0.38. The standard deviation
of an edited bin is 0, because a score that a person sets has no fitted uncertainty.
"""

import abc
from collections.abc import Sequence
from typing import Literal, override

import attrs
import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EditError, SelectionError
from ebaml.edit import model_write
from ebaml.edit.model_view import TermView, term_view
from ebaml.edit.selections import Selection, WholeGraph

Direction = Literal["increasing", "decreasing", "auto"]
AlignTarget = Literal["left", "right", "mean"]

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
    """The term of an edit and the positions that the edit changes."""

    view: TermView
    positions: tuple[int, ...]

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

    def describe(self) -> str:
        """Describe the selected bins for the text of an edit.

        Returns
        -------
        str
            For example ``6 bins of age (95.5 to 105)``.

        """
        return self.view.describe_positions(self.positions)


def _resolve(ebm: ExplainableBoostingModel, feature: str, selection: Selection) -> _Target:
    model_write.check_editable(ebm)
    view = term_view(ebm, feature)
    return _Target(view, selection.resolve(view))


def _resolve_ordered(
    ebm: ExplainableBoostingModel,
    feature: str,
    selection: Selection,
    *,
    minimum_bins: int,
) -> _Target:
    """Resolve a selection for an edit that needs an order: real bins of a continuous or ordinal feature.

    Returns
    -------
    _Target
        The view of the term, and the positions of the bins.

    Raises
    ------
    EditError
        If the feature has no order, the selection holds the missing bin or the unseen bin, or it is too small.

    """
    target = _resolve(ebm, feature, selection)
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


def _resolve_line(ebm: ExplainableBoostingModel, feature: str, selection: Selection) -> _Target:
    return _resolve_ordered(ebm, feature, selection, minimum_bins=_LINE_MINIMUM_BINS)


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

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection)
        model_write.set_scores(ebm, target.view.term, target.positions, target.scores + self.delta)
        text = f"Add {_number(self.delta)} to the scores of {target.describe()}"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class SetScores(Edit):
    """Set each selected score to one value. GAM Changer: ``align`` to a typed value, and ``delete`` with value 0."""

    feature: str
    selection: Selection
    value: float

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection)
        model_write.set_scores(ebm, target.view.term, target.positions, self.value)
        return EditOutcome(self.feature, target.positions, f"Set {target.describe()} to {_number(self.value)}")


@attrs.frozen
class Align(Edit):
    """Set each selected score to the score of the left bin, the score of the right bin, or the weighted mean.

    GAM Changer: ``align``. The bin on the left is the real bin before the first selected bin, and the bin on the
    right is the real bin after the last one. They need a feature with an order. The mean uses the training weight of
    the selected bins, or counts each bin once if no selected bin has weight.
    """

    feature: str
    selection: Selection
    to: AlignTarget

    def _value(self, target: _Target) -> float:
        view, positions = target.view, target.positions
        if self.to == "mean":
            weights = target.weights
            return float(np.average(target.scores, weights=weights if weights.sum() > 0 else None))
        if not view.ordered or not view.is_real(positions):
            msg = (
                f"Align to the {self.to} needs real bins of a feature with an order,"
                f" but {view.feature_name!r} has none in the selection."
            )
            raise EditError(msg)
        neighbour = positions[0] - 1 if self.to == "left" else positions[-1] + 1
        if not view.is_real([neighbour]):
            msg = (
                f"The selection has no bin on the {self.to} in {view.feature_name!r}."
                " Select other bins, or use SetScores."
            )
            raise EditError(msg)
        return float(view.scores[neighbour])

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve(ebm, self.feature, self.selection)
        value = self._value(target)
        model_write.set_scores(ebm, target.view.term, target.positions, value)
        text = f"Align {target.describe()} to the {self.to} ({_number(value)})"
        return EditOutcome(self.feature, target.positions, text)


@attrs.frozen
class Monotonize(Edit):
    """Make the selected scores monotone with an isotonic regression, weighted by the training weight of the bins.

    GAM Changer: ``increasing`` and ``decreasing``. The scores outside the selection stay as they are. The weighted
    isotonic fit keeps the weighted mean of the selected scores, so the weighted mean of the term does not change, and
    the edit adds no constant. ``interpret``'s ``monotonize`` gives the same scores on the whole graph.

    With ``direction="auto"``, the edit fits both directions and keeps the one with the smaller weighted sum of
    squared changes, and ``increasing`` if they tie. A bin with no training weight counts with a tiny weight.
    """

    feature: str
    direction: Direction
    selection: Selection = attrs.field(factory=WholeGraph)

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve_ordered(ebm, self.feature, self.selection, minimum_bins=1)
        scores = target.scores
        weights = _fit_weights(target.weights)
        order = np.arange(len(target.positions), dtype=float)
        candidates = {
            direction: IsotonicRegression(increasing=direction == "increasing").fit_transform(
                order,
                scores,
                sample_weight=weights,
            )
            for direction in (("increasing", "decreasing") if self.direction == "auto" else (self.direction,))
        }
        direction = min(candidates, key=lambda name: float(np.sum(weights * (candidates[name] - scores) ** 2)))
        model_write.set_scores(ebm, target.view.term, target.positions, candidates[direction])
        return EditOutcome(self.feature, target.positions, f"Make {target.describe()} {direction}")


@attrs.frozen
class Interpolate(Edit):
    """Replace the selected scores by a straight line between the first and the last selected score.

    GAM Changer: ``inplace-interpolate``. The place of a bin on the x axis is its left edge, and the place of an
    ordinal bin is its bin number. The bins stay as they are. The first and the last score do not change.
    """

    feature: str
    selection: Selection

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve_line(ebm, self.feature, self.selection)
        places, scores = target.places, target.scores
        slope = (scores[-1] - scores[0]) / (places[-1] - places[0])
        model_write.set_scores(ebm, target.view.term, target.positions, scores[0] + slope * (places - places[0]))
        return EditOutcome(self.feature, target.positions, f"Interpolate {target.describe()} with a straight line")


@attrs.frozen
class FitLine(Edit):
    """Replace the selected scores by the weighted least-squares line through them.

    GAM Changer: ``inplace-regression``. The weights are the training weights of the bins, and the places on the x axis
    are the left edges of the bins, or the bin numbers of an ordinal feature. The bins stay as they are.
    """

    feature: str
    selection: Selection

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        target = _resolve_line(ebm, self.feature, self.selection)
        places, scores = target.places, target.scores
        weights = _fit_weights(target.weights)
        mean_place = np.average(places, weights=weights)
        mean_score = np.average(scores, weights=weights)
        spread = float(np.sum(weights * (places - mean_place) ** 2))
        slope = float(np.sum(weights * (places - mean_place) * (scores - mean_score))) / spread
        model_write.set_scores(ebm, target.view.term, target.positions, mean_score + slope * (places - mean_place))
        return EditOutcome(
            self.feature,
            target.positions,
            f"Replace {target.describe()} by a weighted least-squares line",
        )


@attrs.frozen
class Scale(Edit):
    """Multiply every score of a graph by a factor. The intercept stays. ``interpret`` has the same method."""

    feature: str
    factor: float

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        model_write.check_editable(ebm)
        view = term_view(ebm, self.feature)
        model_write.scale_term(ebm, view.term, self.factor)
        every_position = tuple(range(view.real_bin_count + 2))
        text = f"Multiply the graph of {self.feature} by {_number(self.factor)}"
        return EditOutcome(self.feature, every_position, text)


@attrs.frozen
class ZeroTerm(Edit):
    """Set every score of a graph to 0, the missing bin and the unseen bin too. The term stays in the model.

    GAM Changer: ``delete`` on all bins. Use it for "the model must not use this feature". ``RemoveFeature`` removes
    the terms instead.
    """

    feature: str

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        model_write.check_editable(ebm)
        view = term_view(ebm, self.feature)
        every_position = tuple(range(view.real_bin_count + 2))
        model_write.set_scores(ebm, view.term, every_position, 0.0)
        return EditOutcome(self.feature, every_position, f"Set every score of {self.feature} to 0")


@attrs.frozen
class Recenter(Edit):
    """Move the weighted mean of a graph into the intercept. No prediction changes, except by rounding.

    The weights are the training weights of the bins, the missing bin and the unseen bin included. After the edit the
    weighted mean of the graph is 0. An edit that sets scores moves this mean, and ``Recenter`` puts it back where
    the fitted model had it.
    """

    feature: str

    @override
    def run(self, ebm: ExplainableBoostingModel) -> EditOutcome:
        model_write.check_editable(ebm)
        view = term_view(ebm, self.feature)
        weights = view.weights
        if weights.sum() <= 0:
            msg = f"No bin of {self.feature!r} has training weight, so the weighted mean of its graph is not defined."
            raise EditError(msg)
        mean = float(np.average(view.scores, weights=weights))
        model_write.shift_term(ebm, view.term, -mean, intercept_shift=mean)
        every_position = tuple(range(view.real_bin_count + 2))
        text = f"Move the weighted mean of {self.feature} ({_number(mean)}) into the intercept"
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
