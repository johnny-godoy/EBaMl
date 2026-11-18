"""The only functions that write the fitted attributes of an EBM.

An edited model must work wherever a fitted model works: ``predict``, ``predict_with_uncertainty``, ``explain_global``,
``EBMGraph.from_ebm`` and a save with ``skops``. So every edit writes through these functions, and no edit writes an
attribute itself. The functions keep three facts true:

- ``bagged_scores_`` holds the edit, so ``predict_with_uncertainty`` gives the edited prediction.
- A score that a person sets has no fitted uncertainty, so its standard deviation is 0.
- No attribute holds ``None``. ``interpret`` leaves ``None`` after ``monotonize``, and ``repair_uncertainty`` fills it.

An edit that changes the bins of a feature (``replace_bins``) changes the cuts, the scores, the bags, the standard
deviations and the training weights of the main effect in one step, so their shapes always agree.
"""

from collections.abc import Sequence
from typing import cast

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EditError, UnsupportedModelError
from ebaml.graphs.model_kinds import model_kind


def check_editable(ebm: ExplainableBoostingModel) -> None:
    """Check that the model is one that the edits support.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The fitted EBM.

    Raises
    ------
    UnsupportedModelError
        If the model is a multiclass classifier.

    """
    if len(model_kind(ebm).outputs) > 1:
        msg = (
            "Editing supports binary classifiers and regressors. A multiclass classifier has one score for each class."
        )
        raise UnsupportedModelError(msg)


def _bag_count(ebm: ExplainableBoostingModel) -> int:
    for bags in ebm.bagged_scores_:
        if bags is not None:
            return int(np.asarray(bags).shape[0])
    if ebm.bagged_intercept_ is not None:
        return int(np.asarray(ebm.bagged_intercept_).shape[0])
    return 1


def repair_uncertainty(ebm: ExplainableBoostingModel) -> None:
    """Fill the attributes that ``interpret`` leaves as ``None`` after ``monotonize``.

    The repaired term gets one copy of its scores in each bag and a standard deviation of 0. The fitted uncertainty of
    that term is lost, because ``interpret`` removed it.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to repair, which the function changes.

    """
    bag_count = _bag_count(ebm)
    for term, scores in enumerate(ebm.term_scores_):
        if ebm.bagged_scores_[term] is None:
            ebm.bagged_scores_[term] = np.tile(np.asarray(scores, dtype=float), (bag_count, 1))
        if ebm.standard_deviations_[term] is None:
            ebm.standard_deviations_[term] = np.zeros_like(np.asarray(scores, dtype=float))
    if ebm.bagged_intercept_ is None:
        ebm.bagged_intercept_ = np.full(bag_count, float(np.ravel(ebm.intercept_)[0]))


def _check_finite(values: npt.ArrayLike) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        msg = "An edit must give finite scores, but it gave a value that is not a finite number."
        raise EditError(msg)


def _shift_intercept(ebm: ExplainableBoostingModel, intercept_shift: float) -> None:
    if intercept_shift:
        # interpret types intercept_ as a float on a regressor and as an array on a classifier
        vars(ebm)["intercept_"] = ebm.intercept_ + intercept_shift
        ebm.bagged_intercept_ += intercept_shift


def set_scores(ebm: ExplainableBoostingModel, term: int, positions: Sequence[int], values: npt.ArrayLike) -> None:
    """Set the scores of some bins of a term, in every bag, with a standard deviation of 0.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    term : int
        The index of the term.
    positions : Sequence[int]
        The positions of the bins in ``term_scores_``.
    values : npt.ArrayLike
        The new score of each bin, or one score for all of them.

    """
    _check_finite(values)
    repair_uncertainty(ebm)
    indices = list(positions)
    scores = np.array(ebm.term_scores_[term], dtype=float)
    bags = np.array(ebm.bagged_scores_[term], dtype=float)
    deviations = np.array(ebm.standard_deviations_[term], dtype=float)
    scores[indices] = values
    bags[:, indices] = scores[indices]
    deviations[indices] = 0.0
    ebm.term_scores_[term] = scores
    ebm.bagged_scores_[term] = bags
    ebm.standard_deviations_[term] = deviations


def shift_term(ebm: ExplainableBoostingModel, term: int, delta: float, *, intercept_shift: float) -> None:
    """Add one value to every score of a term, and the opposite value to the intercept if the caller asks.

    The bags move with the scores, so the standard deviations stay as they are.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    term : int
        The index of the term.
    delta : float
        The value to add to each score of the term.
    intercept_shift : float
        The value to add to the intercept and to each bagged intercept.

    """
    _check_finite([delta, intercept_shift])
    repair_uncertainty(ebm)
    ebm.term_scores_[term] = np.asarray(ebm.term_scores_[term], dtype=float) + delta
    ebm.bagged_scores_[term] = np.asarray(ebm.bagged_scores_[term], dtype=float) + delta
    _shift_intercept(ebm, intercept_shift)


def scale_term(ebm: ExplainableBoostingModel, term: int, factor: float) -> None:
    """Multiply the scores, the bags and the standard deviations of a term by a factor.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    term : int
        The index of the term.
    factor : float
        The factor. A negative factor flips the graph, and the standard deviation scales by its absolute value.

    """
    _check_finite(factor)
    repair_uncertainty(ebm)
    ebm.term_scores_[term] = np.asarray(ebm.term_scores_[term], dtype=float) * factor
    ebm.bagged_scores_[term] = np.asarray(ebm.bagged_scores_[term], dtype=float) * factor
    ebm.standard_deviations_[term] = np.asarray(ebm.standard_deviations_[term], dtype=float) * abs(factor)


def remove_terms(ebm: ExplainableBoostingModel, term_names: Sequence[str]) -> None:
    """Remove terms with ``interpret`` and repair the attributes that it leaves incomplete.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    term_names : Sequence[str]
        The names of the terms to remove.

    """
    ebm.remove_terms(list(term_names))
    repair_uncertainty(ebm)


def _cut_levels(ebm: ExplainableBoostingModel, feature_index: int) -> list[npt.NDArray[np.float64]]:
    # a continuous feature holds arrays of cuts, and the stubs of interpret also allow the dicts of a nominal feature
    return cast(list[npt.NDArray[np.float64]], list(ebm.bins_[feature_index]))


def _own_main_cuts(ebm: ExplainableBoostingModel, feature_index: int) -> None:
    """Give the main effect of a feature its own cuts, if a pair term reads the same cuts.

    ``interpret`` reads the cuts of a term with ``n`` features at ``bins_[feature][min(n, len) - 1]``. With one level,
    the pair terms read the cuts of the main effect, so a change of those cuts would break them. A copy as the first
    level gives the main effect its own cuts, and the pair terms then read the second level, which keeps the old cuts.
    """
    levels = _cut_levels(ebm, feature_index)
    used_by_pair = any(len(features) > 1 and feature_index in features for features in ebm.term_features_)
    if len(levels) == 1 and used_by_pair:
        ebm.bins_[feature_index] = [np.array(levels[0], dtype=float), levels[0]]


@attrs.frozen(eq=False)
class BinReplacement:
    """New bins for the real bins from position ``first`` to ``last`` of a continuous main effect.

    The new bins cover the same range of values as the old ones. The cuts between them are ``inner_cuts``, in
    increasing order, so there is one new bin more than inner cuts. ``scores``, ``weights`` and ``deviations`` hold one
    value for each new bin, and ``bags`` holds one row for each bag and one column for each new bin.
    """

    first: int
    last: int
    inner_cuts: Sequence[float]
    scores: npt.ArrayLike
    weights: npt.ArrayLike
    deviations: npt.ArrayLike
    bags: npt.ArrayLike


def replace_bins(ebm: ExplainableBoostingModel, feature_index: int, term: int, replacement: BinReplacement) -> None:
    """Replace real bins of a continuous main effect, and change every attribute of the bins together.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    feature_index : int
        The index of the feature in ``feature_names_in_``.
    term : int
        The index of the main effect of the feature.
    replacement : BinReplacement
        The positions to replace and the new bins.

    Raises
    ------
    EditError
        If a value is not finite, or the shapes do not fit the number of new bins.

    """
    first, last = replacement.first, replacement.last
    new_count = len(replacement.inner_cuts) + 1
    new_bags = np.asarray(replacement.bags, dtype=float)
    columns = [
        np.asarray(values, dtype=float) for values in (replacement.scores, replacement.weights, replacement.deviations)
    ]
    if any(column.shape != (new_count,) for column in columns) or new_bags.shape[1:] != (new_count,):
        msg = f"An edit that changes the bins must give {new_count} values for each attribute of the new bins."
        raise EditError(msg)
    for values in (*columns, new_bags, replacement.inner_cuts):
        _check_finite(values)
    repair_uncertainty(ebm)
    _own_main_cuts(ebm, feature_index)
    levels = _cut_levels(ebm, feature_index)
    cuts = np.asarray(levels[0], dtype=float)
    inner = np.asarray(replacement.inner_cuts, dtype=float)
    levels[0] = np.concatenate((cuts[: first - 1], inner, cuts[last - 1 :]))
    ebm.bins_[feature_index] = levels
    new_scores, new_weights, new_deviations = columns

    def spliced(old: npt.ArrayLike, new: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        values = np.asarray(old, dtype=float)
        return np.concatenate((values[..., :first], new, values[..., last + 1 :]), axis=-1)

    ebm.term_scores_[term] = spliced(ebm.term_scores_[term], new_scores)
    ebm.bin_weights_[term] = spliced(ebm.bin_weights_[term], new_weights)
    ebm.standard_deviations_[term] = spliced(ebm.standard_deviations_[term], new_deviations)
    ebm.bagged_scores_[term] = spliced(ebm.bagged_scores_[term], new_bags)


def set_bounds(ebm: ExplainableBoostingModel, feature_index: int, low: float, high: float) -> None:
    """Set the lowest and the highest value that the graph of a continuous feature shows.

    ``interpret`` bins a value by the cuts only, so the bounds change no prediction. They are the outer edges of the
    first and the last bin, which ``SplitBin`` reads.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM to change.
    feature_index : int
        The index of the feature.
    low, high : float
        The new bounds.

    """
    _check_finite([low, high])
    ebm.feature_bounds_[feature_index] = np.array([low, high], dtype=float)
