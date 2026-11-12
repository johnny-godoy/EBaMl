"""The only functions that write the fitted attributes of an EBM.

An edited model must work wherever a fitted model works: ``predict``, ``predict_with_uncertainty``, ``explain_global``,
``EBMGraph.from_ebm`` and a save with ``skops``. So every edit writes through these functions, and no edit writes an
attribute itself. The functions keep three facts true:

- ``bagged_scores_`` holds the edit, so ``predict_with_uncertainty`` gives the edited prediction.
- A score that a person sets has no fitted uncertainty, so its standard deviation is 0.
- No attribute holds ``None``. ``interpret`` leaves ``None`` after ``monotonize``, and ``repair_uncertainty`` fills it.
"""

from collections.abc import Sequence

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
