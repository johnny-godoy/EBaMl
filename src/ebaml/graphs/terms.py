"""The term of an EBM that holds the main effect of each feature.

interpret keeps the scores, the standard deviations, the bin weights and the importances by term, and the names, the
types and the bins by feature. Term ``i`` is not always feature ``i``: a feature that ``exclude`` leaves out at fit
time, or that ``remove_terms`` removes later, has no term, and the terms after it move. So other code finds the term
of a feature here, and does not index the term attributes with the index of a feature.
"""

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import UnsupportedModelError


def main_term(ebm: ExplainableBoostingModel, feature_index: int) -> int:
    """Find the term that holds the main effect of a feature.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    feature_index: int
        The index of the feature in ``feature_names_in_``.

    Returns
    -------
    int
        The index of the term in ``term_features_``, ``term_scores_`` and the other term attributes.

    Raises
    ------
    UnsupportedModelError
        If the feature has no main-effect term, because ``exclude`` or ``remove_terms`` left it out.

    """
    for term_index, features in enumerate(ebm.term_features_):
        if tuple(features) == (feature_index,):
            return term_index
    msg = (
        f"The feature {ebm.feature_names_in_[feature_index]!r} has no main-effect term in the EBM,"
        " because exclude or remove_terms left it out."
    )
    raise UnsupportedModelError(msg)


def main_effect_features(ebm: ExplainableBoostingModel) -> list[int]:
    """List the features that have a main-effect term.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.

    Returns
    -------
    list[int]
        The index of each such feature in ``feature_names_in_``, in increasing order.

    """
    return sorted(features[0] for features in ebm.term_features_ if len(features) == 1)
