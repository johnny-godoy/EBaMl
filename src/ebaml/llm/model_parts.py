"""Name the parts of an EBM that the graph descriptions cover, and the parts that they leave out.

A description of an EBM covers the main effect of each feature. These functions write the importances of the main
effects for the summary, and name the interaction terms and the features without a main effect, so that the prompts can
say what the descriptions leave out (semantic contract 4).
"""

from collections.abc import Collection

import structlog

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.graphs.terms import main_effect_features, main_term

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)


def feature_importances_text(ebm: ExplainableBoostingModel) -> str:
    """Convert the importances of the main-effect terms of an EBM to text, one line per feature.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM model.

    Returns
    -------
        str: Textual representation of the feature importances.

    """
    # the importances are kept by term, and term i is not always feature i
    importances = ebm.term_importances()
    return "\n".join(
        f"{ebm.feature_names_in_[feature_idx]}: {importances[main_term(ebm, feature_idx)]:.2f}"
        for feature_idx in main_effect_features(ebm)
    )


def skipped_interactions(ebm: ExplainableBoostingModel, described: Collection[int]) -> list[str]:
    """Name the interaction terms of the described features, which the descriptions skip, and log a warning.

    The names go to the prompt, so that the text of the description says what it leaves out. The warning tells the
    program the same.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM.
    described : Collection[int]
        The indices of the features that the call describes.

    Returns
    -------
    list[str]
        The names of the skipped terms, in the order of the model.

    """
    skipped = [
        name
        for name, features in zip(ebm.term_names_, ebm.term_features_, strict=True)
        if (
            len(features) > 1  # pragma: no mutate -- a term of zero features does not exist
            and any(feature in described for feature in features)
        )
    ]
    if skipped:
        logger.warning("interactions_skipped", terms=skipped, reason="the descriptions cover the main effects only")
    return skipped


def features_without_graph(ebm: ExplainableBoostingModel, described: Collection[int]) -> list[str]:
    """Name the features that have no main-effect term, which the descriptions skip, and log a warning.

    Parameters
    ----------
    ebm : ExplainableBoostingModel
        The EBM.
    described : Collection[int]
        The indices of the features that have a graph.

    Returns
    -------
    list[str]
        The names of the features that ``exclude`` or ``remove_terms`` left without a graph.

    """
    skipped = [name for index, name in enumerate(ebm.feature_names_in_) if index not in described]
    if skipped:
        logger.warning("features_skipped", features=skipped, reason="the features have no main-effect term")
    return skipped
