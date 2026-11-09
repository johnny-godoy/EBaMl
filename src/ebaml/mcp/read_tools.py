"""The tools that read an EBM: its summary, its features and the text of a graph.

Each function calls code of a lower layer and returns an attrs class. The server turns the class into JSON.
"""

import attrs
import numpy as np

from ebaml.core.custom_types import ClassLabel, Normalization
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.model_kinds import model_kind
from ebaml.graphs.terms import main_effect_features, main_term
from ebaml.graphs.token_budget import check_token_budget
from ebaml.mcp.model_store import ModelStore


@attrs.frozen
class ModelSummary:
    """The facts about the model that an agent needs first.

    Attributes
    ----------
    kind: str
        ``BinaryClassifier``, ``MulticlassClassifier`` or ``Regressor``.
    outputs: list[str]
        The names of the outputs of the model.
    classes: list[ClassLabel]
        The class labels, in the order of the scores. The list is empty for a regressor.
    score_meaning: str
        What a score of a graph means.
    intercept: list[float]
        The intercept: one value, or one value per class for a multiclass classifier.
    feature_count: int
        The number of features that have a graph.
    skipped_terms: list[str]
        The interaction terms. The tools skip them.

    """

    kind: str
    outputs: list[str]
    classes: list[ClassLabel]
    score_meaning: str
    intercept: list[float]
    feature_count: int
    skipped_terms: list[str]


@attrs.frozen
class FeatureInfo:
    """One feature of the model."""

    name: str
    type: str
    importance: float


def model_summary(store: ModelStore) -> ModelSummary:
    """Summarize the model.

    Parameters
    ----------
    store: ModelStore
        The session.

    Returns
    -------
    ModelSummary
        The kind, the outputs, the intercept, the number of features and the interaction terms that the tools skip.

    """
    ebm = store.ebm
    kind = model_kind(ebm)
    names = list(ebm.feature_names_in_)
    # only a classifier has class labels; one check at the entry narrows the union of the two interpret classes
    classes: list[ClassLabel] = (
        np.asarray(ebm.classes_).tolist() if isinstance(ebm, ExplainableBoostingClassifier) else []
    )
    skipped = [" x ".join(names[index] for index in features) for features in ebm.term_features_ if len(features) > 1]
    return ModelSummary(
        kind=type(kind).__name__,
        outputs=kind.outputs,
        classes=classes,
        score_meaning=kind.score_meaning,
        intercept=np.ravel(ebm.intercept_).astype(float).tolist(),
        feature_count=len(main_effect_features(ebm)),
        skipped_terms=skipped,
    )


def list_features(store: ModelStore) -> list[FeatureInfo]:
    """List the features with a graph, the most important first.

    Parameters
    ----------
    store: ModelStore
        The session.

    Returns
    -------
    list[FeatureInfo]
        The name, the type and the importance of each feature, sorted by importance.

    """
    ebm = store.ebm
    importances = ebm.term_importances()
    features = [
        FeatureInfo(
            name=ebm.feature_names_in_[index],
            type=ebm.feature_types_in_[index],
            importance=float(importances[main_term(ebm, index)]),
        )
        for index in main_effect_features(ebm)
    ]
    return sorted(features, key=lambda feature: feature.importance, reverse=True)


def graph_text(  # ruff: ignore[too-many-arguments] -- one argument for each option of the get_graph tool
    store: ModelStore,
    feature: str,
    *,
    normalization: Normalization | str = Normalization.NONE,
    x_axis_precision: int | None = None,
    y_axis_precision: int | None = None,
    max_tokens: int = 3000,
) -> str:
    """Write the graph of a feature as text.

    Parameters
    ----------
    store: ModelStore
        The session.
    feature: str
        The name of the feature.
    normalization: Normalization | str, default=Normalization.NONE
        See ``EBMGraph.from_ebm``.
    x_axis_precision: int, optional
        See ``EBMGraph.to_text``.
    y_axis_precision: int, optional
        See ``EBMGraph.to_text``.
    max_tokens: int, default=3000
        See ``EBMGraph.to_text``.

    Returns
    -------
    str
        The text of ``EBMGraph.to_text``.

    """
    graph = EBMGraph.from_ebm(store.ebm, store.feature_index(feature), normalization)
    return graph.to_text(
        x_axis_precision=x_axis_precision,
        y_axis_precision=y_axis_precision,
        max_tokens=max_tokens,
        check_budget=check_token_budget,
    )
