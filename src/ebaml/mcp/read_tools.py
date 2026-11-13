"""The tools that read an EBM: its summary, its features and the text of a graph.

Each function calls code of a lower layer and returns an attrs class. The server turns the class into JSON.
"""

import attrs
import numpy as np

from ebaml.core.custom_types import ClassLabel, Normalization
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.edit.edit_session import CURRENT, RevisionInfo
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
    session: RevisionInfo
        The revision that the summary reads, the head, and the number of edits that the head holds.

    """

    kind: str
    outputs: list[str]
    classes: list[ClassLabel]
    score_meaning: str
    intercept: list[float]
    feature_count: int
    skipped_terms: list[str]
    session: RevisionInfo


@attrs.frozen
class FeatureInfo:
    """One feature of the model."""

    name: str
    type: str
    importance: float


@attrs.frozen
class FeatureList:
    """The features with a graph, the most important first, and the revision that the list reads."""

    session: RevisionInfo
    features: list[FeatureInfo]


def model_summary(store: ModelStore, revision: str = CURRENT) -> ModelSummary:
    """Summarize the model.

    Parameters
    ----------
    store: ModelStore
        The session.
    revision: str, default="current"
        ``original``, ``current`` or the id of a revision of the history.

    Returns
    -------
    ModelSummary
        The kind, the outputs, the intercept, the number of features and the interaction terms that the tools skip.

    """
    read = store.read(revision)
    ebm = read.model
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
        session=read.info,
    )


def list_features(store: ModelStore, revision: str = CURRENT) -> FeatureList:
    """List the features with a graph, the most important first.

    Parameters
    ----------
    store: ModelStore
        The session.
    revision: str, default="current"
        The revision to list. See ``model_summary``.

    Returns
    -------
    FeatureList
        The name, the type and the importance of each feature, sorted by importance, and the revision that the list
        reads.

    """
    read = store.read(revision)
    ebm = read.model
    importances = ebm.term_importances()
    features = [
        FeatureInfo(
            name=ebm.feature_names_in_[index],
            type=ebm.feature_types_in_[index],
            importance=float(importances[main_term(ebm, index)]),
        )
        for index in main_effect_features(ebm)
    ]
    return FeatureList(read.info, sorted(features, key=lambda feature: feature.importance, reverse=True))


def graph_text(  # ruff: ignore[too-many-arguments] -- one argument for each option of the get_graph tool
    store: ModelStore,
    feature: str,
    *,
    revision: str = CURRENT,
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
    revision: str, default="current"
        ``original``, ``current`` or the id of a revision of the history.
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
        A line that names the revision and the edits that the head holds, then the text of ``EBMGraph.to_text``.

    """
    read = store.read(revision)
    graph = EBMGraph.from_ebm(read.model, store.feature_index(feature), normalization)
    text = graph.to_text(
        x_axis_precision=x_axis_precision,
        y_axis_precision=y_axis_precision,
        max_tokens=max_tokens,
        check_budget=check_token_budget,
    )
    info = read.info
    note = f"Revision {info.revision}. The head is {info.head}, and it holds {info.edit_count} applied edits."
    return f"{note}\n{text}"
