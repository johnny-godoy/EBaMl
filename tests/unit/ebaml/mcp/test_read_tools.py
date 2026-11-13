"""Tests of the tools that read an EBM."""

import pytest

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import EditError, InvalidInstanceError
from ebaml.edit.edit_session import RevisionInfo
from ebaml.edit.edits import Scale
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.read_tools import graph_text, list_features, model_summary
from tests.unit.ebaml.conftest import FittedClassifier


def test_summary_of_a_binary_classifier(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    summary = model_summary(ModelStore(ebm=ebm))
    assert summary.kind == "BinaryClassifier"
    assert summary.outputs == ["1"]
    assert summary.classes == [0, 1]
    assert summary.feature_count == 3
    assert summary.skipped_terms == []
    assert len(summary.intercept) == 1
    assert "log-odds" in summary.score_meaning


def test_summary_of_a_regressor_has_no_classes(regression_ebm: ExplainableBoostingRegressor) -> None:
    """A regressor stores its intercept as a plain float, not as an array."""
    summary = model_summary(ModelStore(ebm=regression_ebm))
    assert summary.kind == "Regressor"
    assert summary.classes == []
    assert len(summary.intercept) == 1


def test_summary_of_a_multiclass_classifier_has_one_intercept_per_class(
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    summary = model_summary(ModelStore(ebm=multiclass_ebm))
    assert summary.kind == "MulticlassClassifier"
    assert sorted(summary.classes) == ["high", "low", "mid"]
    assert len(summary.intercept) == 3


def test_summary_names_the_skipped_interaction_terms(interaction_ebm: ExplainableBoostingClassifier) -> None:
    summary = model_summary(ModelStore(ebm=interaction_ebm))
    assert summary.skipped_terms == ["p x q"]
    assert summary.feature_count == 2


def test_features_are_sorted_by_importance(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    features = list_features(ModelStore(ebm=ebm)).features
    importances = [feature.importance for feature in features]
    assert importances == sorted(importances, reverse=True)
    assert features[0].name == "a"
    assert {feature.type for feature in features} == {"continuous", "nominal"}


def test_graph_text_names_the_feature(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    text = graph_text(ModelStore(ebm=ebm), "c", max_tokens=1000)
    assert "x" in text
    assert text != graph_text(ModelStore(ebm=ebm), "c", normalization="min_max", max_tokens=1000)


def test_graph_text_of_an_unknown_feature_is_an_error(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    with pytest.raises(InvalidInstanceError, match="no feature 'zzz'"):
        graph_text(ModelStore(ebm=ebm), "zzz")


def test_a_small_max_tokens_gives_a_shorter_graph_text(
    continuous_ebm: FittedClassifier,
) -> None:
    """Bug: ``max_tokens`` had no effect, because ``graph_text`` passed no ``check_budget`` to ``to_text``."""
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    with pytest.warns(UserWarning, match="above the budget"):
        short = graph_text(store, "a", max_tokens=1500)
    with pytest.warns(UserWarning, match="above the budget"):
        long = graph_text(store, "a", max_tokens=3000)
    assert len(short) < len(long)


def test_each_read_tool_uses_the_edited_copy_and_a_revision_reads_an_earlier_model(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    original_text = graph_text(store, "a")
    original_importance = list_features(store).features[0].importance
    store.editing().apply(Scale("a", 0.0), "The model must not use a")
    edit = store.editing().revision
    assert graph_text(store, "a") != original_text
    assert graph_text(store, "a", revision="original").split("\n", 1)[1] == original_text.split("\n", 1)[1]
    assert graph_text(store, "a", revision=edit) == graph_text(store, "a")
    features = list_features(store)
    assert "a" not in [feature.name for feature in features.features[:1]]
    assert list_features(store, "original").features[0].importance == original_importance
    assert features.session == RevisionInfo(edit, edit, 1)
    assert model_summary(store).session == model_summary(store, edit).session == features.session
    assert model_summary(store, "original").session == RevisionInfo("original", edit, 1)


def test_the_graph_text_says_which_revision_it_reads(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    assert graph_text(store, "a").startswith("Revision original. The head is original, and it holds 0 applied edits.\n")
    store.editing().apply(Scale("a", 2.0), "Double a")
    head = store.editing().revision
    assert graph_text(store, "a", revision="original").startswith(
        f"Revision original. The head is {head}, and it holds 1 applied edits.\n",
    )


def test_a_read_of_an_unknown_revision_is_an_error(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    for read in (model_summary, list_features):
        with pytest.raises(EditError, match="no revision 'abc'"):
            read(store, "abc")
    with pytest.raises(EditError, match="no revision 'abc'"):
        graph_text(store, "a", revision="abc")
