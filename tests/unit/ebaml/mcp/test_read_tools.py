"""Tests of the tools that read an EBM."""

import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.exceptions import InvalidInstanceError
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.read_tools import graph_text, list_features, model_summary


def test_summary_of_a_binary_classifier(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
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


def test_summary_names_the_skipped_interaction_terms(fit_ebm: object) -> None:
    rng_data = pd.DataFrame({"p": range(300), "q": [value % 7 for value in range(300)]})
    target = ((rng_data["p"] % 5) * (rng_data["q"] % 3) > 2).astype(int)
    ebm = fit_ebm(rng_data, target, interactions=1, random_state=0)  # pyrefly: ignore[not-callable] -- the fixture is a callable
    summary = model_summary(ModelStore(ebm=ebm))
    assert summary.skipped_terms == ["p x q"]
    assert summary.feature_count == 2


def test_features_are_sorted_by_importance(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, _ = continuous_ebm
    features = list_features(ModelStore(ebm=ebm))
    importances = [feature.importance for feature in features]
    assert importances == sorted(importances, reverse=True)
    assert features[0].name == "a"
    assert {feature.type for feature in features} == {"continuous", "nominal"}


def test_graph_text_names_the_feature(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> None:
    ebm, _ = continuous_ebm
    text = graph_text(ModelStore(ebm=ebm), "c", max_tokens=1000)
    assert "x" in text
    assert text != graph_text(ModelStore(ebm=ebm), "c", normalization="min_max", max_tokens=1000)


def test_graph_text_of_an_unknown_feature_is_an_error(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, _ = continuous_ebm
    with pytest.raises(InvalidInstanceError, match="no feature 'zzz'"):
        graph_text(ModelStore(ebm=ebm), "zzz")
