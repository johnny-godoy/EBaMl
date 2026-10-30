"""Tests for the lookup of the main-effect term of each feature."""

import copy

import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.exceptions import UnsupportedModelError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.terms import main_effect_features, main_term
from tests.unit.ebaml.conftest import fake_model


def test_main_term_finds_the_term_of_a_feature_after_a_left_out_feature() -> None:
    # "a" has no term, so the term of "b" is the first one, and the term of "c" is the second one
    ebm = fake_model(feature_names_in_=["a", "b", "c"], term_features_=[(1,), (2,), (1, 2)])
    assert main_term(ebm, 1) == 0
    assert main_term(ebm, 2) == 1


def test_main_term_skips_the_interaction_terms_of_the_feature() -> None:
    ebm = fake_model(feature_names_in_=["a", "b"], term_features_=[(0, 1), (1,), (0,)])
    assert main_term(ebm, 0) == 2


def test_main_term_accepts_the_terms_as_lists() -> None:
    ebm = fake_model(feature_names_in_=["a"], term_features_=[[0]])
    assert main_term(ebm, 0) == 0


def test_main_term_names_a_feature_without_a_term() -> None:
    ebm = fake_model(feature_names_in_=["a", "b"], term_features_=[(1,), (0, 1)])
    with pytest.raises(UnsupportedModelError, match=r"^The feature 'a' has no main-effect term in the EBM, because"):
        main_term(ebm, 0)


def test_main_effect_features_lists_the_features_with_a_term_in_order() -> None:
    ebm = fake_model(feature_names_in_=["a", "b", "c"], term_features_=[(2,), (0,), (0, 2)])
    assert main_effect_features(ebm) == [0, 2]


def test_main_effect_features_of_a_model_without_terms_is_empty() -> None:
    assert main_effect_features(fake_model(feature_names_in_=["a"], term_features_=[])) == []


def test_from_ebm_after_remove_terms_keeps_the_graph_of_each_feature(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    # after remove_terms(["a"]), the graph named "a" held the scores of "b"
    ebm = copy.deepcopy(continuous_ebm[0])
    before = EBMGraph.from_ebm(ebm, 1)
    ebm.remove_terms(["a"])
    after = EBMGraph.from_ebm(ebm, 1)
    assert after.feature_name == "b"
    assert np.asarray(after.scores).tolist() == np.asarray(before.scores).tolist()
    with pytest.raises(UnsupportedModelError, match="'a' has no main-effect term"):
        EBMGraph.from_ebm(ebm, 0)
