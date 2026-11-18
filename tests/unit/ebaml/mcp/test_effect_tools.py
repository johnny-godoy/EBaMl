"""Tests of the effect tools, called as functions on a store."""

import pytest

from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.edit_effect import ClassifierMetrics, Subgroup
from ebaml.edit.edits import Shift
from ebaml.edit.selections import Bins
from ebaml.mcp import effect_tools
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import FittedClassifier, as_frame


@pytest.fixture
def store(continuous_ebm: FittedClassifier) -> ModelStore:
    ebm, data = continuous_ebm
    labelled = data.assign(outcome=ebm.predict(data))
    return ModelStore(ebm=ebm, data=as_frame(labelled), label_column="outcome")


def test_subgroup_of_needs_both_arguments_or_none() -> None:
    assert effect_tools.subgroup_of(None, None) is None
    assert effect_tools.subgroup_of("c", "x") == Subgroup("c", "x")
    for feature, value in (("c", None), (None, "x")):
        with pytest.raises(InvalidInstanceError, match="A subgroup needs both subgroup_feature and subgroup_value"):
            effect_tools.subgroup_of(feature, value)


def test_the_effect_holds_the_metrics_of_each_scope_when_the_server_has_labels(store: ModelStore) -> None:
    store.editing().apply(Shift("a", Bins(0, 3), 2.0), "Low values of a were under-estimated")
    result = effect_tools.get_edit_effect(store, Subgroup("c", "x"))
    assert result.note is None
    assert result.session == store.editing().revision_info()
    assert [scope.scope for scope in result.effect.scopes] == ["all rows", "affected rows", "subgroup"]
    everything = result.effect.scopes[0]
    assert isinstance(everything.original, ClassifierMetrics)
    assert everything.original.accuracy == pytest.approx(1.0)  # the labels are the predictions of the original model
    assert result.effect.scopes[1].row_count > 0


def test_without_labels_the_effect_counts_rows_and_says_that_it_has_no_metric(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    result = effect_tools.get_edit_effect(ModelStore(ebm=ebm, data=as_frame(data)))
    assert result.note is not None
    assert "no label column (--labels)" in result.note
    assert result.effect.scopes[0].current is None


def test_the_profile_names_the_other_features(store: ModelStore) -> None:
    store.editing().apply(Shift("a", Bins(0, 3), 2.0), "Low values of a were under-estimated")
    result = effect_tools.get_affected_profile(store)
    assert result.profile.edited_feature == "a"
    assert {feature.feature for feature in result.profile.features} == {"b", "c"}
    assert result.session == store.editing().revision_info()


def test_the_effect_tools_need_data(continuous_ebm: FittedClassifier) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    with pytest.raises(InvalidInstanceError, match=r"The server has no data\. Start it with --data"):
        effect_tools.get_edit_effect(store)
    with pytest.raises(InvalidInstanceError, match="The server has no data"):
        effect_tools.get_affected_profile(store)
