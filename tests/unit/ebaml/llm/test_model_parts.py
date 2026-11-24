"""Tests for the parts of an EBM that the descriptions cover and leave out."""

from structlog.testing import capture_logs

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.llm.model_parts import feature_importances_text, features_without_graph, skipped_interactions
from tests.unit.ebaml.conftest import fake_model


def test_feature_importances_text() -> None:
    ebm = fake_model(feature_names_in_=["a", "b"], term_features_=[(0,), (1,)], term_importances=lambda: [1.234, 0.5])
    assert feature_importances_text(ebm) == "a: 1.23\nb: 0.50"


def test_feature_importances_text_pairs_each_feature_with_its_own_term() -> None:
    # "a" has no term, so the first importance is that of "b"; the text gave it to "a" and left out "c"
    ebm = fake_model(
        feature_names_in_=["a", "b", "c"],
        term_features_=[(1,), (2,), (1, 2)],
        term_importances=lambda: [1.0, 2.0, 0.5],
    )
    assert feature_importances_text(ebm) == "b: 1.00\nc: 2.00"


def test_feature_importances_text_computes_the_importances_once() -> None:
    calls = []

    def term_importances() -> list[float]:
        calls.append(1)
        return [1.0, 2.0, 3.0]

    ebm = fake_model(
        feature_names_in_=["a", "b", "c"],
        term_features_=[(0,), (1,), (2,)],
        term_importances=term_importances,
    )
    feature_importances_text(ebm)
    assert len(calls) == 1


INTERACTIONS: ExplainableBoostingClassifier = fake_model(
    feature_names_in_=["a", "b", "c"],
    term_names_=["a", "b", "c", "a & b", "b & c"],
    term_features_=[(0,), (1,), (2,), (0, 1), (1, 2)],
)


def test_the_skipped_interactions_are_the_terms_that_touch_a_described_feature_in_model_order() -> None:
    assert skipped_interactions(INTERACTIONS, {0}) == ["a & b"]
    assert skipped_interactions(INTERACTIONS, {1}) == ["a & b", "b & c"]
    assert skipped_interactions(INTERACTIONS, {0, 1, 2}) == ["a & b", "b & c"]


def test_skipped_interactions_are_logged_with_the_reason() -> None:
    with capture_logs() as logs:
        skipped_interactions(INTERACTIONS, {2})
    assert [(log["event"], log["terms"], log["log_level"]) for log in logs] == [
        ("interactions_skipped", ["b & c"], "warning"),
    ]


def test_a_model_without_interactions_skips_no_term_and_logs_nothing() -> None:
    ebm = fake_model(feature_names_in_=["a"], term_names_=["a"], term_features_=[(0,)])
    with capture_logs() as logs:
        assert skipped_interactions(ebm, {0}) == []
    assert logs == []


def test_the_features_without_a_graph_are_the_features_that_are_not_described() -> None:
    with capture_logs() as logs:
        assert features_without_graph(INTERACTIONS, [0, 2]) == ["b"]
    assert [(log["event"], log["features"], log["log_level"]) for log in logs] == [
        ("features_skipped", ["b"], "warning"),
    ]


def test_a_model_with_a_graph_for_every_feature_logs_nothing() -> None:
    with capture_logs() as logs:
        assert features_without_graph(INTERACTIONS, [0, 1, 2]) == []
    assert logs == []
