"""Tests for the EBM graphs."""

import inspect
import itertools
import warnings
from collections.abc import Callable
from typing import Any, Literal

import attrs
import numpy as np
import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

import baml_sdk
from ebaml.core.custom_types import FeatureType, Normalization
from ebaml.core.exceptions import (
    EBMGraphShapeError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    GraphTextError,
    InvalidOptionError,
    OptionRangeError,
    UnsupportedModelError,
)
from ebaml.graphs.ebm_graph import BudgetCheck, EBMGraph, TokenBudget
from ebaml.graphs.model_kinds import MulticlassClassifier
from tests.unit.ebaml.conftest import appending, call_mismatches, fake_model, fake_sdk, protocol_mismatches

INF = float("inf")
type Status = Literal["ok", "warn", "over"]


def budget(status: Status, tokens: int = 10) -> baml_sdk.TokenBudget:
    return baml_sdk.TokenBudget(estimated_tokens=tokens, status=status)


def checker(*statuses: Status, tokens: int = 10) -> Callable[..., baml_sdk.TokenBudget]:
    # the check answers with each status in turn, and with the last one after that
    answers = itertools.chain(statuses, itertools.repeat(statuses[-1]))

    def check(**_kw: object) -> baml_sdk.TokenBudget:
        return budget(next(answers), tokens)

    return fake_sdk("check_token_budget", check)


def continuous_ebm(scores: tuple[float, ...] = (9.0, 1.0, 2.0, 4.0, 9.0)) -> ExplainableBoostingClassifier:
    return fake_model(
        feature_names_in_=["age"],
        feature_types_in_=["continuous"],
        term_features_=[(0,)],
        term_scores_=[np.array(scores)],
        standard_deviations_=[np.array([0.0, 0.1, 0.2, 0.3, 0.0])],
        # the second resolution level is a decoy: only the first one describes the graph
        bins_=[[np.array([1.0, 2.0]), np.array([5.0, 6.0, 7.0, 8.0])]],
        feature_bounds_=[(0.0, 3.0)],
        link_="logit",
        classes_=np.array([0, 1]),
    )


def nominal_ebm() -> ExplainableBoostingClassifier:
    return fake_model(
        feature_names_in_=["color"],
        feature_types_in_=["nominal"],
        term_features_=[(0,)],
        term_scores_=[np.array([9.0, 1.0, 3.0, 5.0, 7.0, 9.0])],
        standard_deviations_=[np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.0])],
        # unordered on purpose, with the decoy second resolution level
        bins_=[[{"pink": 4, "red": 1, "green": 3, "blue": 2}, {"x": 1}]],
        link_="logit",
        classes_=np.array([0, 1]),
    )


def cont_graph(scores: tuple[float, ...] = (1.0, 1.0, 5.0), stds: tuple[float, ...] = (0.1, 0.2, 0.3)) -> EBMGraph:
    return EBMGraph("age", "continuous", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], list(scores), list(stds))


def cat_graph() -> EBMGraph:
    return EBMGraph("color", "nominal", ["red", "blue"], [1.0, 3.0], [0.1, 0.2])


# from_ebm


def test_from_ebm_continuous_with_bounds() -> None:
    g = EBMGraph.from_ebm(continuous_ebm(), 0)
    assert g.x_vals == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert list(g.scores) == [1.0, 2.0, 4.0]
    assert list(g.stds) == [0.1, 0.2, 0.3]
    assert (g.feature_name, g.feature_type) == ("age", "continuous")


def test_from_ebm_continuous_without_bounds() -> None:
    g = EBMGraph.from_ebm(continuous_ebm(), 0, use_feature_bounds=False)
    assert g.x_vals == [(-INF, 1.0), (1.0, 2.0), (2.0, INF)]


def test_from_ebm_nominal() -> None:
    g = EBMGraph.from_ebm(nominal_ebm(), 0)
    assert g.x_vals == ["red", "blue", "green", "pink"]
    assert list(g.scores) == [1.0, 3.0, 5.0, 7.0]


def test_from_ebm_z_score() -> None:
    g = EBMGraph.from_ebm(continuous_ebm(), 0, Normalization.Z_SCORE)
    assert list(g.scores) == pytest.approx([-4 / 3, -1 / 3, 5 / 3])


def test_from_ebm_min_max() -> None:
    g = EBMGraph.from_ebm(continuous_ebm(), 0, Normalization.MIN_MAX)
    assert list(g.scores) == [0.0, 1.0, 3.0]


@pytest.mark.parametrize("normalization", [Normalization.Z_SCORE, Normalization.MIN_MAX])
def test_from_ebm_does_not_modify_the_ebm(normalization: Normalization) -> None:
    ebm = continuous_ebm()
    before = ebm.term_scores_[0].copy()
    EBMGraph.from_ebm(ebm, 0, normalization)
    assert list(ebm.term_scores_[0]) == list(before)


def test_from_ebm_invalid_normalization() -> None:
    with pytest.raises(InvalidOptionError):
        EBMGraph.from_ebm(continuous_ebm(), 0, "bogus")


def test_from_ebm_unsupported_feature_type() -> None:
    ebm = continuous_ebm()
    ebm.feature_types_in_ = ["interaction"]
    with pytest.raises(InvalidOptionError, match="interaction"):
        EBMGraph.from_ebm(ebm, 0)


def test_from_ebm_bin_score_mismatch() -> None:
    ebm = continuous_ebm(scores=(9.0, 1.0, 2.0, 9.0))
    with pytest.raises(EBMGraphShapeError, match="does not match"):
        EBMGraph.from_ebm(ebm, 0)


# simplify


class SubGraph(EBMGraph):
    """A subclass, to check that the methods that return ``Self`` keep it."""


def test_from_ebm_of_a_subclass_returns_the_subclass() -> None:
    # from_ebm returned a plain EBMGraph, which broke its `Self` hint and failed the runtime type check
    assert type(SubGraph.from_ebm(continuous_ebm(), 0)) is SubGraph


def test_simplify_of_a_subclass_returns_the_subclass() -> None:
    graph = SubGraph("age", "continuous", [(0.0, 1.0), (1.0, 2.0)], [1.0, 1.0], [0.1, 0.2])
    assert type(graph.simplify()) is SubGraph


def test_simplify_rejects_non_continuous() -> None:
    with pytest.raises(InvalidOptionError):
        cat_graph().simplify()


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_simplify_rejects_bad_variation(bad: float) -> None:
    with pytest.raises(OptionRangeError, match="between 0 and 1"):
        cont_graph().simplify(bad)


def test_simplify_zero_merges_only_equal_bins() -> None:
    g = cont_graph().simplify()
    assert g.x_vals == [(0.0, 2.0), (2.0, 3.0)]
    assert g.scores == [1.0, 5.0]
    assert g.stds == [0.2, 0.3]  # max of merged stds


def test_simplify_keeps_distinct_bins() -> None:
    g = cont_graph(scores=(1.0, 2.0, 5.0)).simplify()
    assert g.x_vals == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_simplify_merges_within_threshold() -> None:
    g = cont_graph(scores=(1.0, 2.0, 5.0)).simplify(0.3)  # 0.3 * 4 = 1.2 >= |1-2|
    assert g.x_vals == [(0.0, 2.0), (2.0, 3.0)]


# to_text


def test_to_text_continuous() -> None:
    text = cont_graph().to_text()
    assert text.startswith("This graph represents a continuous-valued feature.")
    assert "Feature Name: age\nFeature Type: continuous\n" in text
    assert "Means: " in text
    assert "Lower Bounds (95%-Confidence Interval):" in text
    assert "Upper Bounds (95%-Confidence Interval):" in text


def test_to_text_no_description_no_bounds_explicit_precisions() -> None:
    text = cont_graph().to_text(
        include_description=False,
        confidence_bounds=False,
        x_axis_precision=1,
        y_axis_precision=2,
    )
    assert text == 'Feature Name: age\nFeature Type: continuous\nMeans: {"(0.0, 2.0)": 1.0, "(2.0, 3.0)": 5.0}\n'


def test_to_text_x_axis_precision_keeps_category_names() -> None:
    # regression: x_axis_precision unpacked every x value as an interval, so a category name raised ValueError
    text = cat_graph().to_text(x_axis_precision=2, confidence_bounds=False, include_description=False)
    assert text == 'Feature Name: color\nFeature Type: categorical\nMeans: {"red": 1.0, "blue": 3.0}\n'


def test_to_body_text_x_axis_precision_rounds_the_intervals_of_a_mixed_sequence() -> None:
    graph = EBMGraph("mixed", "continuous", [(0.123456, 1.0), "other"], [1.0, 2.0], [0.1, 0.1])  # pyrefly: ignore[bad-argument-type] -- a mixed sequence is the case under test
    text = graph.to_body_text(
        feature_format=FeatureType.CATEGORICAL,
        x_axis_precision=2,
        y_axis_precision=1,
        confidence_bounds=False,
        confidence_level=0.95,
    )
    assert 'Means: {"(0.12, 1.0)": 1.0, "other": 2.0}' in text


def test_to_text_y_precision_large_variation_skips_auto_loop() -> None:
    text = cont_graph(scores=(0.0, 0.0, 500.0)).to_text(y_axis_precision="auto", confidence_bounds=False)
    assert '"(2.0, 3.0)": 500.0' in text


def test_to_text_categorical() -> None:
    text = cat_graph().to_text(feature_format=FeatureType.CATEGORICAL)
    assert text.startswith("This graph represents categorical feature.")
    assert '"red": 1.0' in text


def test_to_text_boolean_autodetect() -> None:
    g = EBMGraph("flag", "nominal", ["False", "True"], [0.0, 2.0], [0.1, 0.1])
    text = g.to_text()
    assert text.startswith("This graph represents a boolean feature.")
    assert "Feature Type: boolean" in text


def test_to_text_nominal_defaults_to_categorical() -> None:
    # regression: nominal features used to raise unless feature_format was passed explicitly
    text = cat_graph().to_text()
    assert text.startswith("This graph represents categorical feature.")
    assert "Feature Type: categorical" in text


def test_to_text_flat_graph_terminates() -> None:
    # regression: zero variation made the y-axis precision loop run forever
    g = cont_graph(scores=(2.0, 2.0, 2.0))
    assert '"(0.0, 3.0)": 2.0' in g.to_text(confidence_bounds=False)


def test_to_text_flat_categorical_graph_terminates() -> None:
    g = EBMGraph("color", "nominal", ["red", "blue"], [1.0, 1.0], [0.1, 0.1])
    assert '"blue": 1.0' in g.to_text()


@pytest.mark.parametrize(
    ("x_vals", "count"),
    [(["a", "b", "c"], 3), (["a"], 1)],
)
def test_to_text_boolean_wrong_number_of_values(x_vals: list[str], count: int) -> None:
    # the message said "more than x-axis values", without the number, and was wrong for one value
    g = EBMGraph("flag", "nominal", x_vals, [0.0] * count, [0.1] * count)
    expected = rf"^Requested a boolean format, which needs exactly two x-axis values, but the feature has {count}: \["
    with pytest.raises(EBMGraphShapeError, match=expected):
        g.to_text(feature_format=FeatureType.BOOLEAN)


def test_to_text_unsupported_format() -> None:
    with pytest.raises(InvalidOptionError):
        cat_graph().to_text(feature_format=FeatureType.ORDINAL)


def test_to_text_warn_budget() -> None:
    with pytest.warns(UserWarning, match="estimated at 4000 tokens"):
        text = cont_graph().to_text(check_budget=checker("warn", tokens=4000))
    assert "Feature Name: age" in text


def test_to_text_simplifies_when_over_budget() -> None:
    assert "Feature Name: age" in cont_graph().to_text(check_budget=checker("over", "over", "ok"))


def test_to_text_continuous_cannot_be_simplified() -> None:
    with pytest.raises(EBMGraphSimplificationError, match="99999"):
        cont_graph().to_text(check_budget=checker("over", tokens=99999))


def test_to_text_categorical_over_budget() -> None:
    with pytest.raises(EBMGraphTokenLimitError, match="exceeds the token budget"):
        cat_graph().to_text(feature_format=FeatureType.CATEGORICAL, check_budget=checker("over", tokens=99999))


# mutation-testing regressions: from_ebm


@pytest.mark.parametrize("name", ["z_score", "min_max", "none"])
def test_from_ebm_accepts_plain_string_normalization(name: str) -> None:
    assert list(EBMGraph.from_ebm(continuous_ebm(), 0, name).scores) == list(
        EBMGraph.from_ebm(continuous_ebm(), 0, Normalization(name)).scores,
    )


@pytest.mark.parametrize("bad", ["", "!", "zzz", "min", "z_scorf"])
def test_from_ebm_rejects_every_unknown_normalization(bad: str) -> None:
    with pytest.raises(InvalidOptionError):
        EBMGraph.from_ebm(continuous_ebm(), 0, bad)


@pytest.mark.parametrize("bad", ["", "a", "zzz", "continuou", "nomina", "interaction"])
def test_from_ebm_rejects_every_unknown_feature_type(bad: str) -> None:
    ebm = continuous_ebm()
    ebm.feature_types_in_ = [bad]
    with pytest.raises(InvalidOptionError):
        EBMGraph.from_ebm(ebm, 0)


def test_from_ebm_use_feature_bounds_is_keyword_only() -> None:
    # a plain Callable lets the call break the signature, which is the point of the test
    from_ebm: Callable[..., EBMGraph] = EBMGraph.from_ebm
    with pytest.raises(TypeError):
        from_ebm(continuous_ebm(), 0, *(Normalization.NONE, False))


def test_from_ebm_large_graph() -> None:
    n = 300
    ebm = continuous_ebm()
    ebm.term_scores_ = [np.arange(n + 2, dtype=float)]
    ebm.standard_deviations_ = [np.zeros(n + 2)]
    ebm.bins_ = [[np.arange(n - 1, dtype=float)]]
    graph = EBMGraph.from_ebm(ebm, 0)
    assert len(graph.x_vals) == len(graph.scores) == n


@pytest.mark.parametrize("n_scores", [4, 6])
def test_from_ebm_mismatch_in_both_directions(n_scores: int) -> None:
    ebm = continuous_ebm()
    ebm.term_scores_ = [np.zeros(n_scores)]
    ebm.standard_deviations_ = [np.zeros(n_scores)]
    with pytest.raises(EBMGraphShapeError, match="does not match"):
        EBMGraph.from_ebm(ebm, 0)


# mutation-testing regressions: simplify


def test_simplify_upper_bound_of_variation_is_inclusive() -> None:
    g = cont_graph(scores=(1.0, 2.0, 5.0)).simplify(1.0)
    assert g.x_vals == [(0.0, 3.0)]


def test_simplify_threshold_scales_with_total_variation() -> None:
    # total variation is 3, threshold 0.9 * 3 = 2.7: the 2.5 step merges, the 3.0 step against the first bin does not
    g = cont_graph(scores=(2.0, 4.5, 5.0)).simplify(0.9)
    assert g.x_vals == [(0.0, 2.0), (2.0, 3.0)]
    assert g.scores == [2.0, 5.0]


def test_simplify_merges_into_the_last_bin_only() -> None:
    g = cont_graph(scores=(1.0, 5.0, 5.2), stds=(0.5, 0.9, 0.2)).simplify(0.1)
    assert g.x_vals == [(0.0, 1.0), (1.0, 3.0)]
    assert g.scores == [1.0, 5.0]
    assert g.stds == [0.5, 0.9]


def test_simplify_single_bin() -> None:
    g = EBMGraph("age", "continuous", [(0.0, 1.0)], [1.0], [0.1]).simplify(0.5)
    assert g.x_vals == [(0.0, 1.0)]


# mutation-testing regressions: to_text


def to_text_precision(variation: float) -> tuple[str, float]:
    mid = variation * 0.123456789
    graph = cont_graph(scores=(0.0, mid, variation))
    return graph.to_text(confidence_bounds=False), mid


@pytest.mark.parametrize(
    ("variation", "precision"),
    [(1000, 1), (100.5, 1), (100, 1), (99.5, 2), (99, 2), (9.11, 3), (0.5, 4), (0.3, 4), (0.12, 4)],
)
def test_to_text_auto_y_precision(variation: float, precision: int) -> None:
    text, mid = to_text_precision(variation)
    assert f'"(1.0, 2.0)": {round(mid, precision)},' in text
    assert f'"(2.0, 3.0)": {round(variation, precision)}' in text


def test_to_text_auto_keyword_equals_default() -> None:
    g = cont_graph(scores=(0.0, 0.0123456, 0.3))
    assert g.to_text(y_axis_precision="auto") == g.to_text()


def test_to_text_confidence_bounds_values() -> None:
    g = cont_graph(scores=(1.0, 2.0, 5.0), stds=(0.5, 0.25, 1.0))
    text = g.to_text(include_description=False, y_axis_precision=3)
    assert text.splitlines()[3:] == [
        'Lower Bounds (95%-Confidence Interval): {"(0.0, 1.0)": 0.02, "(1.0, 2.0)": 1.51, "(2.0, 3.0)": 3.04}',
        'Upper Bounds (95%-Confidence Interval): {"(0.0, 1.0)": 1.98, "(1.0, 2.0)": 2.49, "(2.0, 3.0)": 6.96}',
    ]


def test_to_text_confidence_level_changes_bounds() -> None:
    g = cont_graph(scores=(1.0, 2.0, 5.0), stds=(1.0, 1.0, 1.0))
    text = g.to_text(include_description=False, y_axis_precision=2, confidence_level=0.9)
    assert 'Lower Bounds (90%-Confidence Interval): {"(0.0, 1.0)": -0.64,' in text


def test_to_text_default_budget_arguments() -> None:
    seen = {}

    def check(**kw: object) -> baml_sdk.TokenBudget:
        seen.update(kw)
        return budget("ok")

    checked = fake_sdk("check_token_budget", check)
    text = cont_graph().to_text(check_budget=checked)
    assert (seen["max_tokens"], seen["overbudget_percent"]) == (3000, 50)
    assert seen["text"] == text
    cont_graph().to_text(max_tokens=7, overbudget_percent=9, check_budget=checked)
    assert (seen["max_tokens"], seen["overbudget_percent"]) == (7, 9)


def test_to_text_without_a_check_writes_the_graph_once(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = []
    simplify = EBMGraph.simplify

    def spy(self: EBMGraph, min_variation_per_cent: float = 0.0) -> EBMGraph:
        levels.append(min_variation_per_cent)
        return simplify(self, min_variation_per_cent)

    monkeypatch.setattr(EBMGraph, "simplify", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = cont_graph().to_text(max_tokens=1, overbudget_percent=0)
    assert "Feature Name: age" in text
    assert levels == [0.0]


def test_to_text_options_are_keyword_only() -> None:
    to_text: Callable[..., str] = cont_graph().to_text
    with pytest.raises(TypeError):
        to_text(*(True,))


def test_to_text_warning_points_at_the_caller() -> None:
    with pytest.warns(UserWarning, match="estimated at") as record:
        cont_graph().to_text(check_budget=checker("warn"))
    assert record[0].filename == __file__


def test_to_text_tries_every_simplification_level() -> None:
    calls: list[dict[str, Any]] = []
    with pytest.raises(EBMGraphSimplificationError, match="10%"):
        cont_graph().to_text(check_budget=fake_sdk("check_token_budget", appending(calls, budget("over", 5))))
    assert len(calls) == 101  # 0.0%, 0.1%, ..., 10.0%


@pytest.mark.parametrize(
    "x_vals",
    [["a", "b"], ["no", "yes"], ["True", "False"], ["False", "True", "x"], ["x", "False"]],
)
def test_to_text_not_boolean(x_vals: list[str]) -> None:
    g = EBMGraph("flag", "nominal", x_vals, [0.0] * len(x_vals), [0.1] * len(x_vals))
    assert "Feature Type: categorical" in g.to_text()


@pytest.mark.parametrize("x_vals", [["false", "true"], ["FALSE", "TRUE"], ["False", "True"]])
def test_to_text_boolean_case_insensitive(x_vals: list[str]) -> None:
    g = EBMGraph("flag", "nominal", x_vals, [0.0, 2.0], [0.1, 0.1])
    assert "Feature Type: boolean" in g.to_text()
    assert '"False": 0.0, "True": 2.0' in g.to_text(y_axis_precision=1)


def test_to_text_boolean_with_one_value() -> None:
    g = EBMGraph("flag", "nominal", ["a"], [0.0], [0.1])
    with pytest.raises(EBMGraphShapeError, match="boolean format"):
        g.to_text(feature_format=FeatureType.BOOLEAN)


def test_to_text_plain_string_feature_format() -> None:
    assert "Feature Type: categorical" in cat_graph().to_text(feature_format="categorical")


def test_to_text_unknown_feature_format() -> None:
    with pytest.raises(InvalidOptionError):
        cat_graph().to_text(feature_format="bogus")


def test_to_text_continuous_is_simplified() -> None:
    assert '"(0.0, 2.0)"' in cont_graph().to_text(confidence_bounds=False)  # first two bins are equal and merged


def test_to_text_categorical_is_not_simplified() -> None:
    g = EBMGraph("color", "nominal", ["a", "b"], [1.0, 1.0], [0.1, 0.1])
    assert '"a": 1.0, "b": 1.0' in g.to_text()


def test_simplify_rejects_non_continuous_string_variants() -> None:
    g = EBMGraph("x", "boolean", [(0.0, 1.0)], [1.0], [0.1])
    with pytest.raises(InvalidOptionError):
        g.simplify()


def test_to_text_auto_y_precision_depends_on_range_not_offset() -> None:
    # a score range of 0.5 around 1000 needs 4 decimals, whatever the offset
    g = cont_graph(scores=(1000.0, 1000.123456, 1000.5))
    assert '"(1.0, 2.0)": 1000.1235' in g.to_text(confidence_bounds=False)


def test_to_text_simplification_levels_grow_by_a_tenth_of_a_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = []
    simplify = EBMGraph.simplify

    def spy(self: EBMGraph, min_variation_per_cent: float = 0.0) -> EBMGraph:
        levels.append(min_variation_per_cent)
        return simplify(self, min_variation_per_cent)

    monkeypatch.setattr(EBMGraph, "simplify", spy)
    with pytest.raises(EBMGraphSimplificationError):
        cont_graph().to_text(check_budget=checker("over"))
    assert levels == pytest.approx([step * 0.001 for step in range(101)])


def test_to_text_stops_the_simplification_at_ten_per_cent(monkeypatch: pytest.MonkeyPatch) -> None:
    # regression: level 10.1% was tried before the error, above the documented cap of 10%
    levels = []
    simplify = EBMGraph.simplify

    def spy(self: EBMGraph, min_variation_per_cent: float = 0.0) -> EBMGraph:
        levels.append(min_variation_per_cent)
        return simplify(self, min_variation_per_cent)

    monkeypatch.setattr(EBMGraph, "simplify", spy)
    with pytest.raises(EBMGraphSimplificationError, match="10%"):
        cont_graph().to_text(check_budget=checker("over"))
    assert max(levels) == pytest.approx(0.1)


def round_trip_graph() -> EBMGraph:
    return EBMGraph(
        "age",
        "continuous",
        [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0)],
        [0.5, -0.25, 1.0],
        [0.1, 0.2, 0.3],
    )


def test_from_text_reads_a_confidence_level_between_whole_percents() -> None:
    # Hypothesis found it: the label said "88%" for 0.875, so from_text read the bounds at the wrong level
    graph = EBMGraph("age", "continuous", [(0.0, 1.0)], [0.0], [1.0])
    text = graph.to_text(y_axis_precision=2, confidence_level=0.875)
    assert "(87.5%-Confidence Interval)" in text
    assert EBMGraph.from_text(text).stds == pytest.approx([1.0], abs=0.01)


@pytest.mark.parametrize("confidence_level", [0.95, 0.9, 0.99])
def test_from_text_inverts_to_text(confidence_level: float) -> None:
    graph = round_trip_graph()
    text = graph.to_text(y_axis_precision=6, confidence_level=confidence_level)
    parsed = EBMGraph.from_text(text)
    assert parsed.feature_name == "age"
    assert parsed.feature_type == "continuous"
    assert parsed.x_vals == graph.x_vals
    assert parsed.scores == pytest.approx(graph.scores)
    assert parsed.stds == pytest.approx(graph.stds, abs=1e-5)


def test_to_text_writes_valid_json_for_category_names_with_quotes_and_backslashes() -> None:
    # regression: the Means and bounds lines were invalid JSON, so the LLM got malformed input
    names = ['Master "B"', "a\\b"]
    graph = EBMGraph("degree", "nominal", names, [0.5, -0.25], [0.1, 0.2])
    lines = graph.to_text(y_axis_precision=6).splitlines()
    json_lines = [line.split(": ", 1)[1] for line in lines if line.startswith(("Means", "Lower", "Upper"))]
    assert len(json_lines) == 3
    assert all(list(orjson.loads(line)) == names for line in json_lines)


def test_from_text_reads_a_graph_inside_a_prompt() -> None:
    graph_text = round_trip_graph().to_text(y_axis_precision=6)
    indented = "\n".join(f"    {line}" for line in graph_text.splitlines())
    parsed = EBMGraph.from_text(f"Look at this graph:\n\n{indented}\n\nWhat is the largest jump?")
    assert parsed.feature_name == "age"
    assert parsed.x_vals == round_trip_graph().x_vals
    assert parsed.stds == pytest.approx(round_trip_graph().stds, abs=1e-5)


@pytest.mark.parametrize("label", ["Feature Name", "Feature Type", "Means", "Lower Bounds", "Upper Bounds"])
def test_from_text_requires_every_line(label: str) -> None:
    lines = round_trip_graph().to_text(y_axis_precision=6).splitlines()
    text = "\n".join(line for line in lines if not line.startswith(label))
    with pytest.raises(GraphTextError, match=f"no '{label}' line"):
        EBMGraph.from_text(text)


@pytest.mark.parametrize("feature_type", ["nominal", "boolean", "bogus"])
def test_from_text_rejects_other_feature_types(feature_type: str) -> None:
    text = (
        round_trip_graph()
        .to_text(y_axis_precision=6)
        .replace("Feature Type: continuous", f"Feature Type: {feature_type}")
    )
    with pytest.raises(InvalidOptionError, match=feature_type):
        EBMGraph.from_text(text)


FIRST: str = "(0.0, 1.0)"
SECOND: str = "(1.0, 2.0)"


def graph_text_with_intervals(means: list[str], lower: list[str], upper: list[str]) -> str:
    def line(label: str, intervals: list[str]) -> str:
        return f"{label}: {{" + ", ".join(f'"{interval}": 0.5' for interval in intervals) + "}\n"

    return (
        "Feature Name: age\nFeature Type: continuous\n"
        + line("Means", means)
        + line("Lower Bounds (95%-Confidence Interval)", lower)
        + line("Upper Bounds (95%-Confidence Interval)", upper)
    )


@pytest.mark.parametrize(
    ("means", "lower", "upper"),
    [
        ([FIRST], [SECOND], [FIRST]),
        ([FIRST], [FIRST, SECOND], [FIRST, SECOND]),
        ([FIRST, SECOND], [FIRST], [FIRST]),
        ([FIRST], [FIRST], [FIRST, SECOND]),
        ([FIRST, SECOND], [FIRST, SECOND], [FIRST]),
        ([FIRST], [FIRST], [SECOND]),
    ],
)
def test_from_text_rejects_bounds_with_other_intervals(means: list[str], lower: list[str], upper: list[str]) -> None:
    with pytest.raises(GraphTextError, match="same intervals"):
        EBMGraph.from_text(graph_text_with_intervals(means, lower, upper))


def test_from_text_accepts_bounds_with_the_same_intervals_in_any_order() -> None:
    parsed = EBMGraph.from_text(graph_text_with_intervals([FIRST, SECOND], [SECOND, FIRST], [FIRST, SECOND]))
    assert parsed.x_vals == [(0.0, 1.0), (1.0, 2.0)]


def test_to_text_rejects_more_stds_than_scores() -> None:
    # more stds than scores: only the pairing of the bounds notices (the means have as many scores as bins)
    graph = EBMGraph("color", "nominal", ["a", "b"], [0.1, 0.2], [0.1, 0.1, 0.1])
    with pytest.raises(EBMGraphShapeError, match=r"scores of shape \(2,\) but stds of shape \(3,\)"):
        graph.to_text()


def test_the_budget_check_protocol_matches_the_sdk_function() -> None:
    assert call_mismatches(BudgetCheck, "check_token_budget") == []


def test_the_token_budget_protocol_matches_the_sdk_class() -> None:
    assert protocol_mismatches(TokenBudget, "TokenBudget", {}) == []


def test_baml_check_satisfies_the_budget_check_protocol() -> None:
    assert isinstance(baml_sdk.check_token_budget, BudgetCheck)
    assert isinstance(budget("ok"), TokenBudget)


def test_budget_check_protocol_takes_keyword_arguments_only() -> None:
    kinds = {name: parameter.kind for name, parameter in inspect.signature(BudgetCheck.__call__).parameters.items()}
    assert kinds == {
        "self": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "text": inspect.Parameter.KEYWORD_ONLY,
        "max_tokens": inspect.Parameter.KEYWORD_ONLY,
        "overbudget_percent": inspect.Parameter.KEYWORD_ONLY,
    }


def test_token_budget_protocol_members_are_read_only_properties() -> None:
    assert isinstance(TokenBudget.estimated_tokens, property)
    assert isinstance(TokenBudget.status, property)


type GraphParts = tuple[list[tuple[float, float]], list[float], list[float]]


# The bins, the finite scores and the standard deviations of a continuous graph, as from_ebm builds it. The strategy
# draws the parts and not the graph, because beartype samples with `random` when it checks the arguments of EBMGraph,
# and Hypothesis does not allow `random` inside a strategy.
@st.composite
def continuous_graph_parts(draw: st.DrawFn) -> GraphParts:
    finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_subnormal=False)
    cuts = sorted(draw(st.sets(finite, min_size=2, max_size=9)))
    if draw(st.booleans()):
        cuts = [-INF, *cuts[1:-1], INF]
    x_vals = list(itertools.pairwise(cuts))
    scores = draw(st.lists(finite, min_size=len(x_vals), max_size=len(x_vals)))
    stds = draw(st.lists(st.floats(min_value=0, max_value=100), min_size=len(x_vals), max_size=len(x_vals)))
    return x_vals, scores, stds


@given(
    parts=continuous_graph_parts(),
    precision=st.integers(min_value=0, max_value=8),
    confidence_level=st.floats(min_value=0.5, max_value=0.999),
)
def test_from_text_inverts_to_text_within_the_rounding(
    parts: GraphParts,
    precision: int,
    confidence_level: float,
) -> None:
    graph = EBMGraph("age", "continuous", *parts)
    text = graph.to_text(y_axis_precision=precision, confidence_level=confidence_level)
    parsed = EBMGraph.from_text(text)
    # to_text merges the neighboring bins of equal score first, and rounds each mean and each bound to `precision`
    graph = graph.simplify()
    step = 10.0**-precision
    assert (parsed.feature_name, parsed.feature_type, parsed.x_vals) == (graph.feature_name, "continuous", graph.x_vals)
    # a half step is the largest rounding error, and the float error of the rounding can add a tiny bit to it
    assert parsed.scores == pytest.approx(graph.scores, rel=1e-12, abs=step / 2 * (1 + 1e-9))
    assert parsed.stds == pytest.approx(graph.stds, rel=1e-6, abs=step)


@pytest.mark.parametrize("label", ["Means", "Lower Bounds", "Upper Bounds"])
def test_from_text_rejects_a_line_that_is_not_json(label: str) -> None:
    text = graph_text_with_intervals(["(0.0, 1.0)"], ["(0.0, 1.0)"], ["(0.0, 1.0)"])
    broken = "\n".join(
        f"{line[: line.index('{')]}{{not json" if line.startswith(label) else line for line in text.splitlines()
    )
    with pytest.raises(GraphTextError, match=f"'{label}' line of the graph text is not valid JSON") as caught:
        EBMGraph.from_text(broken)
    assert isinstance(caught.value.__cause__, orjson.JSONDecodeError)


# multiclass graphs


def multiclass_fake(
    scores: tuple[tuple[float, ...], ...] = (
        (9.0, 9.0, 9.0),
        (1.0, 2.0, 3.0),
        (4.0, 4.0, 1.0),
        (0.0, 3.0, 0.0),
        (9.0, 9.0, 9.0),
    ),
) -> ExplainableBoostingClassifier:
    return fake_model(
        feature_names_in_=["age"],
        feature_types_in_=["continuous"],
        term_features_=[(0,)],
        term_scores_=[np.array(scores)],
        standard_deviations_=[np.full((len(scores), 3), 0.1)],
        bins_=[[np.array([1.0, 2.0])]],
        feature_bounds_=[(0.0, 3.0)],
        link_="mlogit",
        classes_=np.array(["high", "low", "mid"]),
    )


def multiclass_graph() -> EBMGraph:
    return EBMGraph(
        "age",
        "continuous",
        [(0.0, 1.0), (1.0, 2.0)],
        [[-1.0, 0.0, 1.0], [0.5, 0.5, -1.0]],
        [[0.1, 0.2, 0.3], [0.1, 0.1, 0.1]],
        ["high", "low", "mid"],
    )


def test_from_ebm_of_a_multiclass_classifier_centres_each_bin() -> None:
    g = EBMGraph.from_ebm(multiclass_fake(), 0)
    assert g.classes == ["high", "low", "mid"]
    assert np.asarray(g.scores).tolist() == [[-1.0, 0.0, 1.0], [1.0, 1.0, -2.0], [-1.0, 2.0, -1.0]]
    assert np.asarray(g.stds).shape == (3, 3)


def test_from_ebm_of_a_fitted_multiclass_classifier(multiclass_ebm: ExplainableBoostingClassifier) -> None:
    g = EBMGraph.from_ebm(multiclass_ebm, 0)
    assert g.classes == ["high", "low", "mid"]
    assert np.asarray(g.scores).shape == (len(g.x_vals), 3)
    assert np.asarray(g.scores).sum(axis=1) == pytest.approx(0, abs=1e-9)


def test_from_ebm_of_a_one_output_model_has_no_classes() -> None:
    assert EBMGraph.from_ebm(continuous_ebm(), 0).classes == []


@given(
    raw=st.lists(
        st.tuples(*[st.floats(-10, 10)] * 3),
        min_size=3,
        max_size=6,
    ),
    intercept=st.tuples(*[st.floats(-5, 5)] * 3),
    bin_index=st.integers(0, 3),
)
def test_centring_keeps_the_softmax_of_every_bin(
    raw: list[tuple[float, float, float]],
    intercept: tuple[float, float, float],
    bin_index: int,
) -> None:
    ebm = multiclass_fake(((0.0, 0.0, 0.0), *raw, (0.0, 0.0, 0.0)))
    ebm.bins_ = [[np.arange(len(raw) - 1, dtype=float)]]
    graph = EBMGraph.from_ebm(ebm, 0, use_feature_bounds=False)
    row = bin_index % len(raw)
    kind = MulticlassClassifier(("high", "low", "mid"))
    centred = np.asarray(graph.scores)[row]
    assert kind.predict(np.add(intercept, centred)) == pytest.approx(kind.predict(np.add(intercept, raw[row])))


def test_from_ebm_selects_classes_in_the_given_order() -> None:
    g = EBMGraph.from_ebm(multiclass_fake(), 0, classes=["mid", "high", "mid"])
    assert g.classes == ["mid", "high"]
    # the scores are centred over every class before the selection
    assert np.asarray(g.scores).tolist() == [[1.0, -1.0], [-2.0, 1.0], [-1.0, -1.0]]
    assert np.asarray(g.stds).shape == (3, 2)


@pytest.mark.parametrize(("classes", "bad"), [(["low", "medium"], "medium"), ([], r"\[\]")])
def test_from_ebm_rejects_a_class_that_the_model_lacks(classes: list[str], bad: str) -> None:
    with pytest.raises(
        InvalidOptionError,
        match=rf"Invalid option: {bad}\. Valid options are \['high', 'low', 'mid'\]",
    ):
        EBMGraph.from_ebm(multiclass_fake(), 0, classes=classes)


def test_from_ebm_rejects_classes_for_a_model_with_one_output() -> None:
    with pytest.raises(UnsupportedModelError, match="classes needs a multiclass classifier, but the graph of age"):
        EBMGraph.from_ebm(continuous_ebm(), 0, classes=["1"])


@pytest.mark.parametrize(
    ("normalization", "expected"),
    [
        (Normalization.MIN_MAX, [[1.0, 2.0, 3.0], [3.0, 3.0, 0.0], [1.0, 4.0, 1.0]]),
        # the centred scores have a mean of zero, so the z-score shifts nothing
        (Normalization.Z_SCORE, [[-1.0, 0.0, 1.0], [1.0, 1.0, -2.0], [-1.0, 2.0, -1.0]]),
    ],
)
def test_from_ebm_normalizes_all_classes_by_one_constant(
    normalization: Normalization,
    expected: list[list[float]],
) -> None:
    # a shift of each class on its own changed the differences between the classes of a bin, which the text says carry
    # the meaning
    g = EBMGraph.from_ebm(multiclass_fake(), 0, normalization)
    assert np.asarray(g.scores) == pytest.approx(np.array(expected))


def test_simplify_merges_a_multiclass_bin_only_if_every_class_is_flat() -> None:
    graph = EBMGraph(
        "age",
        "continuous",
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        [[0.0, 0.0], [0.0, 0.1], [0.0, 2.0], [0.1, 2.0]],
        [[0.1, 0.3], [0.2, 0.1], [0.1, 0.1], [0.1, 0.4]],
        ["a", "b"],
    )
    # the threshold is 5% of the range 2.0: the second bin moves class b by 0.1, which stays within it
    g = graph.simplify(0.05)
    assert g.x_vals == [(0.0, 2.0), (2.0, 4.0)]
    assert g.scores == [[0.0, 0.0], [0.0, 2.0]]
    assert g.stds == [[0.2, 0.3], [0.1, 0.4]]
    assert g.classes == ["a", "b"]
    # with no tolerance, a change in one class keeps the bins apart
    assert graph.simplify().x_vals == graph.x_vals


def test_to_text_of_a_multiclass_graph_has_one_series_per_class() -> None:
    text = multiclass_graph().to_text(y_axis_precision=1)
    assert text == (
        "This graph represents a continuous-valued feature. The keys are intervals that represent ranges where the"
        " function predicts the same value.\n\n"
        "The model is a multiclass classifier, so the graph has one series of means and bounds for each class. A line"
        " 'Class: <name>' starts the series of each class. In every bin, the scores have the same mean over all the"
        " classes of the model, so only the differences between the classes change the predicted probabilities.\n\n"
        "Feature Name: age\n"
        "Feature Type: continuous\n"
        "Class: high\n"
        'Means: {"(0.0, 1.0)": -1.0, "(1.0, 2.0)": 0.5}\n'
        'Lower Bounds (95%-Confidence Interval): {"(0.0, 1.0)": -1.2, "(1.0, 2.0)": 0.3}\n'
        'Upper Bounds (95%-Confidence Interval): {"(0.0, 1.0)": -0.8, "(1.0, 2.0)": 0.7}\n'
        "Class: low\n"
        'Means: {"(0.0, 1.0)": 0.0, "(1.0, 2.0)": 0.5}\n'
        'Lower Bounds (95%-Confidence Interval): {"(0.0, 1.0)": -0.4, "(1.0, 2.0)": 0.3}\n'
        'Upper Bounds (95%-Confidence Interval): {"(0.0, 1.0)": 0.4, "(1.0, 2.0)": 0.7}\n'
        "Class: mid\n"
        'Means: {"(0.0, 1.0)": 1.0, "(1.0, 2.0)": -1.0}\n'
        'Lower Bounds (95%-Confidence Interval): {"(0.0, 1.0)": 0.4, "(1.0, 2.0)": -1.2}\n'
        'Upper Bounds (95%-Confidence Interval): {"(0.0, 1.0)": 1.6, "(1.0, 2.0)": -0.8}\n'
    )


def test_to_text_of_a_multiclass_graph_without_bounds_or_preamble() -> None:
    text = multiclass_graph().to_text(y_axis_precision=1, confidence_bounds=False, include_description=False)
    assert text.splitlines() == [
        "Feature Name: age",
        "Feature Type: continuous",
        "Class: high",
        'Means: {"(0.0, 1.0)": -1.0, "(1.0, 2.0)": 0.5}',
        "Class: low",
        'Means: {"(0.0, 1.0)": 0.0, "(1.0, 2.0)": 0.5}',
        "Class: mid",
        'Means: {"(0.0, 1.0)": 1.0, "(1.0, 2.0)": -1.0}',
    ]


def test_to_text_rejects_multiclass_stds_of_another_shape() -> None:
    graph = attrs.evolve(multiclass_graph(), stds=[[0.1, 0.1], [0.1, 0.1]])
    with pytest.raises(EBMGraphShapeError, match=r"scores of shape \(2, 3\) but stds of shape \(2, 2\)"):
        graph.to_text()


def test_from_text_inverts_the_text_of_a_multiclass_graph() -> None:
    graph = multiclass_graph()
    parsed = EBMGraph.from_text(graph.to_text(y_axis_precision=6))
    assert parsed.classes == ["high", "low", "mid"]
    assert parsed.x_vals == graph.x_vals
    assert np.asarray(parsed.scores) == pytest.approx(np.asarray(graph.scores))
    assert np.asarray(parsed.stds) == pytest.approx(np.asarray(graph.stds), abs=1e-5)


type MulticlassParts = tuple[list[tuple[float, float]], list[list[float]], list[list[float]], list[str]]


@st.composite
def multiclass_graph_parts(draw: st.DrawFn) -> MulticlassParts:
    x_vals, _, _ = draw(continuous_graph_parts())
    n_classes = draw(st.integers(min_value=1, max_value=4))
    finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_subnormal=False)
    row = st.lists(finite, min_size=n_classes, max_size=n_classes)
    std_row = st.lists(st.floats(min_value=0, max_value=100), min_size=n_classes, max_size=n_classes)
    scores = draw(st.lists(row, min_size=len(x_vals), max_size=len(x_vals)))
    stds = draw(st.lists(std_row, min_size=len(x_vals), max_size=len(x_vals)))
    name = st.text("abcxyz _-", min_size=1, max_size=6)
    names = draw(st.lists(name, min_size=n_classes, max_size=n_classes, unique=True))
    return x_vals, scores, stds, names


@given(parts=multiclass_graph_parts(), precision=st.integers(min_value=0, max_value=8))
def test_from_text_inverts_a_multiclass_text_within_the_rounding(
    parts: MulticlassParts,
    precision: int,
) -> None:
    graph = EBMGraph("age", "continuous", *parts)
    names = parts[3]
    parsed = EBMGraph.from_text(graph.to_text(y_axis_precision=precision))
    graph = graph.simplify()
    step = 10.0**-precision
    assert (parsed.x_vals, parsed.classes) == (graph.x_vals, names)
    assert np.asarray(parsed.scores) == pytest.approx(np.asarray(graph.scores), rel=1e-12, abs=step / 2 * (1 + 1e-9))
    assert np.asarray(parsed.stds) == pytest.approx(np.asarray(graph.stds), rel=1e-6, abs=step)


def test_from_ebm_rejects_one_string_as_the_classes() -> None:
    # a string is a sequence of strings, so "high" selected the classes "h", "i", "g" and "h" one letter at a time
    with pytest.raises(InvalidOptionError, match=r"classes='high' \(pass a list of class names\)"):
        EBMGraph.from_ebm(multiclass_fake(), 0, classes="high")


def test_from_ebm_reads_the_outputs_from_the_model_kind() -> None:
    ebm = multiclass_fake()
    ebm.link_ = "probit"
    with pytest.raises(UnsupportedModelError, match="EBaMl reads the links"):
        EBMGraph.from_ebm(ebm, 0)


@pytest.mark.parametrize(
    ("ebm", "shape", "outputs"),
    [(multiclass_fake(((1.0, 2.0),) * 5), r"\(3, 2\)", 3), (continuous_ebm(), r"\(3,\)", 3)],
    ids=["two columns for three classes", "one column for three classes"],
)
def test_from_ebm_rejects_scores_that_do_not_fit_the_outputs(
    ebm: ExplainableBoostingClassifier,
    shape: str,
    outputs: int,
) -> None:
    ebm.link_, ebm.classes_ = "mlogit", np.array(["a", "b", "c"])
    with pytest.raises(EBMGraphShapeError, match=rf"have the shape {shape}, but the model has {outputs} outputs"):
        EBMGraph.from_ebm(ebm, 0)


def test_from_ebm_rejects_columns_of_scores_for_a_model_with_one_output() -> None:
    ebm = multiclass_fake()
    ebm.link_, ebm.classes_ = "logit", np.array([0, 1])
    with pytest.raises(EBMGraphShapeError, match=r"have the shape \(3, 3\), but the model has 1 outputs"):
        EBMGraph.from_ebm(ebm, 0)


@pytest.mark.parametrize(
    ("scores", "classes", "shape"),
    [
        ([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], ["a", "b"], r"\(2, 3\) for 2 classes"),
        ([0.0, 1.0], ["a", "b"], r"\(2,\) for 2 classes"),
        ([[0.0, 1.0], [0.0, 1.0]], [], r"\(2, 2\) for 0 classes"),
    ],
    ids=["a column too many", "no columns", "columns without classes"],
)
def test_to_text_rejects_scores_that_do_not_fit_the_classes(
    scores: list[list[float]] | list[float],
    classes: list[str],
    shape: str,
) -> None:
    stds = np.zeros_like(np.asarray(scores)).tolist()
    graph = EBMGraph("age", "continuous", [(0.0, 1.0), (1.0, 2.0)], scores, stds, classes)
    with pytest.raises(EBMGraphShapeError, match=rf"has scores of shape {shape}"):
        graph.to_text()


def test_to_text_without_bounds_ignores_the_confidence_level() -> None:
    # the factor of the bounds was computed for every text, so a level that only the bounds use raised without them
    text = cont_graph().to_text(confidence_bounds=False, confidence_level=95)
    assert "Bounds" not in text


# terms that are not in the order of the features


def test_from_ebm_reads_the_term_of_the_feature_when_another_feature_has_no_term() -> None:
    # with exclude or remove_terms, term 0 is the feature "b"; the graph of "b" read term 1, which does not exist
    ebm = fake_model(
        feature_names_in_=["a", "b"],
        feature_types_in_=["continuous", "nominal"],
        term_features_=[(1,)],
        term_scores_=[np.array([0.0, 1.0, 3.0, 0.0])],
        standard_deviations_=[np.array([0.0, 0.1, 0.2, 0.0])],
        bins_=[[], [{"x": 1, "y": 2}]],
        link_="logit",
        classes_=np.array([0, 1]),
    )
    graph = EBMGraph.from_ebm(ebm, 1)
    assert graph.feature_name == "b"
    assert graph.x_vals == ["x", "y"]
    assert np.asarray(graph.scores).tolist() == [1.0, 3.0]
    assert np.asarray(graph.stds).tolist() == [0.1, 0.2]
    with pytest.raises(UnsupportedModelError, match="'a' has no main-effect term"):
        EBMGraph.from_ebm(ebm, 0)
