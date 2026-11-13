"""Tests for the counterfactual of a prediction."""

import itertools
from collections.abc import Callable, Sequence

import attrs
import numpy as np
import orjson
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ebaml.core.custom_types import FeatureValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import OptionRangeError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.local import counterfactuals as counterfactuals_module
from ebaml.local import fact_features as fact_features_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local.audience import DataScientist
from ebaml.local.covering_backend import Stop
from ebaml.local.covering_solvers import CoveringSolver, GreedyCover, LocalSearchCover, MipCover
from ebaml.local.decisions import Threshold
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.local_kinds import MulticlassLocal
from ebaml.local.minimality import MinimalityStatus
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver
from tests.unit.ebaml.conftest import (
    FittedClassifier,
    FittedRegressor,
    PandasFrame,
    frame_of_matrix,
    frame_of_rows,
    instance_of,
)
from tests.unit.ebaml.local.conftest import (
    AT_BOUND,
    FEATURES,
    INTERCEPT,
    SORTING,
    TOLERANCE,
    UNPROVED,
    FeatureParts,
    SmallModel,
    Templates,
    Z,
    build,
    build_ebm,
    hand,
    multiclass_model,
    probability_after,
    scored_feature,
    sig,
    sign_of,
)


def test_counterfactual_really_flips(categorical_ebm: FittedClassifier) -> None:
    ebm, _ = categorical_ebm
    row = {"f0": "p", "f1": "p", "f2": "p"}
    facts = facts_module.compute_local_facts(
        ebm,
        row,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0),
    )
    assert facts.counterfactual is not None
    changed = {**row, **{c.feature: c.to_value for c in facts.counterfactual.changes}}
    assert ebm.predict(frame_of_rows([changed]))[0] != ebm.predict(frame_of_rows([row]))[0]


def test_counterfactual_is_none_when_no_change_flips() -> None:
    features = [scored_feature([1.0, 0.9], 0), scored_feature([1.0, 0.8], 0)]
    assert (
        counterfactuals_module._counterfactual(
            features,
            intercept=0.0,
            sign=1,
            z=0.0,
            after=probability_after,
        )
        is None
    )


def test_counterfactual_skips_unplausible_bins() -> None:
    features = [scored_feature([1.0, -5.0], 0, plausible=[True, False]), scored_feature([1.0, -5.0], 0)]
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept=0.0,
        sign=1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert [c.feature for c in counterfactual.changes] == ["f"]
    assert len(counterfactual.changes) == 1


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239))
def test_multiclass_counterfactual_changes_the_predicted_class(row: int) -> None:
    model, frame = multiclass_model()
    instance = instance_of(frame, row)
    facts = facts_module.compute_local_facts(
        model,
        instance,
        fact_types_module.LocalOptions(confidence_level=0.0, min_support=0.0),
    )
    if facts.counterfactual is None:
        return
    changed = {
        **instance,
        **{change.feature: float(change.to_value.strip("[)").split(",")[0]) for change in facts.counterfactual.changes},
    }
    after = model.predict_proba(frame_of_rows([changed]))[0]
    assert model.classes_[int(np.argmax(after))] == facts.counterfactual.new_decision
    assert facts.counterfactual.new_decision != facts.prediction.decision
    assert facts.counterfactual.new_output == pytest.approx(float(after.max()), abs=1e-6)


def _changed_row(instance: dict[str, FeatureValue], counterfactual: fact_types_module.Counterfactual) -> PandasFrame:
    """Return the instance with each changed feature set to the lower end of its new bin.

    Returns
    -------
    PandasFrame
        One row.

    """
    changed = {
        **instance,
        **{change.feature: float(change.to_value.strip("[)").split(",")[0]) for change in counterfactual.changes},
    }
    return frame_of_rows([changed])


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239), target=st.sampled_from([0, 1, 2]))
def test_a_target_counterfactual_reaches_the_target_class(row: int, target: int) -> None:
    model, frame = multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = fact_types_module.LocalOptions(
        confidence_level=0.0,
        min_support=0.0,
        target_class=int(model.classes_[target]),
    )
    facts = facts_module.compute_local_facts(model, instance, options)
    if facts.counterfactual is None:
        return
    counterfactual = facts.counterfactual
    assert counterfactual.new_decision == model.classes_[target]
    assert model.predict(_changed_row(instance, counterfactual))[0] == model.classes_[target]
    assert 1 <= len(counterfactual.changes) <= 3


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239), target=st.sampled_from([0, 1, 2]))
def test_a_target_counterfactual_has_no_feature_to_revert(row: int, target: int) -> None:
    model, frame = multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = fact_types_module.LocalOptions(
        confidence_level=0.0,
        min_support=0.0,
        target_class=int(model.classes_[target]),
    )
    counterfactual = facts_module.compute_local_facts(model, instance, options).counterfactual
    assume(counterfactual is not None)
    assert counterfactual is not None
    for change in counterfactual.changes:
        fewer = attrs.evolve(counterfactual, changes=[other for other in counterfactual.changes if other != change])
        assert model.predict(_changed_row(instance, fewer))[0] != model.classes_[target]


def _brute_force_target_size(
    features: Sequence[fact_features_module.Feature],
    scores: Sequence[float],
    target: int,
) -> int | None:
    """Try every set of features and every plausible bin for each, and return the fewest changes that make a class win.

    Returns
    -------
    int | None
        The fewest changes, or ``None`` if no choice of bins makes the target class win.

    """
    feature_indexes = [index for index, _ in enumerate(features)]
    for size in range(1, len(features) + 1):
        for chosen in itertools.combinations(feature_indexes, size):
            bin_choices = [
                [
                    int(bin_index)
                    for bin_index in np.flatnonzero(features[index].plausible)
                    if bin_index != features[index].bin_index
                ]
                for index in chosen
            ]
            for bins in itertools.product(*bin_choices):
                new_scores = np.array(scores, dtype=float)
                for index, bin_index in zip(chosen, bins, strict=True):
                    new_scores += features[index].outputs(bin_index) - features[index].outputs(
                        features[index].bin_index,
                    )
                if all(new_scores[target] > new_scores[rival] for rival, _ in enumerate(scores) if rival != target):
                    return size
    return None


TARGET_SEARCHES: list[CoveringSolver] = [
    MipCover(),
    MipCover(OrToolsBackend(OrToolsSolver.SCIP)),
    MipCover(OrToolsBackend(OrToolsSolver.CP_SAT)),
    MipCover(stop=Stop.FIRST),
    LocalSearchCover(),
    GreedyCover(),
]


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239), target=st.sampled_from([0, 1, 2]), search=st.sampled_from(TARGET_SEARCHES))
def test_a_target_counterfactual_is_the_smallest_over_every_choice_of_bins(
    row: int,
    target: int,
    search: CoveringSolver,
) -> None:
    model, frame = multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = fact_types_module.LocalOptions(
        confidence_level=0.0,
        min_support=0.0,
        target_class=int(model.classes_[target]),
        search=search,
    )
    counterfactual = facts_module.compute_local_facts(model, instance, options).counterfactual
    features = fact_features_module.read_features(model, instance, 0.0)
    logits: np.ndarray = model.decision_function(frame.iloc[[row]])
    scores: list[float] = logits[0].tolist()
    smallest = _brute_force_target_size(features, scores, target)
    if smallest is None:
        assert counterfactual is None
        return
    assert counterfactual is not None
    assert model.predict(_changed_row(instance, counterfactual))[0] == model.classes_[target]
    assert len(counterfactual.changes) >= smallest
    assert counterfactual.lower_bound <= smallest
    if counterfactual.minimality.is_smallest:
        assert len(counterfactual.changes) == smallest
    if isinstance(search, MipCover) and search.stop is Stop.OPTIMAL:
        assert counterfactual.minimality.is_smallest
        assert len(counterfactual.changes) == smallest


def _target_feature(name: str, rows: list[list[float]]) -> fact_features_module.Feature:
    # a feature of a three-class model with one row of scores for each bin, the instance in bin 0, and no deviation
    graph = EBMGraph(
        name,
        "nominal",
        [f"{name}{bin_index}" for bin_index, _ in enumerate(rows)],
        rows,
        [[0.0] * 3] * len(rows),
        classes=("c0", "c1", "c2"),
    )
    return fact_features_module.Feature(name, graph, np.ones(len(rows)), 0, f"{name}0", np.ones(len(rows), dtype=bool))


@pytest.mark.parametrize("solver", list(OrToolsSolver))
def test_a_target_counterfactual_finds_one_change_where_the_bin_with_the_largest_sum_needs_two(
    solver: OrToolsSolver,
) -> None:
    # Bin 1 of f0 alone makes c0 win against both rivals. Bin 2 of f0 has the larger sum of gains over the two margins
    # (13 against 12), and it covers neither margin alone, so a search that keeps only that bin needs f1 as well. The
    # covering problem has every bin as an alternative, so the search finds the one change and proves it smallest.
    features = [
        _target_feature("f0", [[0, 0, 0], [10, 0, 8], [6, 0, -1]]),
        _target_feature("f1", [[0, 0, 0], [4, 0, 4]]),
    ]
    intercepts = [0.0, 9.0, 1.0]
    scores = [intercepts[output] + sum(f.outputs(0)[output] for f in features) for output in range(3)]
    kind = MulticlassLocal(("c0", "c1", "c2"))
    frame = kind.frame(scores, [0.0] * 3, intercepts=intercepts, confidence_level=0.0, decision=None)
    margins = kind.target_margins("c0", frame)
    totals = counterfactuals_module.Totals(scores, intercepts, 0.0)
    assert _brute_force_target_size(features, scores, 0) == 1
    options = fact_types_module.LocalOptions(search=MipCover(OrToolsBackend(solver)))
    counterfactual = counterfactuals_module._target_counterfactual(features, frame, margins, totals, options=options)
    assert counterfactual is not None
    assert counterfactual.new_decision == "c0"
    assert [(change.feature, change.to_value) for change in counterfactual.changes] == [("f0", "f01")]
    # one change reaches the lower bound of 1, so no proof of the solver is needed
    assert counterfactual.minimality is MinimalityStatus.AT_LOWER_BOUND
    assert counterfactual.lower_bound == 1


def test_a_target_class_is_none_when_no_choice_of_bins_reaches_it() -> None:
    model, frame = multiclass_model()
    row = 0
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    target = (predicted + 1) % 3
    options = fact_types_module.LocalOptions(
        confidence_level=0.999999,
        min_support=0.0,
        target_class=int(model.classes_[target]),
    )
    facts = facts_module.compute_local_facts(model, instance_of(frame, row), options)
    assert facts.counterfactual is None


def test_a_target_class_that_the_model_does_not_have_is_rejected() -> None:
    model, frame = multiclass_model()
    with pytest.raises(OptionRangeError, match="is not a class of the model"):
        facts_module.compute_local_facts(model, instance_of(frame, 0), fact_types_module.LocalOptions(target_class=99))


def test_a_binary_classifier_rejects_a_target_class(fit_ebm: Callable[..., ExplainableBoostingClassifier]) -> None:
    rng = np.random.default_rng(0)
    frame = frame_of_matrix(rng.normal(size=(300, 2)), ["a", "b"])
    ebm = fit_ebm(frame, (frame["a"] > 0).astype(int), interactions=0, random_state=0)
    with pytest.raises(OptionRangeError, match="needs a multiclass classifier"):
        facts_module.compute_local_facts(ebm, instance_of(frame, 0), fact_types_module.LocalOptions(target_class=1))


def test_counterfactual_changes_the_biggest_drop_first() -> None:
    features = [hand("f1", [1.0, -2.0]), hand("f2", [0.5, -0.25])]
    # the margin is 0.5 + 1 + 0.5 = 2; f1 can drop by 3, f2 by 0.75
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept=0.5,
        sign=1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.changes == [fact_types_module.Change("f1", "f10", "f11", -3.0)]
    # the new logit is 0.5 + 1.5 - 3 = -1, so the new class (negative) has the probability sigmoid(1)
    assert counterfactual.new_output == pytest.approx(sig(1.0))


def test_counterfactual_of_the_negative_class() -> None:
    features = [hand("f1", [-1.0, 2.0]), hand("f2", [-0.5, 0.25])]
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept=-0.5,
        sign=-1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.changes == [fact_types_module.Change("f1", "f10", "f11", 3.0)]
    assert counterfactual.new_output == pytest.approx(sig(1.0))  # the new logit is +1


def test_counterfactual_uses_as_many_changes_as_it_takes() -> None:
    features = [hand("f1", [1.0, 0.0]), hand("f2", [1.0, 0.0]), hand("f3", [0.5, -1.0])]
    # the margin is 2.5 and the drops are 1, 1 and 1.5: the margin is 1 and then 0 (a tie, not a flip) and then -1
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept=0.0,
        sign=1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f3", "f1", "f2"]


def test_counterfactual_with_two_changes() -> None:
    features = [hand("f1", [1.0, 0.0]), hand("f2", [1.0, -0.5]), hand("f3", [1.0, 0.5])]
    # the margin is 3 and the drops are 1, 1.5 and 0.5: after the two largest drops it is 0.5, so it is not yet flipped
    assert (
        counterfactuals_module._counterfactual(
            features,
            intercept=0.0,
            sign=1,
            z=0.0,
            after=probability_after,
        )
        is None
    )
    features[0] = hand("f1", [1.0, -3.0])
    # the drops are now 4, 1.5 and 0.5, and the margin 3 flips after the first; with a margin of 4.5 it takes two
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept=1.5,
        sign=1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f1", "f2"]


def test_counterfactual_probability_of_the_new_class() -> None:
    # the margin is 2.5 and the drop is 4, so the new logit is -1.5 and the new (negative) class has sigmoid(1.5)
    counterfactual = counterfactuals_module._counterfactual(
        [hand("f", [1.0, -3.0])],
        intercept=1.5,
        sign=1,
        z=0.0,
        after=probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.new_output == pytest.approx(sig(1.5))


def test_counterfactual_is_none_for_a_tie() -> None:
    assert (
        counterfactuals_module._counterfactual(
            [hand("f", [1.0, 0.0])],
            intercept=0.0,
            sign=1,
            z=0.0,
            after=probability_after,
        )
        is None
    )
    # the margin counts the score of the feature itself, not just its sign
    assert (
        counterfactuals_module._counterfactual(
            [hand("f", [2.0, 0.0])],
            intercept=0.0,
            sign=1,
            z=0.0,
            after=probability_after,
        )
        is None
    )


def test_counterfactual_asks_for_the_worst_case_of_the_other_class() -> None:
    feature = hand("f", [1.0, -2.0], stds=[0.5, 0.5])
    # with z = 1 the scores are raised by 0.5, so the drop is 1 + 1.5 = 2.5, and the margin 2.8 is not enough
    assert (
        counterfactuals_module._counterfactual(
            [feature],
            intercept=1.8,
            sign=1,
            z=1.0,
            after=probability_after,
        )
        is None
    )


@given(parts=FEATURES, intercept=INTERCEPT, z=Z)
def test_the_changes_of_a_counterfactual_flip_the_prediction(
    parts: list[FeatureParts],
    intercept: float,
    z: float,
) -> None:
    features = build(parts)
    sign = sign_of(features, intercept)
    counterfactual = counterfactuals_module._counterfactual(
        features,
        intercept,
        sign,
        z,
        after=probability_after,
    )
    assume(counterfactual is not None)
    assert counterfactual is not None
    new_bins = {change.feature: int(change.to_value) for change in counterfactual.changes}
    # the margin of the outcome after the changes, with each changed score at its most unfavourable value
    margin = sign * intercept
    for feature in features:
        if feature.name in new_bins:
            new = new_bins[feature.name]
            assert feature.plausible[new]
            margin += sign * float(feature.scores[new]) + z * float(feature.stds[new])
        else:
            margin += sign * feature.current
    assert margin < TOLERANCE


def test_regressor_counterfactual_gives_the_new_value(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, _ = categorical_regressor
    row = {"f0": "p", "f1": "p", "f2": "p"}
    facts = facts_module.compute_local_facts(
        ebm,
        row,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0, decision=Threshold(0.5)),
    )
    assert facts.counterfactual is not None
    assert facts.counterfactual.new_decision == "at most 0.5"
    changed = {**row, **{c.feature: c.to_value for c in facts.counterfactual.changes}}
    new_value = float(ebm.predict(frame_of_rows([changed]))[0])
    assert facts.counterfactual.new_output == pytest.approx(new_value)
    assert new_value <= 0.5


# A classifier with three classes. The instance has class 0, and class 2 needs a change of both features.
TWO_CHANGES_MODEL: SmallModel = SmallModel(
    kind="multiclass",
    scores=[
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
    ],
    deviations=[[[0.0] * 3 for _ in range(2)], [[0.0] * 3 for _ in range(3)]],
    weights=[[1.0, 1.0], [1.0, 1.0, 1.0]],
    intercepts=[1 / 16, 2 / 16, 3 / 16],
    instance_bins=[0, 0],
)


@pytest.mark.parametrize(
    ("search", "lower_bound", "status"),
    [
        pytest.param(MipCover(), 2, AT_BOUND, id="mip-optimal"),
        pytest.param(GreedyCover(), 2, AT_BOUND, id="greedy-at-the-bound"),
        pytest.param(GreedyCover(bound_backend=None), 1, UNPROVED, id="greedy-no-bound"),
        pytest.param(LocalSearchCover(bound_backend=None), 1, UNPROVED, id="local-search-no-bound"),
    ],
)
def test_a_counterfactual_of_a_target_class_reports_the_guarantee_of_its_search(
    small_templates: Templates,
    search: CoveringSolver,
    lower_bound: int,
    status: MinimalityStatus,
) -> None:
    ebm = build_ebm(small_templates, TWO_CHANGES_MODEL)
    options = fact_types_module.LocalOptions(confidence_level=0, min_support=0, target_class=2, search=search)
    facts = facts_module.compute_local_facts(ebm, TWO_CHANGES_MODEL.instance, options)
    counterfactual = facts.counterfactual
    assert counterfactual is not None
    assert counterfactual.new_decision == 2
    assert (len(counterfactual.changes), counterfactual.lower_bound, counterfactual.minimality) == (
        2,
        lower_bound,
        status,
    )
    expert = orjson.loads(facts_to_texts(facts, DataScientist()).counterfactual)
    assert expert["is_smallest_set"] is status.is_smallest
    assert expert["smallest_possible_number_of_changes"] == lower_bound
    # the default counterfactual of the same instance is one margin at a time, so sorting proves it
    default = facts_module.compute_local_facts(
        ebm,
        TWO_CHANGES_MODEL.instance,
        attrs.evolve(options, target_class=None),
    )
    assert default.counterfactual is not None
    assert default.counterfactual.minimality is SORTING
