"""Tests for the sufficient reason of a prediction."""

import itertools
from collections.abc import Callable

import numpy as np
import orjson
import pytest
from hypothesis import assume, given

from ebaml.local import completions as completions_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local import sufficient_reasons as sufficient_reasons_module
from ebaml.local.audience import DataScientist
from ebaml.local.covering_backend import CoveringModel, ModelResult, ModelStatus
from ebaml.local.covering_solvers import CoveringSolver, GreedyCover, LocalSearchCover, MipCover
from ebaml.local.decisions import Interval, Threshold
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.minimality import MinimalityStatus
from tests.unit.ebaml.conftest import (
    FittedClassifier,
    frame_of_rows,
    instance_of,
)
from tests.unit.ebaml.local.conftest import (
    AT_BOUND,
    FEATURES,
    INTERCEPT,
    NO_LP,
    SORTING,
    TOLERANCE,
    UNPROVED,
    FeatureParts,
    FittedModel,
    ScriptedBackend,
    SmallModel,
    Z,
    build,
    build_ebm,
    fit_template,
    hand,
    scored_feature,
    sign_of,
)


def test_sufficient_reason_decides_outcome(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 0)
    facts = facts_module.compute_local_facts(
        ebm,
        row,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0),
    )
    assert facts.sufficient_reason is not None
    free = [n for n in row if n not in facts.sufficient_reason.features]
    rng = np.random.default_rng(0)
    for _ in range(50):
        other = data.iloc[int(rng.integers(len(data)))].to_dict()
        mixed = {**row, **{n: other[n] for n in free}}
        assert ebm.predict(frame_of_rows([mixed]))[0] == facts.prediction.decision


def test_sufficient_reason_is_none_when_not_even_all_features_guarantee() -> None:
    features = [scored_feature([0.5, -2.0], 0, std=1.0), scored_feature([0.5, -2.0], 0, std=1.0)]
    assert sufficient_reasons_module._sufficient_reason(features, intercept=0.0, sign=1, z=1.96) is None


def test_fixing_order_counts_what_a_known_value_guarantees() -> None:
    first = hand("f1", [2.0, -1.0], stds=[1.0, 0.0])
    second = hand("f2", [1.5, 0.0])
    # at z = 1 the guaranteed scores of f1 are 1 and -1, so knowing its value gains 2; f2 gains 1.5
    order = sufficient_reasons_module.fixing_order([second, first], sign=1, z=1.0)
    assert [(step.gain, step.feature.name) for step in order] == [(2.0, "f1"), (1.5, "f2")]


def test_sufficient_reason_fixes_the_largest_gains_first() -> None:
    features = [hand("f3", [0.5, 0.0]), hand("f1", [2.0, -1.0]), hand("f2", [1.0, -0.5])]
    # the worst case is -1.5 (-1, -0.5 and 0); the gains are 3 (f1), 1.5 (f2) and 0.5 (f3)
    reason = sufficient_reasons_module._sufficient_reason(features, intercept=0.0, sign=1, z=0.0)
    assert reason == fact_types_module.SufficientReason(
        ["f1"],
        1.5,
        minimality=MinimalityStatus.SORTING_PROOF,
        lower_bound=1,
    )


def test_sufficient_reason_needs_no_feature_when_the_outcome_is_certain() -> None:
    # even the lowest score keeps the margin above 0, so no input has to be fixed
    assert sufficient_reasons_module._sufficient_reason(
        [hand("f", [1.0, 2.0])],
        0.0,
        1,
        0.0,
    ) == fact_types_module.SufficientReason(
        [],
        1.0,
        minimality=MinimalityStatus.SORTING_PROOF,
        lower_bound=0,
    )


def test_sufficient_reason_needs_a_margin_above_zero() -> None:
    # the worst case is exactly 0: knowing the value gains 1, and the margin is 0 before and 1 after
    assert sufficient_reasons_module._sufficient_reason(
        [hand("f", [0.0, 1.0], 1)],
        0.0,
        1,
        0.0,
    ) == fact_types_module.SufficientReason(
        ["f"],
        1.0,
        minimality=MinimalityStatus.SORTING_PROOF,
        lower_bound=1,
    )
    # when the margin stays at 0, the outcome is not certain
    assert sufficient_reasons_module._sufficient_reason([hand("f", [0.0, 0.0])], 0.0, 1, 0.0) is None


@pytest.mark.parametrize("max_steps", [1, 3])
def test_completions_reveal_max_steps_features(max_steps: int) -> None:
    features = [hand(f"f{i}", [1.0, 1.0]) for i in range(4)]
    options = fact_types_module.LocalOptions(n_samples=5, max_steps=max_steps)
    steps = completions_module.sample_completions(
        features,
        features,
        [sufficient_reasons_module.Bound(0.0, 1)],
        options,
    )
    assert [len(step.fixed) for step in steps] == list(range(max_steps + 1))


@given(parts=FEATURES, intercept=INTERCEPT, z=Z)
def test_the_features_of_a_sufficient_reason_fix_the_outcome_for_every_completion(
    parts: list[FeatureParts],
    intercept: float,
    z: float,
) -> None:
    features = build(parts)
    sign = sign_of(features, intercept)
    reason = sufficient_reasons_module._sufficient_reason(features, intercept, sign, z)
    assume(reason is not None)
    assert reason is not None
    # each feature of the reason keeps the bin of the instance; each other one takes any plausible bin
    choices = [[f.bin_index] if f.name in reason.features else np.flatnonzero(f.plausible).tolist() for f in features]
    for completion in itertools.product(*choices):
        margin = sign * intercept + sum(
            sign * float(f.scores[b]) - z * float(f.stds[b]) for f, b in zip(features, completion, strict=True)
        )
        assert margin > -TOLERANCE


def test_no_reason_with_several_margins_when_all_features_are_not_enough() -> None:
    features = [scored_feature([0.5, -2.0], 0, std=1.0), scored_feature([0.5, -2.0], 0, std=1.0)]
    bounds = [sufficient_reasons_module.Bound(0.0, 1), sufficient_reasons_module.Bound(0.0, -1)]
    assert sufficient_reasons_module.find_reason(features, bounds, 1.96, fact_types_module.LocalOptions()) is None


# An interval regressor whose sufficient reason is a covering problem that the greedy does not solve. Each feature has
# the instance in bin 0 with the score zero, a bin that lowers the value and a bin that raises it. The value is 0,
# inside the interval, so the two margins have the worst cases -3.5 and the gains (2, 2), (1, 2), (1, 4) and (2, 2).
# The greedy needs three features. The features 0 and 3 are enough, and the LP bound is 2.
GREEDY_GAP_SHAPE: tuple[int, ...] = (3, 3, 3, 3)


GREEDY_GAP_MODEL: SmallModel = SmallModel(
    kind="regressor",
    scores=[[[0.0], [-low], [high]] for low, high in [(2.0, 2.0), (1.0, 2.0), (1.0, 4.0), (2.0, 2.0)]],
    deviations=[[[0.0], [0.0], [0.0]] for _ in GREEDY_GAP_SHAPE],
    weights=[[1.0, 1.0, 1.0] for _ in GREEDY_GAP_SHAPE],
    intercepts=[0.0],
    instance_bins=[0, 0, 0, 0],
)


GREEDY_GAP_INTERVAL: Interval = Interval(-2.5, 6.5)


def _feasible_pair(model: CoveringModel) -> ModelResult:
    # the answer of a MIP solver that ends at its time limit: a set that is enough, with no proof of its size
    picks = [choice for choice in model.choices if choice.group in {0, 3}]
    return ModelResult(ModelStatus.FEASIBLE, picks, float(len(picks)))


def _optimal_pair(model: CoveringModel) -> ModelResult:
    # the answer of a MIP solver that proves the optimum
    picks = [choice for choice in model.choices if choice.group in {0, 3}]
    return ModelResult(ModelStatus.OPTIMAL, picks, float(len(picks)))


@pytest.fixture(scope="module")
def greedy_gap_ebm(fit_ebm: Callable[..., FittedModel]) -> FittedModel:
    templates = {("regressor", GREEDY_GAP_SHAPE): fit_template(fit_ebm, "regressor", GREEDY_GAP_SHAPE)}
    return build_ebm(templates, GREEDY_GAP_MODEL)


SOLVER: MinimalityStatus = MinimalityStatus.SOLVER_PROOF


@pytest.mark.parametrize(
    ("decision", "search", "expected"),
    [
        # one margin: sorting proves the size, whatever the search is
        pytest.param(Threshold(-2.5), GreedyCover(bound_backend=None), (2, 2, SORTING), id="sorting"),
        pytest.param(GREEDY_GAP_INTERVAL, MipCover(), (2, 2, AT_BOUND), id="mip-optimal-at-the-bound"),
        pytest.param(
            GREEDY_GAP_INTERVAL,
            MipCover(ScriptedBackend(answer=_optimal_pair, lp=NO_LP)),
            (2, 1, SOLVER),
            id="mip-optimal-above-the-bound",
        ),
        pytest.param(
            GREEDY_GAP_INTERVAL,
            MipCover(ScriptedBackend(answer=_feasible_pair, lp=NO_LP)),
            (2, 1, UNPROVED),
            id="mip-time-limit",
        ),
        pytest.param(
            GREEDY_GAP_INTERVAL,
            MipCover(ScriptedBackend(answer=_feasible_pair)),
            (2, 2, AT_BOUND),
            id="mip-time-limit-at-the-bound",
        ),
        pytest.param(GREEDY_GAP_INTERVAL, GreedyCover(), (3, 2, UNPROVED), id="greedy"),
        pytest.param(GREEDY_GAP_INTERVAL, GreedyCover(bound_backend=None), (3, 1, UNPROVED), id="greedy-no-bound"),
        pytest.param(GREEDY_GAP_INTERVAL, LocalSearchCover(iterations=0), (3, 2, UNPROVED), id="local-search-no-step"),
        pytest.param(GREEDY_GAP_INTERVAL, LocalSearchCover(), (2, 2, AT_BOUND), id="local-search-at-the-bound"),
    ],
)
def test_each_way_to_find_a_sufficient_reason_reports_the_guarantee_that_it_has(
    greedy_gap_ebm: FittedModel,
    decision: Threshold | Interval,
    search: CoveringSolver,
    expected: tuple[int, int, MinimalityStatus],
) -> None:
    # expected: the size of the set, its lower bound, and the reason why the set is, or is not, the smallest
    # contract 2: the claim "smallest" reaches the prompt only on a path that established it. The fallback to every
    # group (_no_set_found) cannot be reached from the facts, and test_covering_solvers covers it.
    options = fact_types_module.LocalOptions(confidence_level=0, min_support=0, decision=decision, search=search)
    facts = facts_module.compute_local_facts(greedy_gap_ebm, GREEDY_GAP_MODEL.instance, options)
    reason = facts.sufficient_reason
    assert reason is not None
    assert (len(reason.features), reason.lower_bound, reason.minimality) == expected
    _, lower_bound, status = expected
    assert reason.guaranteed_margin > 0
    expert = orjson.loads(facts_to_texts(facts, DataScientist()).sufficient_reason)
    assert expert["is_smallest_set"] is status.is_smallest
    assert expert["smallest_possible_number_of_features"] == lower_bound
    assert orjson.loads(facts_to_texts(facts).sufficient_reason)["is_smallest_set"] is status.is_smallest
