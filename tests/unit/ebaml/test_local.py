"""Tests for the deterministic local explanations."""

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import local
from ebaml.graphs import EBMGraph
from tests.unit.ebaml.conftest import fake_model

CATEGORIES = ["p", "q", "r"]
NAMES = ["f0", "f1", "f2"]


@pytest.fixture(scope="module")
def categorical_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> tuple[ExplainableBoostingClassifier, pd.DataFrame]:
    rng = np.random.default_rng(1)
    data = pd.DataFrame({name: rng.choice(CATEGORIES, 900) for name in NAMES})
    effect = data["f0"].map({"p": 1.5, "q": 0.0, "r": -1.5}) + data["f1"].map({"p": 0.5, "q": 0.0, "r": -0.5})
    effect += data["f2"].map({"p": 0.2, "q": 0.0, "r": -0.2}) + rng.normal(size=900) * 0.3
    target = (effect > 0).astype(int)
    return fit_ebm(data, target, interactions=0, random_state=0), data


def _scores(ebm: ExplainableBoostingClassifier, combo: tuple[str, ...]) -> float:
    frame = pd.DataFrame([dict(zip(NAMES, combo, strict=True))])
    return float(ebm.decision_function(frame)[0])


def _all_combos() -> Iterator[tuple[str, ...]]:
    return itertools.product(CATEGORIES, repeat=len(NAMES))


def _brute_force(ebm: ExplainableBoostingClassifier, row: tuple[str, ...]) -> tuple[int | None, int | None]:
    """Minimum change required and size of the minimum sufficient reason, by enumeration.

    Returns
    -------
    tuple[int | None, int | None]
        The sizes, ``None`` when there is none.

    """
    positive = _scores(ebm, row) > 0
    outcome = {combo: (_scores(ebm, combo) > 0) == positive for combo in _all_combos()}
    flips = [sum(a != b for a, b in zip(row, combo, strict=True)) for combo, same in outcome.items() if not same]
    sufficient = [
        size
        for size in range(len(NAMES) + 1)
        for subset in itertools.combinations(range(len(NAMES)), size)
        if all(same for combo, same in outcome.items() if all(combo[i] == row[i] for i in subset))
    ]
    return (min(flips) if flips else None), (min(sufficient) if sufficient else None)


@pytest.mark.parametrize("row", list(itertools.product(CATEGORIES, repeat=3)))
def test_matches_brute_force(categorical_ebm: tuple, row: tuple[str, ...]) -> None:
    ebm, _ = categorical_ebm
    facts = local.compute_local_facts(
        ebm,
        dict(zip(NAMES, row, strict=True)),
        local.LocalOptions(confidence_level=0, min_support=0),
    )
    flip, sufficient = _brute_force(ebm, row)
    assert (facts.counterfactual is None) == (flip is None)
    if flip is not None:
        assert facts.counterfactual is not None
        assert len(facts.counterfactual.changes) == flip
    assert facts.sufficient_reason is not None
    assert len(facts.sufficient_reason.features) == sufficient


def test_counterfactual_really_flips(categorical_ebm: tuple) -> None:
    ebm, _ = categorical_ebm
    row = {"f0": "p", "f1": "p", "f2": "p"}
    facts = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0, min_support=0))
    assert facts.counterfactual is not None
    changed = {**row, **{c.feature: c.to_value for c in facts.counterfactual.changes}}
    assert ebm.predict(pd.DataFrame([changed]))[0] != ebm.predict(pd.DataFrame([row]))[0]


def test_prediction_matches_model(continuous_ebm: tuple) -> None:
    ebm, data = continuous_ebm
    row = data.iloc[3].to_dict()
    facts = local.compute_local_facts(ebm, row)
    assert facts.prediction.logit == pytest.approx(ebm.decision_function(data.iloc[[3]])[0])
    assert facts.prediction.lower <= facts.prediction.probability <= facts.prediction.upper
    assert facts.prediction.label == ebm.predict(data.iloc[[3]])[0]


def test_confidence_makes_claims_harder(continuous_ebm: tuple) -> None:
    ebm, data = continuous_ebm
    row = data.iloc[0].to_dict()
    loose = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0))
    strict = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0.99))
    assert loose.sufficient_reason is not None
    assert strict.sufficient_reason is None or len(strict.sufficient_reason.features) >= len(
        loose.sufficient_reason.features,
    )
    assert loose.counterfactual is not None
    assert strict.counterfactual is None or len(strict.counterfactual.changes) >= len(loose.counterfactual.changes)


def test_completions_are_reproducible_and_grow(continuous_ebm: tuple) -> None:
    ebm, data = continuous_ebm
    row = data.iloc[5].to_dict()
    first = local.compute_local_facts(ebm, row, local.LocalOptions(seed=7, max_steps=2))
    again = local.compute_local_facts(ebm, row, local.LocalOptions(seed=7, max_steps=2))
    assert first.completions == again.completions
    assert [len(s.fixed) for s in first.completions] == [0, 1, 2]
    assert all(0 <= s.share <= 1 for s in first.completions)
    assert first.completions[-1].share >= first.completions[0].share


def test_sufficient_reason_decides_outcome(continuous_ebm: tuple) -> None:
    ebm, data = continuous_ebm
    row = data.iloc[0].to_dict()
    facts = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0, min_support=0))
    assert facts.sufficient_reason is not None
    free = [n for n in row if n not in facts.sufficient_reason.features]
    rng = np.random.default_rng(0)
    for _ in range(50):
        other = data.iloc[int(rng.integers(len(data)))].to_dict()
        mixed = {**row, **{n: other[n] for n in free}}
        assert ebm.predict(pd.DataFrame([mixed]))[0] == facts.prediction.label


@pytest.mark.parametrize(
    ("instance", "message"),
    [
        ({"a": 1.0, "b": 0.0}, "no value"),
        ({"a": float("nan"), "b": 0, "c": "x"}, "missing"),
        ({"a": 0, "b": 0, "c": "?"}, "Unknown"),
    ],
)
def test_bad_instances(continuous_ebm: tuple, instance: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        local.compute_local_facts(continuous_ebm[0], instance)


def test_rejects_interactions_and_bad_confidence(continuous_ebm: tuple, fit_ebm: Callable) -> None:
    ebm, data = continuous_ebm
    with pytest.raises(ValueError, match="confidence_level"):
        local.compute_local_facts(ebm, data.iloc[0].to_dict(), local.LocalOptions(confidence_level=1))
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    with_interactions = fit_ebm(frame, (frame["a"] * frame["b"] > 0).astype(int), interactions=1, random_state=0)
    with pytest.raises(ValueError, match="interactions"):
        local.compute_local_facts(with_interactions, frame.iloc[0].to_dict())


def _feature(
    scores: list[float],
    bin_index: int,
    *,
    std: float = 0.0,
    plausible: list[bool] | None = None,
) -> local._Feature:
    graph = EBMGraph("f", "nominal", [str(i) for i in range(len(scores))], scores, [std] * len(scores))
    weights = np.ones(len(scores))
    return local._Feature("f", graph, weights, bin_index, "v", np.array(plausible or [True] * len(scores)))


def test_counterfactual_is_none_when_no_change_flips() -> None:
    features = [_feature([1.0, 0.9], 0), _feature([1.0, 0.8], 0)]
    assert local._counterfactual(features, intercept=0.0, sign=1, z=0.0) is None


def test_counterfactual_skips_unplausible_bins() -> None:
    features = [_feature([1.0, -5.0], 0, plausible=[True, False]), _feature([1.0, -5.0], 0)]
    counterfactual = local._counterfactual(features, intercept=0.0, sign=1, z=0.0)
    assert counterfactual is not None
    assert [c.feature for c in counterfactual.changes] == ["f"]
    assert len(counterfactual.changes) == 1


def test_sufficient_reason_is_none_when_not_even_all_features_guarantee() -> None:
    features = [_feature([0.5, -2.0], 0, std=1.0), _feature([0.5, -2.0], 0, std=1.0)]
    assert local._sufficient_reason(features, intercept=0.0, sign=1, z=1.96) is None


def test_rejects_multiclass(fit_ebm: Callable) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    ebm = fit_ebm(frame, rng.integers(0, 3, 300), interactions=0, random_state=0)
    with pytest.raises(ValueError, match="binary"):
        local.compute_local_facts(ebm, frame.iloc[0].to_dict())


# The tests below use hand-built features and hand-computed numbers, so each expected value is checked by arithmetic
# and not by the code under test.


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# build a feature with categories name0, name1, ... and the instance in bin ``bin_index``
def _hand(  # ruff: ignore[too-many-arguments] -- keyword-only options of a test helper, each one set by few tests
    name: str,
    scores: list[float],
    bin_index: int = 0,
    *,
    stds: list[float] | None = None,
    weights: list[float] | None = None,
    plausible: list[bool] | None = None,
) -> local._Feature:
    x_vals = [f"{name}{i}" for i in range(len(scores))]
    graph = EBMGraph(name, "nominal", x_vals, scores, stds or [0.0] * len(scores))
    return local._Feature(
        name,
        graph,
        np.array(weights or [1.0] * len(scores)),
        bin_index,
        x_vals[bin_index],
        np.array(plausible or [True] * len(scores)),
    )


def test_local_options_defaults() -> None:
    options = local.LocalOptions()
    assert (options.confidence_level, options.max_steps, options.n_samples, options.seed, options.min_support) == (
        0.95,
        5,
        10_000,
        0,
        0.01,
    )


def test_bin_label() -> None:
    assert local._bin_label((1.5, 2.5)) == "[1.5, 2.5)"
    assert local._bin_label("red") == "red"


@pytest.mark.parametrize(
    ("logit", "probability"),
    [(0.0, 0.5), (2.0, 0.8807970779778823), (-2.0, 0.11920292202211755)],
)
def test_sigmoid(logit: float, probability: float) -> None:
    assert local._sigmoid(logit) == pytest.approx(probability)


def test_bin_index_of_a_category_is_its_position() -> None:
    graph = EBMGraph("f", "categorical", ["a", "b", "c"], [0.0] * 3, [0.0] * 3)
    assert local._bin_index(fake_model(), 0, graph, "b") == 1


def test_bin_index_of_a_number_uses_the_first_cuts_and_the_right_side() -> None:
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [0.0] * 3, [0.0] * 3)
    # the second array stands for a finer resolution that the function must not use
    ebm = fake_model(bins_=[[np.array([1.0, 2.0]), np.array([5.0])]])
    assert local._bin_index(ebm, 0, graph, 1.5) == 1
    assert local._bin_index(ebm, 0, graph, 2.0) == 2  # a value on a cut belongs to the bin on its right


def test_plausible_categories_need_the_support() -> None:
    weights = np.array([1.0, 6.0, 1.0])  # the shares are 0.125, 0.75 and 0.125
    plausible = local._plausible_bins(weights, continuous=False, min_support=0.2)
    assert plausible.tolist() == [False, True, False]
    # a share equal to the minimum is enough
    assert local._plausible_bins(weights, continuous=False, min_support=0.125).tolist() == [True, True, True]


def test_plausible_bins_of_a_continuous_feature_trim_the_tails() -> None:
    weights = np.array([1.0, 6.0, 1.0])
    # the bins cover 0 to 0.125, 0.125 to 0.875 and 0.875 to 1 of the data
    assert local._plausible_bins(weights, continuous=True, min_support=0.2).tolist() == [False, True, False]
    # a bin that ends exactly at the minimum, or starts exactly at 1 minus the minimum, is out
    assert local._plausible_bins(weights, continuous=True, min_support=0.125).tolist() == [False, True, False]


def _model_features(*, term_features: list[tuple[int, ...]], classes: list[str]) -> ExplainableBoostingClassifier:
    return fake_model(feature_names_in_=["a", "b"], term_features_=term_features, classes_=np.array(classes))


@pytest.mark.parametrize(
    "term_features",
    [[(0,)], [(0,), (1,), (0, 1)], [(0, 1), (1,)], [(), (0,)]],
    ids=["missing term", "extra term", "interaction", "empty term"],
)
def test_features_reject_interactions(term_features: list[tuple[int, ...]]) -> None:
    with pytest.raises(ValueError, match="interactions"):
        local._features(_model_features(term_features=term_features, classes=["n", "y"]), {}, 0.01)


@pytest.mark.parametrize("classes", [["only"], ["a", "b", "c"]])
def test_features_reject_other_than_two_classes(classes: list[str]) -> None:
    with pytest.raises(ValueError, match="binary"):
        local._features(_model_features(term_features=[(0,), (1,)], classes=classes), {}, 0.01)


@pytest.mark.parametrize(
    ("feature_type", "expected"),
    [
        ("categorical", [False, False, True]),  # sorts before "continuous"
        ("nominal", [False, False, True]),  # sorts after "continuous"
        ("continuous", [False, True, True]),
    ],
)
def test_features_trim_by_the_type_of_the_feature(
    monkeypatch: pytest.MonkeyPatch,
    feature_type: str,
    expected: list[bool],
) -> None:
    # the shares are 0.25, 0.25 and 0.5: with a support of 0.3 a category needs 0.3 and the first bin of a number is cut
    graph = EBMGraph("a", feature_type, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [0.0] * 3, [0.0] * 3)
    if feature_type != "continuous":
        graph = EBMGraph("a", feature_type, ["p", "q", "r"], [0.0] * 3, [0.0] * 3)
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(lambda _ebm, _index: graph))
    ebm = fake_model(
        feature_names_in_=["a"],
        term_features_=[(0,)],
        classes_=np.array(["n", "y"]),
        bins_=[[np.array([1.0, 2.0])]],
        bin_weights_=[np.array([0.0, 2.0, 2.0, 4.0, 0.0])],
    )
    value = "r" if feature_type != "continuous" else 2.5
    (feature,) = local._features(ebm, {"a": value}, 0.3)
    assert feature.plausible.tolist() == expected
    assert feature.bin_index == 2


def test_robust_scores_move_by_z_standard_deviations() -> None:
    feature = _hand("f", [1.0, -1.0, 4.0], stds=[0.5, 0.25, 1.0], plausible=[True, True, False])
    # z is 2, so the scores move by 1, 0.5 and 2; the unplausible bin is inf
    lowered = local._robust_scores(feature, 1, 2.0, towards_outcome=True)
    raised = local._robust_scores(feature, 1, 2.0, towards_outcome=False)
    assert lowered.tolist() == [0.0, -1.5, np.inf]
    assert raised.tolist() == [2.0, -0.5, np.inf]
    # for the negative class the scores are turned around first
    assert local._robust_scores(feature, -1, 2.0, towards_outcome=True).tolist() == [-2.0, 0.5, np.inf]


def test_counterfactual_changes_the_biggest_drop_first() -> None:
    features = [_hand("f1", [1.0, -2.0]), _hand("f2", [0.5, -0.25])]
    # the margin is 0.5 + 1 + 0.5 = 2; f1 can drop by 3, f2 by 0.75
    counterfactual = local._counterfactual(features, intercept=0.5, sign=1, z=0.0)
    assert counterfactual is not None
    assert counterfactual.changes == [local.Change("f1", "f10", "f11", -3.0)]
    # the new logit is 0.5 + 1.5 - 3 = -1, so the new class (negative) has the probability sigmoid(1)
    assert counterfactual.new_probability == pytest.approx(_sig(1.0))


def test_counterfactual_of_the_negative_class() -> None:
    features = [_hand("f1", [-1.0, 2.0]), _hand("f2", [-0.5, 0.25])]
    counterfactual = local._counterfactual(features, intercept=-0.5, sign=-1, z=0.0)
    assert counterfactual is not None
    assert counterfactual.changes == [local.Change("f1", "f10", "f11", 3.0)]
    assert counterfactual.new_probability == pytest.approx(_sig(1.0))  # the new logit is +1


def test_counterfactual_uses_as_many_changes_as_it_takes() -> None:
    features = [_hand("f1", [1.0, 0.0]), _hand("f2", [1.0, 0.0]), _hand("f3", [0.5, -1.0])]
    # the margin is 2.5 and the drops are 1, 1 and 1.5: the margin is 1 and then 0 (a tie, not a flip) and then -1
    counterfactual = local._counterfactual(features, intercept=0.0, sign=1, z=0.0)
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f3", "f1", "f2"]


def test_counterfactual_with_two_changes() -> None:
    features = [_hand("f1", [1.0, 0.0]), _hand("f2", [1.0, -0.5]), _hand("f3", [1.0, 0.5])]
    # the margin is 3 and the drops are 1, 1.5 and 0.5: after the two largest drops it is 0.5, so it is not yet flipped
    assert local._counterfactual(features, intercept=0.0, sign=1, z=0.0) is None
    features[0] = _hand("f1", [1.0, -3.0])
    # the drops are now 4, 1.5 and 0.5, and the margin 3 flips after the first; with a margin of 4.5 it takes two
    counterfactual = local._counterfactual(features, intercept=1.5, sign=1, z=0.0)
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f1", "f2"]


def test_counterfactual_probability_of_the_new_class() -> None:
    # the margin is 2.5 and the drop is 4, so the new logit is -1.5 and the new (negative) class has sigmoid(1.5)
    counterfactual = local._counterfactual([_hand("f", [1.0, -3.0])], intercept=1.5, sign=1, z=0.0)
    assert counterfactual is not None
    assert counterfactual.new_probability == pytest.approx(_sig(1.5))


def test_counterfactual_is_none_for_a_tie() -> None:
    assert local._counterfactual([_hand("f", [1.0, 0.0])], intercept=0.0, sign=1, z=0.0) is None
    # the margin counts the score of the feature itself, not just its sign
    assert local._counterfactual([_hand("f", [2.0, 0.0])], intercept=0.0, sign=1, z=0.0) is None


def test_counterfactual_asks_for_the_worst_case_of_the_other_class() -> None:
    feature = _hand("f", [1.0, -2.0], stds=[0.5, 0.5])
    # with z = 1 the scores are raised by 0.5, so the drop is 1 + 1.5 = 2.5, and the margin 2.8 is not enough
    assert local._counterfactual([feature], intercept=1.8, sign=1, z=1.0) is None


def test_fixing_order_counts_what_a_known_value_guarantees() -> None:
    first = _hand("f1", [2.0, -1.0], stds=[1.0, 0.0])
    second = _hand("f2", [1.5, 0.0])
    # at z = 1 the guaranteed scores of f1 are 1 and -1, so knowing its value gains 2; f2 gains 1.5
    order = local._fixing_order([second, first], sign=1, z=1.0)
    assert [(gain, feature.name) for gain, feature in order] == [(2.0, "f1"), (1.5, "f2")]


def test_sufficient_reason_fixes_the_largest_gains_first() -> None:
    features = [_hand("f3", [0.5, 0.0]), _hand("f1", [2.0, -1.0]), _hand("f2", [1.0, -0.5])]
    # the worst case is -1.5 (-1, -0.5 and 0); the gains are 3 (f1), 1.5 (f2) and 0.5 (f3)
    reason = local._sufficient_reason(features, intercept=0.0, sign=1, z=0.0)
    assert reason == local.SufficientReason(["f1"], 1.5)


def test_sufficient_reason_needs_no_feature_when_the_outcome_is_certain() -> None:
    # even the lowest score keeps the margin above 0, so no input has to be fixed
    assert local._sufficient_reason([_hand("f", [1.0, 2.0])], 0.0, 1, 0.0) == local.SufficientReason([], 1.0)


def test_sufficient_reason_needs_a_margin_above_zero() -> None:
    # the worst case is exactly 0: knowing the value gains 1, and the margin is 0 before and 1 after
    assert local._sufficient_reason([_hand("f", [0.0, 1.0], 1)], 0.0, 1, 0.0) == local.SufficientReason(["f"], 1.0)
    # when the margin stays at 0, the outcome is not certain
    assert local._sufficient_reason([_hand("f", [0.0, 0.0])], 0.0, 1, 0.0) is None


# the share of completions with the same prediction, without and with the value of the one feature known
def _completion_shares(cur: float, sampled: float, *, intercept: float = 0.0, sign: int = 1) -> list[float]:
    feature = _hand("f", [cur, sampled], weights=[0.0, 1.0])  # the sample always falls in the second bin
    options = local.LocalOptions(n_samples=5, max_steps=1)
    return [step.share for step in local._completions([feature], [feature], intercept, sign, options)]


@pytest.mark.parametrize(
    ("cur", "sampled", "sign", "expected"),
    [
        (0.5, -0.5, 1, [0.0, 1.0]),
        (0.5, -0.5, -1, [1.0, 0.0]),
        (-0.5, 0.5, 1, [1.0, 0.0]),
        (0.0, 0.0, 1, [0.0, 0.0]),
        (0.0, 0.0, -1, [0.0, 0.0]),
    ],
)
def test_completions_share_the_samples_with_the_same_sign(
    cur: float,
    sampled: float,
    sign: int,
    expected: list[float],
) -> None:
    assert _completion_shares(cur, sampled, sign=sign) == expected


def test_completions_add_the_intercept() -> None:
    # the logits are 2 - 3 = -1 and 2 - 1 = +1
    assert _completion_shares(-1.0, -3.0, intercept=2.0) == [0.0, 1.0]


@pytest.mark.parametrize("max_steps", [1, 3])
def test_completions_reveal_max_steps_features(max_steps: int) -> None:
    features = [_hand(f"f{i}", [1.0, 1.0]) for i in range(4)]
    options = local.LocalOptions(n_samples=5, max_steps=max_steps)
    steps = local._completions(features, features, 0.0, 1, options)
    assert [len(step.fixed) for step in steps] == list(range(max_steps + 1))


def _patch_features(monkeypatch: pytest.MonkeyPatch, features: list[local._Feature]) -> None:
    monkeypatch.setattr(local, "_features", lambda _ebm, _instance, _support: features)


def _hand_model(intercept: float = 0.25) -> ExplainableBoostingClassifier:
    return fake_model(intercept_=np.array([intercept, 9.0]), classes_=np.array(["no", "yes"]))


Z95 = 1.959963984540054


def test_facts_of_a_positive_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [_hand("b", [-0.5, 0.5], stds=[0.4, 0.3]), _hand("a", [1.0, -1.0], stds=[0.3, 0.4])]
    _patch_features(monkeypatch, features)
    facts = local.compute_local_facts(_hand_model(), {})
    prediction = facts.prediction
    logit, std = 0.25 - 0.5 + 1.0, 0.5  # the standard deviation is sqrt(0.4**2 + 0.3**2)
    assert prediction.label == "yes"
    assert prediction.logit == pytest.approx(logit)
    assert prediction.logit_std == pytest.approx(std)
    assert prediction.probability == pytest.approx(_sig(logit))
    assert prediction.lower == pytest.approx(_sig(logit - Z95 * std))
    assert prediction.upper == pytest.approx(_sig(logit + Z95 * std))
    assert facts.confidence_level == pytest.approx(0.95)
    # the largest effect first
    assert facts.contributions == [local.Contribution("a", "a0", 1.0, 0.3), local.Contribution("b", "b0", -0.5, 0.4)]
    assert facts.counterfactual == local._counterfactual(features, 0.25, 1, Z95)
    assert facts.sufficient_reason == local._sufficient_reason(features, 0.25, 1, Z95)


def test_facts_of_a_negative_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [_hand("a", [-1.0, 1.0], stds=[0.3, 0.4]), _hand("b", [0.5, -0.5], stds=[0.4, 0.3], bin_index=1)]
    _patch_features(monkeypatch, features)
    facts = local.compute_local_facts(_hand_model(-0.25), {}, local.LocalOptions(confidence_level=0.9, seed=3))
    prediction = facts.prediction
    logit, std, z = -0.25 - 1.0 - 0.5, math.hypot(0.3, 0.3), 1.6448536269514722
    assert prediction.label == "no"
    assert prediction.probability == pytest.approx(1 - _sig(logit))
    assert prediction.lower == pytest.approx(1 - _sig(logit + z * std))
    assert prediction.upper == pytest.approx(1 - _sig(logit - z * std))
    assert facts.counterfactual == local._counterfactual(features, -0.25, -1, z)


def test_facts_of_a_prediction_on_the_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    # a logit of exactly 0 belongs to the negative class
    _patch_features(monkeypatch, [_hand("a", [1.0, -1.0]), _hand("b", [-1.0, 1.0])])
    facts = local.compute_local_facts(_hand_model(0.0), {}, local.LocalOptions(confidence_level=0))
    assert (facts.prediction.label, facts.prediction.probability) == ("no", 0.5)
    # the negative class is the prediction, so only b can still move the logit up, and a cannot
    assert facts.counterfactual is not None
    assert [change.feature for change in facts.counterfactual.changes] == ["b"]


def test_facts_use_the_confidence_level_as_a_two_sided_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_features(monkeypatch, [_hand("a", [1.0, -1.0], stds=[1.0, 1.0])])
    prediction = local.compute_local_facts(_hand_model(), {}, local.LocalOptions(confidence_level=0.5)).prediction
    z = 0.6744897501960817  # the 75th percentile of the normal distribution
    assert prediction.lower == pytest.approx(_sig(1.25 - z))
    assert prediction.upper == pytest.approx(_sig(1.25 + z))


@pytest.mark.parametrize("confidence_level", [-0.5, 1.5])
def test_facts_reject_a_confidence_level_out_of_range(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        local.compute_local_facts(_hand_model(), {}, local.LocalOptions(confidence_level=confidence_level))
