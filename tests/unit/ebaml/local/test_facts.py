"""Tests for the deterministic local explanations."""

import itertools
import math
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from decimal import Decimal

import attrs
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ebaml.core.custom_types import ClassLabel, FeatureValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError, OptionRangeError, UnsupportedModelError
from ebaml.local import completions as completions_module
from ebaml.local import counterfactuals as counterfactuals_module
from ebaml.local import fact_features as fact_features_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local import sufficient_reasons as sufficient_reasons_module
from ebaml.local.covering_backend import Stop
from ebaml.local.covering_solvers import CoveringSolver, GreedyCover, LocalSearchCover, MipCover
from ebaml.local.decisions import Interval, Threshold
from ebaml.local.local_kinds import Outcome, OutputKind
from ebaml.local.minimality import MinimalityStatus
from tests.unit.ebaml.conftest import (
    MISSING_VALUE,
    FittedClassifier,
    FittedRegressor,
    UnknownKind,
    categorical,
    fake_model,
    frame_of_columns,
    frame_of_matrix,
    frame_of_rows,
    instance_of,
    returning,
)
from tests.unit.ebaml.local.conftest import (
    CATEGORIES,
    CONFIDENCE_LEVELS,
    MIN_SUPPORTS,
    NAMES,
    TOLERANCE,
    FittedModel,
    SmallModel,
    Templates,
    build_ebm,
    exact_share,
    feature_name,
    hand,
    model_cases,
    model_features,
    model_row,
    model_scores,
    multiclass_model,
    oracle_problem,
    plausible_bins,
    probability_after,
    sig,
)

FEATURE_INDEXES: tuple[int, ...] = tuple(index for index, _ in enumerate(NAMES))


ALL_FEATURES = frozenset(FEATURE_INDEXES)


def _scores(ebm: ExplainableBoostingClassifier, combo: tuple[str, ...]) -> float:
    frame = frame_of_rows([dict(zip(NAMES, combo, strict=True))])
    logits: np.ndarray = ebm.decision_function(frame)
    return float(logits[0])


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
        for subset in itertools.combinations(FEATURE_INDEXES, size)
        if all(same for combo, same in outcome.items() if all(combo[i] == row[i] for i in subset))
    ]
    return (min(flips) if flips else None), (min(sufficient) if sufficient else None)


@pytest.mark.parametrize("row", list(itertools.product(CATEGORIES, repeat=3)))
def test_matches_brute_force(
    categorical_ebm: FittedClassifier,
    row: tuple[str, ...],
) -> None:
    ebm, _ = categorical_ebm
    facts = facts_module.compute_local_facts(
        ebm,
        dict(zip(NAMES, row, strict=True)),
        fact_types_module.LocalOptions(confidence_level=0, min_support=0),
    )
    flip, sufficient = _brute_force(ebm, row)
    assert (facts.counterfactual is None) == (flip is None)
    if flip is not None:
        assert facts.counterfactual is not None
        assert len(facts.counterfactual.changes) == flip
    assert facts.sufficient_reason is not None
    assert len(facts.sufficient_reason.features) == sufficient
    # the brute force shows that no smaller set is sufficient, which is what minimum claims
    assert facts.sufficient_reason.minimality.is_smallest


def test_prediction_matches_model(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 3)
    facts = facts_module.compute_local_facts(ebm, row)
    logits: np.ndarray = ebm.decision_function(data.iloc[[3]])
    assert facts.prediction.score == pytest.approx(logits[0])
    assert facts.prediction.lower <= facts.prediction.output <= facts.prediction.upper
    assert facts.prediction.decision == ebm.predict(data.iloc[[3]])[0]


def test_confidence_makes_claims_harder(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 0)
    loose = facts_module.compute_local_facts(ebm, row, fact_types_module.LocalOptions(confidence_level=0))
    strict = facts_module.compute_local_facts(ebm, row, fact_types_module.LocalOptions(confidence_level=0.99))
    assert loose.sufficient_reason is not None
    assert strict.sufficient_reason is None or len(strict.sufficient_reason.features) >= len(
        loose.sufficient_reason.features,
    )
    assert loose.counterfactual is not None
    assert strict.counterfactual is None or len(strict.counterfactual.changes) >= len(loose.counterfactual.changes)


@pytest.mark.parametrize(
    ("instance", "message"),
    [
        ({"a": 1.0, "b": 0.0}, "no value"),
        ({"a": float("nan"), "b": 0, "c": "x"}, "missing"),
        # a NaN of NumPy and the pd.NA of a nullable column were not seen as missing, so they got a bin or a TypeError
        ({"a": np.float32("nan"), "b": 0, "c": "x"}, "missing"),
        ({"a": 0, "b": 0, "c": MISSING_VALUE}, "missing"),
        ({"a": 0, "b": 0, "c": "?"}, "Unknown"),
    ],
)
def test_bad_instances(
    continuous_ebm: FittedClassifier,
    instance: dict[str, FeatureValue],
    message: str,
) -> None:
    with pytest.raises(InvalidInstanceError, match=message):
        facts_module.compute_local_facts(continuous_ebm[0], instance)


def test_rejects_interactions_and_bad_confidence(
    continuous_ebm: FittedClassifier,
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> None:
    ebm, data = continuous_ebm
    with pytest.raises(OptionRangeError, match="confidence_level"):
        facts_module.compute_local_facts(ebm, instance_of(data, 0), fact_types_module.LocalOptions(confidence_level=1))
    rng = np.random.default_rng(0)
    frame = frame_of_matrix(rng.normal(size=(300, 3)), list("abc"))
    with_interactions = fit_ebm(frame, (frame["a"] * frame["b"] > 0).astype(int), interactions=1, random_state=0)
    with pytest.raises(UnsupportedModelError, match="interactions"):
        facts_module.compute_local_facts(with_interactions, instance_of(frame, 0))


def _margin_views(
    features: Sequence[fact_features_module.Feature],
    winner: int,
    intercepts: Sequence[float],
) -> list[tuple[float, list[fact_features_module.Feature]]]:
    """Return the intercept and the views of every margin of the class ``winner`` against its rivals.

    Returns
    -------
    list[tuple[float, list[fact_features_module.Feature]]]
        One pair for each rival class.

    """
    return [
        (intercepts[winner] - intercepts[rival], [feature.against((winner, rival)) for feature in features])
        for rival, _ in enumerate(intercepts)
        if rival != winner
    ]


def _brute_force_counterfactual_size(views: Sequence[tuple[float, list[fact_features_module.Feature]]]) -> int | None:
    """Try every set of features and every plausible bin for each, and return the fewest features that flip.

    Returns
    -------
    int | None
        The fewest features, or ``None`` if no change flips the class.

    """
    feature_count = len(views[0][1])
    for size in range(1, feature_count + 1):
        for chosen in itertools.combinations(range(feature_count), size):
            bin_choices = [np.flatnonzero(views[0][1][index].plausible) for index in chosen]
            for bins in itertools.product(*bin_choices):
                for intercept, feature_views in views:
                    margin = intercept + sum(view.current for view in feature_views)
                    for index, bin_index in zip(chosen, bins, strict=True):
                        margin += feature_views[index].scores[bin_index] - feature_views[index].current
                    if margin < 0:
                        return size
    return None


def _brute_force_reason_size(views: Sequence[tuple[float, list[fact_features_module.Feature]]]) -> int | None:
    """Try every set of fixed features, and return the fewest that keep every margin positive for all completions.

    Returns
    -------
    int | None
        The fewest features, or ``None`` if no set is sufficient.

    """
    feature_count = len(views[0][1])
    for size in range(feature_count + 1):
        for fixed in itertools.combinations(range(feature_count), size):
            free = [index for index in range(feature_count) if index not in fixed]
            bin_choices = [np.flatnonzero(views[0][1][index].plausible) for index in free]
            holds = True
            for bins in itertools.product(*bin_choices):
                for intercept, feature_views in views:
                    margin = intercept + sum(feature_views[index].current for index in fixed)
                    margin += sum(
                        feature_views[index].scores[bin_index] for index, bin_index in zip(free, bins, strict=True)
                    )
                    holds = holds and margin > 0
            if holds:
                return size
    return None


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239))
def test_multiclass_facts_are_the_smallest_by_brute_force(row: int) -> None:
    model, frame = multiclass_model()
    options = fact_types_module.LocalOptions(confidence_level=0.0, min_support=0.0)
    facts = facts_module.compute_local_facts(model, instance_of(frame, row), options)
    features = fact_features_module.read_features(model, instance_of(frame, row), 0.0)
    intercepts = np.ravel(model.intercept_).tolist()
    winner = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    views = _margin_views(features, winner, intercepts)
    expected_flip = _brute_force_counterfactual_size(views)
    expected_reason = _brute_force_reason_size(views)
    assert facts.prediction.decision == model.classes_[winner]
    assert (None if facts.counterfactual is None else len(facts.counterfactual.changes)) == expected_flip
    assert (None if facts.sufficient_reason is None else len(facts.sufficient_reason.features)) == expected_reason
    if facts.sufficient_reason is not None:
        assert facts.sufficient_reason.minimality.is_smallest


def test_multiclass_prediction_reports_the_predicted_class() -> None:
    model, frame = multiclass_model()
    facts = facts_module.compute_local_facts(
        model,
        instance_of(frame, 0),
        fact_types_module.LocalOptions(confidence_level=0.0),
    )
    probabilities = model.predict_proba(frame.iloc[[0]])[0]
    winner = int(np.argmax(probabilities))
    prediction = facts.prediction
    assert prediction.decision == model.classes_[winner]
    assert prediction.output == pytest.approx(float(probabilities[winner]), abs=1e-6)
    assert (prediction.lower, prediction.upper) == pytest.approx((prediction.output, prediction.output))
    # the class holds while its score is above the best score of the other classes
    assert prediction.threshold == pytest.approx(prediction.score - prediction.margin)
    assert prediction.margin > 0
    assert prediction.output_kind is OutputKind.CLASS_PROBABILITY


@pytest.fixture(scope="module")
def coded_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> FittedClassifier:
    rng = np.random.default_rng(5)
    size = 600
    data = frame_of_columns(
        {
            "a": rng.normal(size=size),
            "grade": categorical(rng.choice([1, 2, 3], size)),
            "flag": categorical(rng.choice([True, False], size)),
            "share": categorical(rng.choice([0.5, 1.0], size)),
        },
    )
    signal = data["a"] + data["grade"].astype(int) - 2 + 0.8 * data["flag"].astype(bool) + rng.normal(size=size)
    return fit_ebm(data, (signal > 0.3).astype(int), interactions=0, random_state=0), data


def test_facts_of_training_rows_with_number_and_boolean_categories_agree_with_the_ebm(
    coded_ebm: FittedClassifier,
) -> None:
    # a row of the training data has NumPy integers, booleans and floats for the categories that interpret keeps as
    # text, so every such row raised "Unknown category"
    ebm, data = coded_ebm
    for index in range(20):
        facts = facts_module.compute_local_facts(ebm, instance_of(data, index))
        probabilities = ebm.predict_proba(data.iloc[[index]])[0]
        assert facts.prediction.decision == ebm.predict(data.iloc[[index]])[0]
        assert facts.prediction.output == pytest.approx(probabilities.max())


def test_facts_reject_a_model_kind_that_has_no_local_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(facts_module, "model_kind", returning(UnknownKind()))
    with pytest.raises(UnsupportedModelError, match=r"^Local explanations need a classifier or a regressor\. Got a"):
        facts_module.compute_local_facts(model_features(term_features=[(0,)], classes=["a", "b"], link="logit"), {})


@pytest.mark.parametrize(
    ("classes", "link", "detail"),
    [
        (["only"], "logit", "An EBM with the logit link needs two classes"),
        (["a", "b"], "probit", "EBaMl reads the links"),
    ],
)
def test_facts_reject_other_than_a_binary_classifier_or_a_regressor(classes: list[str], link: str, detail: str) -> None:
    model = model_features(term_features=[(0,), (1,)], classes=classes, link=link)
    with pytest.raises(
        UnsupportedModelError,
        match=rf"^Local explanations need a classifier or a regressor\. {detail}",
    ):
        facts_module.compute_local_facts(model, {})


# the share of completions with the same prediction, without and with the value of the one feature known
def _completion_shares(cur: float, sampled: float, *, intercept: float = 0.0, sign: int = 1) -> list[float]:
    feature = hand("f", [cur, sampled], weights=[0.0, 1.0])  # the sample always falls in the second bin
    options = fact_types_module.LocalOptions(n_samples=5, max_steps=1)
    return [
        step.share
        for step in completions_module.sample_completions(
            [feature],
            [feature],
            [sufficient_reasons_module.Bound(intercept, sign)],
            options,
        )
    ]


@pytest.mark.parametrize(
    ("cur", "sampled", "sign", "expected"),
    [
        (0.5, -0.5, 1, [0.0, 1.0]),
        (0.5, -0.5, -1, [1.0, 0.0]),
        (-0.5, 0.5, 1, [1.0, 0.0]),
        (0.0, 0.0, 1, [0.0, 0.0]),
        (0.0, 0.0, -1, [1.0, 1.0]),  # regression: a logit of 0 is the negative class, so it matches sign -1
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


def _patch_features(monkeypatch: pytest.MonkeyPatch, features: list[fact_features_module.Feature]) -> None:
    monkeypatch.setattr(facts_module, "read_features", returning(features))


def _hand_model(intercept: float = 0.25) -> ExplainableBoostingClassifier:
    return fake_model(intercept_=np.array([intercept, 9.0]), classes_=np.array(["no", "yes"]), link_="logit")


Z95 = 1.959963984540054


def test_facts_of_a_positive_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [hand("b", [-0.5, 0.5], stds=[0.4, 0.3]), hand("a", [1.0, -1.0], stds=[0.3, 0.4])]
    _patch_features(monkeypatch, features)
    facts = facts_module.compute_local_facts(_hand_model(), {})
    prediction = facts.prediction
    logit, std = 0.25 - 0.5 + 1.0, 0.5  # the standard deviation is sqrt(0.4**2 + 0.3**2)
    assert prediction.decision == "yes"
    assert prediction.score == pytest.approx(logit)
    assert prediction.score_std == pytest.approx(std)
    assert prediction.output == pytest.approx(sig(logit))
    assert prediction.margin == pytest.approx(logit)
    assert prediction.lower == pytest.approx(sig(logit - Z95 * std))
    assert prediction.upper == pytest.approx(sig(logit + Z95 * std))
    assert facts.confidence_level == pytest.approx(0.95)
    # the largest effect first
    assert facts.contributions == [
        fact_types_module.Contribution("a", "a0", 1.0, 0.3),
        fact_types_module.Contribution("b", "b0", -0.5, 0.4),
    ]
    assert facts.counterfactual == counterfactuals_module._counterfactual(
        features,
        0.25,
        1,
        Z95,
        after=lambda margin, _replaced: Outcome("no", probability_after(margin, _replaced).output),
    )
    assert facts.sufficient_reason == sufficient_reasons_module._sufficient_reason(features, 0.25, 1, Z95)


def test_facts_of_a_negative_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [hand("a", [-1.0, 1.0], stds=[0.3, 0.4]), hand("b", [0.5, -0.5], stds=[0.4, 0.3], bin_index=1)]
    _patch_features(monkeypatch, features)
    facts = facts_module.compute_local_facts(
        _hand_model(-0.25),
        {},
        fact_types_module.LocalOptions(confidence_level=0.9, seed=3),
    )
    prediction = facts.prediction
    logit, std, z = -0.25 - 1.0 - 0.5, math.hypot(0.3, 0.3), 1.6448536269514722
    assert prediction.decision == "no"
    assert prediction.output == pytest.approx(1 - sig(logit))
    # the margin is the log-odds of the predicted class, so it is positive for the negative class too
    assert prediction.margin == pytest.approx(-logit)
    assert prediction.lower == pytest.approx(1 - sig(logit + z * std))
    assert prediction.upper == pytest.approx(1 - sig(logit - z * std))
    assert facts.counterfactual == counterfactuals_module._counterfactual(
        features,
        -0.25,
        -1,
        z,
        after=lambda margin, _replaced: Outcome("yes", probability_after(margin, _replaced).output),
    )


def test_facts_of_a_prediction_on_the_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    # a logit of exactly 0 belongs to the negative class
    _patch_features(monkeypatch, [hand("a", [1.0, -1.0]), hand("b", [-1.0, 1.0])])
    facts = facts_module.compute_local_facts(_hand_model(0.0), {}, fact_types_module.LocalOptions(confidence_level=0))
    assert (facts.prediction.decision, facts.prediction.output) == ("no", 0.5)
    # the negative class is the prediction, so only b can still move the logit up, and a cannot
    assert facts.counterfactual is not None
    assert [change.feature for change in facts.counterfactual.changes] == ["b"]
    assert facts.counterfactual.new_decision == "yes"


def test_facts_use_the_confidence_level_as_a_two_sided_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_features(monkeypatch, [hand("a", [1.0, -1.0], stds=[1.0, 1.0])])
    prediction = facts_module.compute_local_facts(
        _hand_model(),
        {},
        fact_types_module.LocalOptions(confidence_level=0.5),
    ).prediction
    z = 0.6744897501960817  # the 75th percentile of the normal distribution
    assert prediction.lower == pytest.approx(sig(1.25 - z))
    assert prediction.upper == pytest.approx(sig(1.25 + z))


@pytest.mark.parametrize("confidence_level", [-0.5, 1.5])
def test_facts_reject_a_confidence_level_out_of_range(confidence_level: float) -> None:
    with pytest.raises(OptionRangeError, match="confidence_level"):
        facts_module.compute_local_facts(
            _hand_model(),
            {},
            fact_types_module.LocalOptions(confidence_level=confidence_level),
        )


def test_a_decimal_value_falls_in_the_bin_of_its_float(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    as_float = facts_module.compute_local_facts(ebm, {"a": 0.25, "b": -0.5, "c": "x"})
    as_decimal = facts_module.compute_local_facts(ebm, {"a": Decimal("0.25"), "b": -0.5, "c": "x"})
    assert as_decimal.prediction == as_float.prediction


def _brute_force_regressor(
    ebm: ExplainableBoostingRegressor,
    row: tuple[str, ...],
    holds: Callable[[float], bool],
) -> tuple[int | None, int | None]:
    """Minimum change required and size of the minimum sufficient reason of a regressor, by enumeration.

    Returns
    -------
    tuple[int | None, int | None]
        The sizes, ``None`` when there is none.

    """
    values = {
        combo: float(ebm.predict(frame_of_rows([dict(zip(NAMES, combo, strict=True))]))[0]) for combo in _all_combos()
    }
    outcome = {combo: holds(value) == holds(values[row]) for combo, value in values.items()}
    flips = [sum(a != b for a, b in zip(row, combo, strict=True)) for combo, same in outcome.items() if not same]
    sufficient = [
        size
        for size in range(len(NAMES) + 1)
        for subset in itertools.combinations(FEATURE_INDEXES, size)
        if all(same for combo, same in outcome.items() if all(combo[i] == row[i] for i in subset))
    ]
    return (min(flips) if flips else None), (min(sufficient) if sufficient else None)


@pytest.mark.parametrize("threshold", [None, 0.5, -1.0, 2.5])
@pytest.mark.parametrize("row", list(itertools.product(CATEGORIES, repeat=3)))
def test_regressor_threshold_matches_brute_force(
    categorical_regressor: FittedRegressor,
    row: tuple[str, ...],
    threshold: float | None,
) -> None:
    ebm, _ = categorical_regressor
    cut = float(ebm.intercept_) if threshold is None else threshold
    facts = facts_module.compute_local_facts(
        ebm,
        dict(zip(NAMES, row, strict=True)),
        fact_types_module.LocalOptions(
            confidence_level=0,
            min_support=0,
            decision=None if threshold is None else Threshold(threshold),
        ),
    )
    flip, sufficient = _brute_force_regressor(ebm, row, lambda value: value > cut)
    assert (facts.counterfactual is None) == (flip is None)
    if flip is not None:
        assert facts.counterfactual is not None
        assert len(facts.counterfactual.changes) == flip
    assert (facts.sufficient_reason is None) == (sufficient is None)
    if sufficient is not None:
        assert facts.sufficient_reason is not None
        assert len(facts.sufficient_reason.features) == sufficient
        assert facts.sufficient_reason.minimality.is_smallest


def _sufficient_sizes(
    ebm: ExplainableBoostingRegressor,
    row: tuple[str, ...],
    holds: Callable[[float], bool],
) -> dict[frozenset[int], bool]:
    """Tell for every set of fixed features whether the answer holds for all completions, by enumeration.

    Returns
    -------
    dict[frozenset[int], bool]
        For each set of positions, True if every completion keeps the answer of the row.

    """
    values = {
        combo: float(ebm.predict(frame_of_rows([dict(zip(NAMES, combo, strict=True))]))[0]) for combo in _all_combos()
    }
    same = {combo: holds(value) == holds(values[row]) for combo, value in values.items()}
    return {
        frozenset(subset): all(ok for combo, ok in same.items() if all(combo[i] == row[i] for i in subset))
        for size in range(len(NAMES) + 1)
        for subset in itertools.combinations(FEATURE_INDEXES, size)
    }


def _interval_answer(interval: Interval, value: float) -> Callable[[float], bool]:
    """Return the test of the answer that the interval gives for ``value``: at most the low end, inside, or above.

    Returns
    -------
    Callable[[float], bool]
        The test.

    """
    if value <= interval.low:
        return lambda other: other <= interval.low
    if value > interval.high:
        return lambda other: other > interval.high
    return lambda other: interval.low < other <= interval.high


@pytest.mark.parametrize("interval", [Interval(-1.5, 1.5), Interval(-0.7, 0.7), Interval(-3.5, 3.5)])
@pytest.mark.parametrize("search", [MipCover(), GreedyCover(), LocalSearchCover()], ids=repr)
@pytest.mark.parametrize("row", list(itertools.product(CATEGORIES, repeat=3)))
def test_regressor_interval_matches_brute_force(
    categorical_regressor: FittedRegressor,
    row: tuple[str, ...],
    search: CoveringSolver,
    interval: Interval,
) -> None:
    ebm, _ = categorical_regressor
    instance = dict(zip(NAMES, row, strict=True))
    value = float(ebm.predict(frame_of_rows([instance]))[0])
    holds = _interval_answer(interval, value)
    facts = facts_module.compute_local_facts(
        ebm,
        instance,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0, decision=interval, search=search),
    )
    flip, _ = _brute_force_regressor(ebm, row, holds)
    assert (facts.counterfactual is None) == (flip is None)
    if flip is not None:
        assert facts.counterfactual is not None
        assert len(facts.counterfactual.changes) == flip
    sufficient = _sufficient_sizes(ebm, row, holds)
    reason = facts.sufficient_reason
    if not sufficient[ALL_FEATURES]:
        assert reason is None
        return
    assert reason is not None
    chosen = frozenset(NAMES.index(name) for name in reason.features)
    assert sufficient[chosen]
    # minimal: no feature can be removed
    assert not any(sufficient[chosen - {feature}] for feature in chosen)
    smallest = min(len(subset) for subset, ok in sufficient.items() if ok)
    assert reason.lower_bound <= smallest
    if reason.minimality.is_smallest:
        assert len(chosen) == smallest
    if isinstance(search, MipCover) and search.stop is Stop.OPTIMAL:
        assert reason.minimality.is_smallest
        assert len(chosen) == smallest


def test_regressor_prediction_is_the_predicted_value(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, data = categorical_regressor
    facts = facts_module.compute_local_facts(
        ebm,
        instance_of(data, 3),
        fact_types_module.LocalOptions(decision=Threshold(1.0)),
    )
    prediction = facts.prediction
    value = float(ebm.predict(data.iloc[[3]])[0])
    assert prediction.output == pytest.approx(value)
    assert prediction.score == pytest.approx(value)
    assert prediction.threshold == pytest.approx(1.0)
    assert prediction.output_kind is OutputKind.VALUE
    assert prediction.decision == ("above 1" if value > 1.0 else "at most 1")
    assert prediction.margin == pytest.approx(abs(value - 1.0))
    assert prediction.lower < value < prediction.upper
    assert prediction.upper - value == pytest.approx(value - prediction.lower)


def test_a_classifier_rejects_a_decision(categorical_ebm: FittedClassifier) -> None:
    ebm, _ = categorical_ebm
    with pytest.raises(OptionRangeError, match="Only a regressor accepts a decision"):
        facts_module.compute_local_facts(
            ebm,
            {"f0": "p", "f1": "p", "f2": "p"},
            fact_types_module.LocalOptions(decision=Threshold(0.0)),
        )


def _decision(limits: Sequence[float]) -> Threshold | Interval | None:
    """Build the question of a regressor from its limits.

    Returns
    -------
    Threshold | Interval | None
        ``None`` for no limit, a threshold for one limit, an interval for two.

    """
    if not limits:
        return None
    return Threshold(limits[0]) if len(limits) == 1 else Interval(limits[0], limits[1])


def _small_facts(
    templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    *,
    confidence_level: float,
    min_support: float,
) -> tuple[FittedModel, fact_types_module.LocalFacts]:
    model, limits = case
    ebm = build_ebm(templates, model)
    # the completions are not under test here, so the sample is tiny
    options = fact_types_module.LocalOptions(
        confidence_level=confidence_level,
        min_support=min_support,
        decision=_decision(limits),
        n_samples=8,
        max_steps=0,
    )
    return ebm, facts_module.compute_local_facts(ebm, model.instance, options)


@settings(max_examples=300, deadline=None)
@given(
    case=model_cases(),
    confidence_level=st.sampled_from(CONFIDENCE_LEVELS),
    min_support=st.sampled_from(MIN_SUPPORTS),
)
def test_the_facts_match_the_oracle_of_the_paper(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    confidence_level: float,
    min_support: float,
) -> None:
    # contract 1: the code computes what the paper proves, with scores that move and bins that are left out
    model, limits = case
    ebm, facts = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    assert model_scores(ebm, [model.instance])[0] == pytest.approx(model.output_scores(model.instance_bins))
    problem = oracle_problem(model, confidence_level=confidence_level, min_support=min_support, limits=limits)
    names = [feature_name(index) for index, _ in enumerate(model.instance_bins)]

    counterfactual = facts.counterfactual
    assert (None if counterfactual is None else len(counterfactual.changes)) == problem.smallest_flip()
    if counterfactual is not None:
        # a tie can give another set of the same size, so the oracle checks the set and not its members
        assert problem.flips([names.index(change.feature) for change in counterfactual.changes])
        assert counterfactual.minimality is MinimalityStatus.SORTING_PROOF
        assert counterfactual.lower_bound == len(counterfactual.changes)

    reason = facts.sufficient_reason
    assert (None if reason is None else len(reason.features)) == problem.smallest_sufficient()
    if reason is not None:
        guaranteed = problem.worst_case_margin([names.index(name) for name in reason.features])
        assert guaranteed > 0
        assert reason.guaranteed_margin == pytest.approx(guaranteed)
        assert reason.minimality.is_smallest


@settings(max_examples=200, deadline=None)
@given(
    case=model_cases(),
    levels=st.lists(st.sampled_from([0.0, 0.5, 0.8, 0.95, 0.99]), min_size=2, max_size=2, unique=True).map(sorted),
    min_support=st.sampled_from(MIN_SUPPORTS),
)
def test_more_confidence_never_makes_a_claim_easier(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    levels: list[float],
    min_support: float,
) -> None:
    # contract 7: more uncertainty never turns a failed claim into a valid one, and never shrinks a proved set
    _, low = _small_facts(small_templates, case, confidence_level=levels[0], min_support=min_support)
    _, high = _small_facts(small_templates, case, confidence_level=levels[1], min_support=min_support)
    assert low.prediction.decision == high.prediction.decision
    assert high.prediction.lower <= low.prediction.lower + TOLERANCE
    assert high.prediction.upper >= low.prediction.upper - TOLERANCE

    if high.counterfactual is not None:
        assert low.counterfactual is not None
        assert len(low.counterfactual.changes) <= len(high.counterfactual.changes)
    if high.sufficient_reason is not None:
        assert low.sufficient_reason is not None
        assert low.sufficient_reason.minimality.is_smallest
        assert high.sufficient_reason.minimality.is_smallest
        assert len(low.sufficient_reason.features) <= len(high.sufficient_reason.features)
        # sorting gives the set with the largest guaranteed margin of its size, so two sets of one size compare
        if case[0].kind == "binary" and len(low.sufficient_reason.features) == len(high.sufficient_reason.features):
            assert high.sufficient_reason.guaranteed_margin <= low.sufficient_reason.guaranteed_margin + TOLERANCE


def _tie_model(instance_score: float) -> SmallModel:
    """Build a binary model with two bins of one feature that matter: the instance and a bin with the score zero.

    Returns
    -------
    SmallModel
        A model whose logit is ``instance_score`` for the instance and exactly zero after the one possible change.

    """
    return SmallModel(
        kind="binary",
        scores=[[[instance_score], [0.0]], [[0.0], [0.0], [0.0]]],
        deviations=[[[0.0], [0.0]], [[0.0], [0.0], [0.0]]],
        weights=[[1.0, 1.0], [1.0, 1.0, 1.0]],
        intercepts=[0.0],
        instance_bins=[0, 0],
    )


@pytest.mark.parametrize("instance_score", [1.0, -1.0])
def test_a_change_that_lands_on_a_logit_of_zero_is_not_called_a_flip(
    small_templates: Templates,
    instance_score: float,
) -> None:
    # a logit of zero belongs to the first class. The facts claim a flip only below zero, so a tie is never a claim
    # for the side that the tie rule gives to the other decision.
    model = _tie_model(instance_score)
    options = fact_types_module.LocalOptions(confidence_level=0, min_support=0)
    facts = facts_module.compute_local_facts(build_ebm(small_templates, model), model.instance, options)
    assert facts.prediction.margin == pytest.approx(1.0)
    assert facts.counterfactual is None
    # the worst completion has the margin zero, and a guarantee needs a margin above zero
    assert facts.sufficient_reason is not None
    assert facts.sufficient_reason.features == [feature_name(0)]


def test_a_prediction_on_a_logit_of_zero_has_no_guarantee(small_templates: Templates) -> None:
    model = _tie_model(0.0)
    ebm = build_ebm(small_templates, model)
    facts = facts_module.compute_local_facts(
        ebm,
        model.instance,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0),
    )
    assert facts.prediction.decision == ebm.predict(frame_of_rows([model.instance]))[0] == ebm.classes_[0]
    assert facts.prediction.margin == pytest.approx(0.0)
    assert facts.counterfactual is None
    assert facts.sufficient_reason is None


@settings(max_examples=60, deadline=None)
@given(case=model_cases(), min_support=st.sampled_from(MIN_SUPPORTS))
def test_scores_without_deviation_give_the_same_claims_at_every_confidence(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    min_support: float,
) -> None:
    model, limits = case
    certain = attrs.evolve(
        model,
        deviations=[[[0.0 for _ in outputs] for outputs in feature] for feature in model.deviations],
    )
    _, nominal = _small_facts(small_templates, (certain, limits), confidence_level=0.0, min_support=min_support)
    _, strict = _small_facts(small_templates, (certain, limits), confidence_level=0.99, min_support=min_support)
    assert strict.counterfactual == nominal.counterfactual
    assert strict.sufficient_reason == nominal.sufficient_reason
    assert strict.prediction.lower == pytest.approx(strict.prediction.output)
    assert strict.prediction.upper == pytest.approx(strict.prediction.output)


@settings(max_examples=60, deadline=None)
@given(case=model_cases(), min_support=st.sampled_from(MIN_SUPPORTS))
def test_a_confidence_level_close_to_one_keeps_the_claims_of_the_paper(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    min_support: float,
) -> None:
    confidence_level = 1 - 1e-9
    model, limits = case
    _, facts = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    problem = oracle_problem(model, confidence_level=confidence_level, min_support=min_support, limits=limits)
    counterfactual, reason = facts.counterfactual, facts.sufficient_reason
    assert (None if counterfactual is None else len(counterfactual.changes)) == problem.smallest_flip()
    assert (None if reason is None else len(reason.features)) == problem.smallest_sufficient()
    assert math.isfinite(facts.prediction.lower)
    assert math.isfinite(facts.prediction.upper)
    assert facts.prediction.lower <= facts.prediction.output <= facts.prediction.upper


def _label_of(ebm: FittedModel, score: float, limits: Sequence[float]) -> str:
    """Name the answer of a regressor to its question for a predicted value.

    Returns
    -------
    str
        The label of the side of the value.

    """
    decision = _decision(limits) or Threshold(float(ebm.intercept_))
    return str(decision.side(score).label)


def _model_decision(ebm: FittedModel, row: Mapping[str, FeatureValue], limits: Sequence[float]) -> ClassLabel:
    """Ask the EBM for its decision on one input, through the API of interpret.

    Returns
    -------
    ClassLabel
        The predicted class, or the answer of a regressor to its question.

    """
    predicted = ebm.predict(frame_of_rows([row]))[0]
    if isinstance(ebm, ExplainableBoostingRegressor):
        return _label_of(ebm, float(predicted), limits)
    label: ClassLabel = predicted.item()
    return label


def _value_in(feature: str, bin_label: str) -> FeatureValue:
    """Pick a value inside the bin that a change names.

    Returns
    -------
    FeatureValue
        The category of a nominal feature, or the middle of the interval of a continuous feature.

    """
    if feature.startswith("n"):
        return bin_label
    low, high = (float(end) for end in bin_label.strip("[)").split(","))
    return (low + high) / 2


MODEL_CASE: Callable[[Callable[..., None]], Callable[..., None]] = given(
    case=model_cases(),
    confidence_level=st.sampled_from(CONFIDENCE_LEVELS),
    min_support=st.sampled_from(MIN_SUPPORTS),
)


@settings(max_examples=150, deadline=None)
@MODEL_CASE
def test_the_prediction_of_the_facts_agrees_with_the_fitted_model(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    confidence_level: float,
    min_support: float,
) -> None:
    # contract 5: the class, the scores and the values of the facts are those of the EBM
    model, limits = case
    ebm, facts = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    prediction = facts.prediction
    assert prediction.decision == _model_decision(ebm, model.instance, limits)
    scores = model_scores(ebm, [model.instance])[0]
    column = int(np.argmax(scores)) if model.kind == "multiclass" else 0
    # The graph of a multiclass model centres the scores of each bin over the classes (the multiclass scores proof).
    # The class score of the facts is then lower than decision_function by one number for all the classes, which
    # changes no margin and no probability. A model with one output has no centring.
    centring = 0.0
    if model.kind == "multiclass":
        centring = sum(
            statistics.fmean(model.scores[feature][bin_index]) for feature, bin_index in enumerate(model.instance_bins)
        )
    assert prediction.score == pytest.approx(scores[column] - centring)
    intercept = float(np.ravel(ebm.intercept_)[column])
    assert intercept + sum(contribution.score for contribution in facts.contributions) == pytest.approx(
        prediction.score,
    )
    assert prediction.output == pytest.approx(_model_output(ebm, model.instance, prediction.decision))
    assert {contribution.feature: contribution.value for contribution in facts.contributions} == {
        name: (f"{value:.4g}" if isinstance(value, float) else value) for name, value in model.instance.items()
    }
    _, again = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    assert again == facts


def _model_output(ebm: FittedModel, row: Mapping[str, FeatureValue], decision: ClassLabel) -> float:
    """Ask the EBM for its output on one input: the predicted value, or the probability of a class.

    Returns
    -------
    float
        The output.

    """
    frame = frame_of_rows([row])
    if isinstance(ebm, ExplainableBoostingRegressor):
        return float(ebm.predict(frame)[0])
    return float(ebm.predict_proba(frame)[0][list(ebm.classes_).index(decision)])


@settings(max_examples=150, deadline=None)
@MODEL_CASE
def test_the_changes_of_a_counterfactual_change_the_decision_of_the_fitted_model(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    confidence_level: float,
    min_support: float,
) -> None:
    # contract 5: a counterfactual flips the decision of the EBM, with a value inside each new bin
    model, limits = case
    ebm, facts = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    counterfactual = facts.counterfactual
    assume(counterfactual is not None)
    assert counterfactual is not None
    for change in counterfactual.changes:
        assert change.from_value == next(c.value for c in facts.contributions if c.feature == change.feature)
    changed = model.instance | {
        change.feature: _value_in(change.feature, change.to_value) for change in counterfactual.changes
    }
    assert _model_decision(ebm, changed, limits) != facts.prediction.decision
    if isinstance(ebm, ExplainableBoostingRegressor):
        old_value = model.output_scores(model.instance_bins)[0]
        new_value = _model_output(ebm, changed, counterfactual.new_decision)
        # the new decision of a regressor names the limit that the value crossed, and the side that it is on
        crossed = [limit for limit in limits or (model.intercepts[0],) if (old_value > limit) != (new_value > limit)]
        assert counterfactual.new_decision in {
            f"at most {limit:.4g}" if old_value > limit else f"above {limit:.4g}" for limit in crossed
        }
    else:
        assert counterfactual.new_decision == _model_decision(ebm, changed, limits)
    assert counterfactual.new_output == pytest.approx(_model_output(ebm, changed, counterfactual.new_decision))


@settings(max_examples=150, deadline=None)
@MODEL_CASE
def test_a_sufficient_reason_keeps_the_decision_of_the_fitted_model_for_every_completion(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    confidence_level: float,
    min_support: float,
) -> None:
    # contract 5: with the features of the reason fixed, each allowed completion keeps the decision of the EBM
    model, limits = case
    ebm, facts = _small_facts(small_templates, case, confidence_level=confidence_level, min_support=min_support)
    reason = facts.sufficient_reason
    assume(reason is not None)
    assert reason is not None
    free = [index for index, _ in enumerate(model.instance_bins) if feature_name(index) not in reason.features]
    allowed = [
        plausible_bins(
            model.weights[index],
            continuous=index % 2 == 1,
            min_support=min_support,
            instance_bin=model.instance_bins[index],
        )
        for index in free
    ]
    for bins in itertools.product(*allowed):
        completion = list(model.instance_bins)
        for index, bin_index in zip(free, bins, strict=True):
            completion[index] = bin_index
        assert _model_decision(ebm, model_row(completion), limits) == facts.prediction.decision


@pytest.fixture(scope="module")
def unsorted_ebm(fit_ebm: Callable[..., ExplainableBoostingClassifier]) -> FittedClassifier:
    """Fit a classifier on categories that are not in sorted order, with missing values in both columns.

    Returns
    -------
    FittedClassifier
        The model and its training data.

    """
    random_generator = np.random.default_rng(11)
    # "10" sorts before "9" as text and after it as a number, and "b" is not a number
    sizes = random_generator.choice(np.array(["9", "10", "b", "2"], dtype=object), 400)
    amounts = random_generator.normal(size=400)
    effect = np.select([sizes == "10", sizes == "2", sizes == "b"], [2.0, -2.0, 0.5], 0.0)
    target = (effect + amounts + random_generator.normal(scale=0.3, size=400) > 0).astype(int)
    sizes[:20] = None
    amounts[20:40] = np.nan
    data = frame_of_columns({"size": sizes, "amount": amounts})
    return fit_ebm(data, target, interactions=0, random_state=0), data


@pytest.mark.parametrize("row", range(40, 64))
def test_the_facts_align_with_unsorted_categories_and_a_model_with_missing_values(
    unsorted_ebm: FittedClassifier,
    row: int,
) -> None:
    ebm, data = unsorted_ebm
    instance = instance_of(data, row)
    facts = facts_module.compute_local_facts(
        ebm,
        instance,
        fact_types_module.LocalOptions(confidence_level=0, min_support=0),
    )
    frame = frame_of_rows([instance])
    assert facts.prediction.decision == ebm.predict(frame)[0]
    logits: np.ndarray = ebm.decision_function(frame)
    assert facts.prediction.score == pytest.approx(logits[0])
    values = {contribution.feature: contribution.value for contribution in facts.contributions}
    assert values == {"size": instance["size"], "amount": f"{instance['amount']:.4g}"}
    by_feature = {contribution.feature: contribution.score for contribution in facts.contributions}
    # the score of the category is the score that the EBM gives to it alone
    category_bin = ebm.bins_[0][0][str(instance["size"])]
    category_score = ebm.term_scores_[0][category_bin]
    assert by_feature["size"] == pytest.approx(category_score)
    counterfactual = facts.counterfactual
    assert counterfactual is not None
    changed = instance | {change.feature: _value_of_change(change) for change in counterfactual.changes}
    assert ebm.predict(frame_of_rows([changed]))[0] == counterfactual.new_decision != facts.prediction.decision


def _value_of_change(change: fact_types_module.Change) -> FeatureValue:
    if change.feature == "size":
        return change.to_value
    low, high = (float(end) for end in change.to_value.strip("[)").split(","))
    return (low + high) / 2


@pytest.mark.parametrize("missing", [{"size": None}, {"amount": float("nan")}])
def test_a_model_with_missing_values_in_training_still_needs_every_input(
    unsorted_ebm: FittedClassifier,
    missing: dict[str, FeatureValue],
) -> None:
    ebm, data = unsorted_ebm
    with pytest.raises(InvalidInstanceError, match="is missing"):
        facts_module.compute_local_facts(ebm, instance_of(data, 50) | missing)


def _completion_facts(
    templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    *,
    n_samples: int,
    seed: int = 0,
) -> fact_types_module.LocalFacts:
    model, limits = case
    options = fact_types_module.LocalOptions(n_samples=n_samples, seed=seed, decision=_decision(limits), max_steps=4)
    return facts_module.compute_local_facts(build_ebm(templates, model), model.instance, options)


def _share_error(exact: float, n_samples: int) -> float:
    # four binomial standard errors, and a few samples for a share so close to 0 or 1 that the count is a rare event
    return 4 * math.sqrt(max(0.0, exact * (1 - exact)) / n_samples) + 8 / n_samples


# the examples are fixed, so the bound on the sampling error holds or fails the same way in every run
@settings(max_examples=60, deadline=None, derandomize=True)
@given(case=model_cases(), n_samples=st.sampled_from([400, 4000]))
def test_each_completion_share_estimates_the_exact_weighted_count(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    n_samples: int,
) -> None:
    # contract 6: the shares are estimates of the count under independent inputs with the training weights of every bin
    model, limits = case
    facts = _completion_facts(small_templates, case, n_samples=n_samples)
    names = [feature_name(index) for index, _ in enumerate(model.instance_bins)]
    assert [len(step.fixed) for step in facts.completions] == list(range(len(names) + 1))
    for step in facts.completions:
        exact = exact_share(model, [names.index(name) for name in step.fixed], limits)
        assert abs(step.share - exact) <= _share_error(exact, n_samples)
    assert facts.completions[-1].share == pytest.approx(1.0)


# Feature 0 decides the class, and its bin 0 has 95 % of the training weight. Sampling the bins uniformly gives 50 %.
SKEWED_MODEL: SmallModel = SmallModel(
    kind="binary",
    scores=[[[1.0], [-1.0]], [[0.0], [0.0], [0.0]]],
    deviations=[[[0.0], [0.0]], [[0.0], [0.0], [0.0]]],
    weights=[[19.0, 1.0], [1.0, 1.0, 1.0]],
    intercepts=[1 / 16],
    instance_bins=[0, 0],
)


def test_the_completions_sample_the_bins_by_their_training_weight(small_templates: Templates) -> None:
    n_samples = 4000
    facts = _completion_facts(small_templates, (SKEWED_MODEL, ()), n_samples=n_samples)
    assert exact_share(SKEWED_MODEL, []) == pytest.approx(0.95)
    assert abs(facts.completions[0].share - 0.95) <= _share_error(0.95, n_samples)
    assert abs(facts.completions[0].share - 0.5) > _share_error(0.5, n_samples)


# The instance is positive. With nothing known, 85 % of the completions are positive, because feature 1 is mostly in
# its high bin. Feature 1 is revealed first, and its bin of the instance leaves the class to feature 0: 50 %.
FALLING_MODEL: SmallModel = SmallModel(
    kind="binary",
    scores=[[[1.0], [-1.0]], [[0.0], [3.0], [-3.0]]],
    deviations=[[[0.0], [0.0]], [[0.0], [0.0], [0.0]]],
    weights=[[1.0, 1.0], [2.0, 16.0, 2.0]],
    intercepts=[1 / 16],
    instance_bins=[0, 0],
)


def test_a_completion_share_can_fall_when_an_input_becomes_known(small_templates: Templates) -> None:
    # a fall is a correct result, so no text and no test may claim that the share grows
    n_samples = 4000
    facts = _completion_facts(small_templates, (FALLING_MODEL, ()), n_samples=n_samples)
    assert facts.completions[1].fixed == [feature_name(1)]
    assert exact_share(FALLING_MODEL, []) == pytest.approx(0.85)
    assert exact_share(FALLING_MODEL, [1]) == pytest.approx(0.5)
    assert abs(facts.completions[0].share - 0.85) <= _share_error(0.85, n_samples)
    assert abs(facts.completions[1].share - 0.5) <= _share_error(0.5, n_samples)
    assert facts.completions[1].share < facts.completions[0].share


def test_the_seed_fixes_the_draws_and_the_sample_count_sets_the_precision(small_templates: Templates) -> None:
    case = (FALLING_MODEL, ())
    first = _completion_facts(small_templates, case, n_samples=400, seed=1)
    assert _completion_facts(small_templates, case, n_samples=400, seed=1).completions == first.completions
    other_seed = _completion_facts(small_templates, case, n_samples=400, seed=2)
    assert other_seed.completions != first.completions
    # more samples give another estimate of the same share, so both are near the exact count and need not be equal
    for n_samples in (400, 40_000):
        share = _completion_facts(small_templates, case, n_samples=n_samples, seed=1).completions[0].share
        assert abs(share - 0.85) <= _share_error(0.85, n_samples)
