"""Tests for the deterministic local explanations."""

import functools
import itertools
import math
import re
from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal

import attrs
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ebaml.core.custom_types import FeatureValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import InvalidInstanceError, OptionRangeError, UnsupportedModelError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.local import facts as local
from ebaml.local.covering_backend import Stop
from ebaml.local.covering_problem import Cover, CoveringProblem
from ebaml.local.covering_solvers import CoveringSolver, GreedyCover, LocalSearchCover, MipCover
from ebaml.local.decisions import Interval, Threshold
from ebaml.local.local_kinds import MulticlassLocal, Outcome, OutputKind
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver
from tests.unit.ebaml.conftest import (
    MISSING_VALUE,
    FittedClassifier,
    FittedRegressor,
    PandasFrame,
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
    CONFIDENCE_LEVELS,
    MIN_SUPPORTS,
    FittedModel,
    SmallModel,
    Templates,
    build_ebm,
    feature_name,
    model_cases,
    model_scores,
    oracle_problem,
)

CATEGORIES = ["p", "q", "r"]
NAMES = ["f0", "f1", "f2"]
FEATURE_INDEXES: tuple[int, ...] = tuple(index for index, _ in enumerate(NAMES))
ALL_FEATURES = frozenset(FEATURE_INDEXES)


@pytest.fixture(scope="module")
def categorical_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> FittedClassifier:
    rng = np.random.default_rng(1)
    data = frame_of_columns({name: rng.choice(CATEGORIES, 900) for name in NAMES})
    effect = data["f0"].map({"p": 1.5, "q": 0.0, "r": -1.5}) + data["f1"].map({"p": 0.5, "q": 0.0, "r": -0.5})
    effect += data["f2"].map({"p": 0.2, "q": 0.0, "r": -0.2}) + rng.normal(size=900) * 0.3
    target = (effect > 0).astype(int)
    return fit_ebm(data, target, interactions=0, random_state=0), data


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
    # the brute force shows that no smaller set is sufficient, which is what minimum claims
    assert facts.sufficient_reason.minimum


def test_counterfactual_really_flips(categorical_ebm: FittedClassifier) -> None:
    ebm, _ = categorical_ebm
    row = {"f0": "p", "f1": "p", "f2": "p"}
    facts = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0, min_support=0))
    assert facts.counterfactual is not None
    changed = {**row, **{c.feature: c.to_value for c in facts.counterfactual.changes}}
    assert ebm.predict(frame_of_rows([changed]))[0] != ebm.predict(frame_of_rows([row]))[0]


def test_prediction_matches_model(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 3)
    facts = local.compute_local_facts(ebm, row)
    logits: np.ndarray = ebm.decision_function(data.iloc[[3]])
    assert facts.prediction.score == pytest.approx(logits[0])
    assert facts.prediction.lower <= facts.prediction.output <= facts.prediction.upper
    assert facts.prediction.decision == ebm.predict(data.iloc[[3]])[0]


def test_confidence_makes_claims_harder(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 0)
    loose = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0))
    strict = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0.99))
    assert loose.sufficient_reason is not None
    assert strict.sufficient_reason is None or len(strict.sufficient_reason.features) >= len(
        loose.sufficient_reason.features,
    )
    assert loose.counterfactual is not None
    assert strict.counterfactual is None or len(strict.counterfactual.changes) >= len(loose.counterfactual.changes)


def test_completions_are_reproducible_and_grow(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 5)
    first = local.compute_local_facts(ebm, row, local.LocalOptions(seed=7, max_steps=2))
    again = local.compute_local_facts(ebm, row, local.LocalOptions(seed=7, max_steps=2))
    assert first.completions == again.completions
    assert [len(s.fixed) for s in first.completions] == [0, 1, 2]
    assert all(0 <= s.share <= 1 for s in first.completions)
    assert first.completions[-1].share >= first.completions[0].share


def test_sufficient_reason_decides_outcome(continuous_ebm: FittedClassifier) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 0)
    facts = local.compute_local_facts(ebm, row, local.LocalOptions(confidence_level=0, min_support=0))
    assert facts.sufficient_reason is not None
    free = [n for n in row if n not in facts.sufficient_reason.features]
    rng = np.random.default_rng(0)
    for _ in range(50):
        other = data.iloc[int(rng.integers(len(data)))].to_dict()
        mixed = {**row, **{n: other[n] for n in free}}
        assert ebm.predict(frame_of_rows([mixed]))[0] == facts.prediction.decision


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
        local.compute_local_facts(continuous_ebm[0], instance)


def test_rejects_interactions_and_bad_confidence(
    continuous_ebm: FittedClassifier,
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> None:
    ebm, data = continuous_ebm
    with pytest.raises(OptionRangeError, match="confidence_level"):
        local.compute_local_facts(ebm, instance_of(data, 0), local.LocalOptions(confidence_level=1))
    rng = np.random.default_rng(0)
    frame = frame_of_matrix(rng.normal(size=(300, 3)), list("abc"))
    with_interactions = fit_ebm(frame, (frame["a"] * frame["b"] > 0).astype(int), interactions=1, random_state=0)
    with pytest.raises(UnsupportedModelError, match="interactions"):
        local.compute_local_facts(with_interactions, instance_of(frame, 0))


def _feature(
    scores: list[float],
    bin_index: int,
    *,
    std: float = 0.0,
    plausible: list[bool] | None = None,
) -> local._Feature:
    graph = EBMGraph(
        "f",
        "nominal",
        [str(bin_index) for bin_index, _ in enumerate(scores)],
        scores,
        [std] * len(scores),
    )
    weights = np.ones(len(scores))
    return local._Feature("f", graph, weights, bin_index, "v", np.array(plausible or [True] * len(scores)))


def _probability_after(margin: float, _replaced: Sequence[tuple[local._Feature, int]]) -> Outcome:
    """Return the other class, with its probability from the log-odds margin of the class that holds.

    Returns
    -------
    Outcome
        The other class and its probability.

    """
    return Outcome("other", 1 / (1 + math.exp(margin)))


def test_counterfactual_is_none_when_no_change_flips() -> None:
    features = [_feature([1.0, 0.9], 0), _feature([1.0, 0.8], 0)]
    assert (
        local._counterfactual(
            features,
            intercept=0.0,
            sign=1,
            z=0.0,
            after=_probability_after,
        )
        is None
    )


def test_counterfactual_skips_unplausible_bins() -> None:
    features = [_feature([1.0, -5.0], 0, plausible=[True, False]), _feature([1.0, -5.0], 0)]
    counterfactual = local._counterfactual(
        features,
        intercept=0.0,
        sign=1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert [c.feature for c in counterfactual.changes] == ["f"]
    assert len(counterfactual.changes) == 1


def test_sufficient_reason_is_none_when_not_even_all_features_guarantee() -> None:
    features = [_feature([0.5, -2.0], 0, std=1.0), _feature([0.5, -2.0], 0, std=1.0)]
    assert local._sufficient_reason(features, intercept=0.0, sign=1, z=1.96) is None


@functools.cache
def _multiclass_model() -> FittedClassifier:
    """Fit a three-class model on a small grid, so that each feature has few bins and a search can try every one.

    Returns
    -------
    FittedClassifier
        The model and its training data.

    """
    rng = np.random.default_rng(0)
    frame = frame_of_matrix(rng.integers(0, 4, size=(240, 3)).astype(float), ["a", "b", "c"])
    target = (frame["a"] + frame["b"] + rng.integers(0, 2, 240) > 3).astype(int) + (frame["c"] > 2).astype(int)
    model = ExplainableBoostingClassifier(interactions=0, random_state=0).fit(frame, target)
    return model, frame


def _margin_views(
    features: Sequence[local._Feature],
    winner: int,
    intercepts: Sequence[float],
) -> list[tuple[float, list[local._Feature]]]:
    """Return the intercept and the views of every margin of the class ``winner`` against its rivals.

    Returns
    -------
    list[tuple[float, list[local._Feature]]]
        One pair for each rival class.

    """
    return [
        (intercepts[winner] - intercepts[rival], [feature.against((winner, rival)) for feature in features])
        for rival, _ in enumerate(intercepts)
        if rival != winner
    ]


def _brute_force_counterfactual_size(views: Sequence[tuple[float, list[local._Feature]]]) -> int | None:
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


def _brute_force_reason_size(views: Sequence[tuple[float, list[local._Feature]]]) -> int | None:
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
    model, frame = _multiclass_model()
    options = local.LocalOptions(confidence_level=0.0, min_support=0.0)
    facts = local.compute_local_facts(model, instance_of(frame, row), options)
    features = local._features(model, instance_of(frame, row), 0.0)
    intercepts = np.ravel(model.intercept_).tolist()
    winner = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    views = _margin_views(features, winner, intercepts)
    expected_flip = _brute_force_counterfactual_size(views)
    expected_reason = _brute_force_reason_size(views)
    assert facts.prediction.decision == model.classes_[winner]
    assert (None if facts.counterfactual is None else len(facts.counterfactual.changes)) == expected_flip
    assert (None if facts.sufficient_reason is None else len(facts.sufficient_reason.features)) == expected_reason
    if facts.sufficient_reason is not None:
        assert facts.sufficient_reason.minimum


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239))
def test_multiclass_counterfactual_changes_the_predicted_class(row: int) -> None:
    model, frame = _multiclass_model()
    instance = instance_of(frame, row)
    facts = local.compute_local_facts(model, instance, local.LocalOptions(confidence_level=0.0, min_support=0.0))
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


def _changed_row(instance: dict[str, FeatureValue], counterfactual: local.Counterfactual) -> PandasFrame:
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
    model, frame = _multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = local.LocalOptions(confidence_level=0.0, min_support=0.0, target_class=int(model.classes_[target]))
    facts = local.compute_local_facts(model, instance, options)
    if facts.counterfactual is None:
        return
    counterfactual = facts.counterfactual
    assert counterfactual.new_decision == model.classes_[target]
    assert model.predict(_changed_row(instance, counterfactual))[0] == model.classes_[target]
    assert 1 <= len(counterfactual.changes) <= 3


@settings(max_examples=40, deadline=None)
@given(row=st.integers(0, 239), target=st.sampled_from([0, 1, 2]))
def test_a_target_counterfactual_has_no_feature_to_revert(row: int, target: int) -> None:
    model, frame = _multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = local.LocalOptions(confidence_level=0.0, min_support=0.0, target_class=int(model.classes_[target]))
    counterfactual = local.compute_local_facts(model, instance, options).counterfactual
    assume(counterfactual is not None)
    assert counterfactual is not None
    for change in counterfactual.changes:
        fewer = attrs.evolve(counterfactual, changes=[other for other in counterfactual.changes if other != change])
        assert model.predict(_changed_row(instance, fewer))[0] != model.classes_[target]


def _brute_force_target_size(features: Sequence[local._Feature], scores: Sequence[float], target: int) -> int | None:
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
    model, frame = _multiclass_model()
    instance = instance_of(frame, row)
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    assume(target != predicted)
    options = local.LocalOptions(
        confidence_level=0.0,
        min_support=0.0,
        target_class=int(model.classes_[target]),
        search=search,
    )
    counterfactual = local.compute_local_facts(model, instance, options).counterfactual
    features = local._features(model, instance, 0.0)
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
    if counterfactual.minimum:
        assert len(counterfactual.changes) == smallest
    if isinstance(search, MipCover) and search.stop is Stop.OPTIMAL:
        assert counterfactual.minimum
        assert len(counterfactual.changes) == smallest


def _target_feature(name: str, rows: list[list[float]]) -> local._Feature:
    # a feature of a three-class model with one row of scores for each bin, the instance in bin 0, and no deviation
    graph = EBMGraph(
        name,
        "nominal",
        [f"{name}{bin_index}" for bin_index, _ in enumerate(rows)],
        rows,
        [[0.0] * 3] * len(rows),
        classes=("c0", "c1", "c2"),
    )
    return local._Feature(name, graph, np.ones(len(rows)), 0, f"{name}0", np.ones(len(rows), dtype=bool))


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
    totals = local._Totals(scores, intercepts, 0.0)
    assert _brute_force_target_size(features, scores, 0) == 1
    options = local.LocalOptions(search=MipCover(OrToolsBackend(solver)))
    counterfactual = local._target_counterfactual(features, frame, margins, totals, options=options)
    assert counterfactual is not None
    assert counterfactual.new_decision == "c0"
    assert [(change.feature, change.to_value) for change in counterfactual.changes] == [("f0", "f01")]
    assert counterfactual.minimum is True
    assert counterfactual.lower_bound == 1


@attrs.frozen
class _SeedSpy:
    """A ``CoveringSolver`` that runs the greedy and records the seed of each call."""

    seeds: list[int] = attrs.field(factory=list)

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Record the seed, and solve with the greedy.

        Returns
        -------
        Cover | None
            The answer of ``GreedyCover``.

        """
        self.seeds.append(seed)
        return GreedyCover().solve(problem, seed=seed)


def test_the_searches_get_the_seed_of_the_options(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, _ = categorical_regressor
    spy = _SeedSpy()
    options = local.LocalOptions(confidence_level=0, min_support=0, decision=Interval(-1.5, 1.5), seed=13, search=spy)
    local.compute_local_facts(ebm, {"f0": "q", "f1": "q", "f2": "q"}, options)
    assert spy.seeds == [13]


def test_a_target_class_is_none_when_no_choice_of_bins_reaches_it() -> None:
    model, frame = _multiclass_model()
    row = 0
    predicted = int(np.argmax(model.decision_function(frame.iloc[[row]])))
    target = (predicted + 1) % 3
    options = local.LocalOptions(confidence_level=0.999999, min_support=0.0, target_class=int(model.classes_[target]))
    facts = local.compute_local_facts(model, instance_of(frame, row), options)
    assert facts.counterfactual is None


def test_a_target_class_that_the_model_does_not_have_is_rejected() -> None:
    model, frame = _multiclass_model()
    with pytest.raises(OptionRangeError, match="is not a class of the model"):
        local.compute_local_facts(model, instance_of(frame, 0), local.LocalOptions(target_class=99))


def test_a_binary_classifier_rejects_a_target_class(fit_ebm: Callable[..., ExplainableBoostingClassifier]) -> None:
    rng = np.random.default_rng(0)
    frame = frame_of_matrix(rng.normal(size=(300, 2)), ["a", "b"])
    ebm = fit_ebm(frame, (frame["a"] > 0).astype(int), interactions=0, random_state=0)
    with pytest.raises(OptionRangeError, match="needs a multiclass classifier"):
        local.compute_local_facts(ebm, instance_of(frame, 0), local.LocalOptions(target_class=1))


def test_multiclass_prediction_reports_the_predicted_class() -> None:
    model, frame = _multiclass_model()
    facts = local.compute_local_facts(model, instance_of(frame, 0), local.LocalOptions(confidence_level=0.0))
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


def test_multiclass_completions_fall_to_the_share_of_the_class() -> None:
    model, frame = _multiclass_model()
    options = local.LocalOptions(confidence_level=0.0, max_steps=3, n_samples=2000)
    facts = local.compute_local_facts(model, instance_of(frame, 0), options)
    assert facts.completions[-1].share >= facts.completions[0].share
    assert 0.0 <= facts.completions[0].share <= 1.0


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
    x_vals = [f"{name}{bin_index}" for bin_index, _ in enumerate(scores)]
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


def test_bin_index_of_a_category_is_its_position() -> None:
    graph = EBMGraph("f", "categorical", ["a", "b", "c"], [0.0] * 3, [0.0] * 3)
    assert local._bin_index(fake_model(), 0, graph, "b") == 1


@pytest.mark.parametrize(
    ("value", "category"),
    [
        (2, "2"),
        (2.0, "2"),
        (np.int64(3), "3"),
        (0.5, "0.5"),
        (np.float32(1.5), "1.5"),
        (True, "1"),
        (np.False_, "0"),
        ("x", "x"),
        (np.str_("y"), "y"),
    ],
)
def test_category_writes_a_value_as_interpret_writes_its_categories(value: FeatureValue, category: str) -> None:
    assert local._category(value, "f") == category


def test_category_rejects_a_value_that_cannot_be_a_category() -> None:
    with pytest.raises(
        InvalidInstanceError,
        match=r"^The value np.complex128\(1j\) of feature 'f' cannot be a category\.$",
    ):
        local._category(np.complex128(1j), "f")


def test_bin_index_of_a_number_finds_its_category() -> None:
    # interpret keeps the categories 1, 2 and 3 as text, so the integer 2 was an unknown category
    graph = EBMGraph("f", "nominal", ["1", "2", "3"], [0.0] * 3, [0.0] * 3)
    assert local._bin_index(fake_model(), 0, graph, np.int64(2)) == 1


def test_bin_index_names_the_value_of_an_unknown_category() -> None:
    graph = EBMGraph("f", "nominal", ["1", "2"], [0.0] * 2, [0.0] * 2)
    with pytest.raises(InvalidInstanceError, match=r"^Unknown category 7 for feature 'f'\.$"):
        local._bin_index(fake_model(), 0, graph, 7)


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
        facts = local.compute_local_facts(ebm, instance_of(data, index))
        probabilities = ebm.predict_proba(data.iloc[[index]])[0]
        assert facts.prediction.decision == ebm.predict(data.iloc[[index]])[0]
        assert facts.prediction.output == pytest.approx(probabilities.max())


def test_bin_index_of_a_number_uses_the_first_cuts_and_the_right_side() -> None:
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], [0.0] * 3, [0.0] * 3)
    # the second array stands for a finer resolution that the function must not use
    ebm = fake_model(bins_=[[np.array([1.0, 2.0]), np.array([5.0])]])
    assert local._bin_index(ebm, 0, graph, 1.5) == 1
    assert local._bin_index(ebm, 0, graph, 2.0) == 2  # a value on a cut belongs to the bin on its right


@pytest.mark.parametrize("value", ["abc", "", np.str_("1,5"), np.datetime64("2020-01-01")])
def test_bin_index_of_a_continuous_feature_rejects_a_value_that_is_not_a_number(value: FeatureValue) -> None:
    # float() raised a bare ValueError or TypeError, which is not an EBaMlError, so a caller could not catch it as one
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0)], [0.0] * 2, [0.0] * 2)
    ebm = fake_model(bins_=[[np.array([1.0])]])
    with pytest.raises(InvalidInstanceError, match=r"^Feature 'f' is continuous, but its value .* is not a number\.$"):
        local._bin_index(ebm, 0, graph, value)


@pytest.mark.parametrize(("value", "expected"), [("1.5", 1), (np.str_("0.5"), 0), (True, 1), (Decimal("0.5"), 0)])
def test_bin_index_of_a_continuous_feature_accepts_a_value_that_float_reads(value: FeatureValue, expected: int) -> None:
    graph = EBMGraph("f", "continuous", [(0.0, 1.0), (1.0, 2.0)], [0.0] * 2, [0.0] * 2)
    ebm = fake_model(bins_=[[np.array([1.0])]])
    assert local._bin_index(ebm, 0, graph, value) == expected


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


def _model_features(
    *,
    term_features: list[tuple[int, ...]],
    classes: list[str],
    link: str = "logit",
) -> ExplainableBoostingClassifier:
    return fake_model(
        feature_names_in_=["a", "b"],
        term_features_=term_features,
        classes_=np.array(classes),
        link_=link,
    )


@pytest.mark.parametrize(
    "term_features",
    [[(0,), (1,), (0, 1)], [(0, 1), (1,)]],
    ids=["extra term", "interaction"],
)
def test_features_reject_interactions(term_features: list[tuple[int, ...]]) -> None:
    with pytest.raises(UnsupportedModelError, match="interactions"):
        local._features(_model_features(term_features=term_features, classes=["n", "y"]), {}, 0.01)


@pytest.mark.parametrize(
    ("term_features", "missing"),
    [([(0,)], "['b']"), ([(1,)], "['a']"), ([(), (0,)], "['b']"), ([], "['a', 'b']")],
    ids=["last feature", "first feature", "empty term", "no term"],
)
def test_features_name_the_features_without_a_main_effect_term(
    term_features: list[tuple[int, ...]],
    missing: str,
) -> None:
    # a model fit with exclude, or after remove_terms, was rejected with a message about interactions it did not have
    model = _model_features(term_features=term_features, classes=["n", "y"])
    expected = rf"main-effect term for every feature\. These features have none: {re.escape(missing)}\.$"
    with pytest.raises(UnsupportedModelError, match=expected):
        local._features(model, {}, 0.01)


def test_features_read_the_bin_weights_of_the_term_of_each_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    # the terms are in another order than the features, so the weights of feature "a" are those of the second term
    graph = EBMGraph("a", "nominal", ["p", "q"], [0.0, 0.0], [0.0, 0.0])
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(returning(graph)))
    ebm = fake_model(
        feature_names_in_=["a", "b"],
        term_features_=[(1,), (0,)],
        classes_=np.array(["n", "y"]),
        link_="logit",
        bin_weights_=[np.array([0.0, 9.0, 9.0, 0.0]), np.array([0.0, 1.0, 3.0, 0.0])],
    )
    features = local._features(ebm, {"a": "p", "b": "p"}, 0.01)
    assert features[0].weights.tolist() == [1.0, 3.0]


def test_facts_reject_a_model_kind_that_has_no_local_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "model_kind", returning(UnknownKind()))
    with pytest.raises(UnsupportedModelError, match=r"^Local explanations need a classifier or a regressor\. Got a"):
        local.compute_local_facts(_model_features(term_features=[(0,)], classes=["a", "b"], link="logit"), {})


@pytest.mark.parametrize(
    ("classes", "link", "detail"),
    [
        (["only"], "logit", "An EBM with the logit link needs two classes"),
        (["a", "b"], "probit", "EBaMl reads the links"),
    ],
)
def test_facts_reject_other_than_a_binary_classifier_or_a_regressor(classes: list[str], link: str, detail: str) -> None:
    model = _model_features(term_features=[(0,), (1,)], classes=classes, link=link)
    with pytest.raises(
        UnsupportedModelError,
        match=rf"^Local explanations need a classifier or a regressor\. {detail}",
    ):
        local.compute_local_facts(model, {})


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
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(returning(graph)))
    ebm = fake_model(
        feature_names_in_=["a"],
        term_features_=[(0,)],
        classes_=np.array(["n", "y"]),
        link_="logit",
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
    counterfactual = local._counterfactual(
        features,
        intercept=0.5,
        sign=1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.changes == [local.Change("f1", "f10", "f11", -3.0)]
    # the new logit is 0.5 + 1.5 - 3 = -1, so the new class (negative) has the probability sigmoid(1)
    assert counterfactual.new_output == pytest.approx(_sig(1.0))


def test_counterfactual_of_the_negative_class() -> None:
    features = [_hand("f1", [-1.0, 2.0]), _hand("f2", [-0.5, 0.25])]
    counterfactual = local._counterfactual(
        features,
        intercept=-0.5,
        sign=-1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.changes == [local.Change("f1", "f10", "f11", 3.0)]
    assert counterfactual.new_output == pytest.approx(_sig(1.0))  # the new logit is +1


def test_counterfactual_uses_as_many_changes_as_it_takes() -> None:
    features = [_hand("f1", [1.0, 0.0]), _hand("f2", [1.0, 0.0]), _hand("f3", [0.5, -1.0])]
    # the margin is 2.5 and the drops are 1, 1 and 1.5: the margin is 1 and then 0 (a tie, not a flip) and then -1
    counterfactual = local._counterfactual(
        features,
        intercept=0.0,
        sign=1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f3", "f1", "f2"]


def test_counterfactual_with_two_changes() -> None:
    features = [_hand("f1", [1.0, 0.0]), _hand("f2", [1.0, -0.5]), _hand("f3", [1.0, 0.5])]
    # the margin is 3 and the drops are 1, 1.5 and 0.5: after the two largest drops it is 0.5, so it is not yet flipped
    assert (
        local._counterfactual(
            features,
            intercept=0.0,
            sign=1,
            z=0.0,
            after=_probability_after,
        )
        is None
    )
    features[0] = _hand("f1", [1.0, -3.0])
    # the drops are now 4, 1.5 and 0.5, and the margin 3 flips after the first; with a margin of 4.5 it takes two
    counterfactual = local._counterfactual(
        features,
        intercept=1.5,
        sign=1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert [change.feature for change in counterfactual.changes] == ["f1", "f2"]


def test_counterfactual_probability_of_the_new_class() -> None:
    # the margin is 2.5 and the drop is 4, so the new logit is -1.5 and the new (negative) class has sigmoid(1.5)
    counterfactual = local._counterfactual(
        [_hand("f", [1.0, -3.0])],
        intercept=1.5,
        sign=1,
        z=0.0,
        after=_probability_after,
    )
    assert counterfactual is not None
    assert counterfactual.new_output == pytest.approx(_sig(1.5))


def test_counterfactual_is_none_for_a_tie() -> None:
    assert (
        local._counterfactual(
            [_hand("f", [1.0, 0.0])],
            intercept=0.0,
            sign=1,
            z=0.0,
            after=_probability_after,
        )
        is None
    )
    # the margin counts the score of the feature itself, not just its sign
    assert (
        local._counterfactual(
            [_hand("f", [2.0, 0.0])],
            intercept=0.0,
            sign=1,
            z=0.0,
            after=_probability_after,
        )
        is None
    )


def test_counterfactual_asks_for_the_worst_case_of_the_other_class() -> None:
    feature = _hand("f", [1.0, -2.0], stds=[0.5, 0.5])
    # with z = 1 the scores are raised by 0.5, so the drop is 1 + 1.5 = 2.5, and the margin 2.8 is not enough
    assert (
        local._counterfactual(
            [feature],
            intercept=1.8,
            sign=1,
            z=1.0,
            after=_probability_after,
        )
        is None
    )


def test_fixing_order_counts_what_a_known_value_guarantees() -> None:
    first = _hand("f1", [2.0, -1.0], stds=[1.0, 0.0])
    second = _hand("f2", [1.5, 0.0])
    # at z = 1 the guaranteed scores of f1 are 1 and -1, so knowing its value gains 2; f2 gains 1.5
    order = local._fixing_order([second, first], sign=1, z=1.0)
    assert [(step.gain, step.feature.name) for step in order] == [(2.0, "f1"), (1.5, "f2")]


def test_sufficient_reason_fixes_the_largest_gains_first() -> None:
    features = [_hand("f3", [0.5, 0.0]), _hand("f1", [2.0, -1.0]), _hand("f2", [1.0, -0.5])]
    # the worst case is -1.5 (-1, -0.5 and 0); the gains are 3 (f1), 1.5 (f2) and 0.5 (f3)
    reason = local._sufficient_reason(features, intercept=0.0, sign=1, z=0.0)
    assert reason == local.SufficientReason(["f1"], 1.5, minimum=True, lower_bound=1)


def test_sufficient_reason_needs_no_feature_when_the_outcome_is_certain() -> None:
    # even the lowest score keeps the margin above 0, so no input has to be fixed
    assert local._sufficient_reason([_hand("f", [1.0, 2.0])], 0.0, 1, 0.0) == local.SufficientReason(
        [],
        1.0,
        minimum=True,
        lower_bound=0,
    )


def test_sufficient_reason_needs_a_margin_above_zero() -> None:
    # the worst case is exactly 0: knowing the value gains 1, and the margin is 0 before and 1 after
    assert local._sufficient_reason([_hand("f", [0.0, 1.0], 1)], 0.0, 1, 0.0) == local.SufficientReason(
        ["f"],
        1.0,
        minimum=True,
        lower_bound=1,
    )
    # when the margin stays at 0, the outcome is not certain
    assert local._sufficient_reason([_hand("f", [0.0, 0.0])], 0.0, 1, 0.0) is None


# the share of completions with the same prediction, without and with the value of the one feature known
def _completion_shares(cur: float, sampled: float, *, intercept: float = 0.0, sign: int = 1) -> list[float]:
    feature = _hand("f", [cur, sampled], weights=[0.0, 1.0])  # the sample always falls in the second bin
    options = local.LocalOptions(n_samples=5, max_steps=1)
    return [step.share for step in local._completions([feature], [feature], [local._Bound(intercept, sign)], options)]


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


@pytest.mark.parametrize("max_steps", [1, 3])
def test_completions_reveal_max_steps_features(max_steps: int) -> None:
    features = [_hand(f"f{i}", [1.0, 1.0]) for i in range(4)]
    options = local.LocalOptions(n_samples=5, max_steps=max_steps)
    steps = local._completions(features, features, [local._Bound(0.0, 1)], options)
    assert [len(step.fixed) for step in steps] == list(range(max_steps + 1))


def _patch_features(monkeypatch: pytest.MonkeyPatch, features: list[local._Feature]) -> None:
    monkeypatch.setattr(local, "_features", returning(features))


def _hand_model(intercept: float = 0.25) -> ExplainableBoostingClassifier:
    return fake_model(intercept_=np.array([intercept, 9.0]), classes_=np.array(["no", "yes"]), link_="logit")


Z95 = 1.959963984540054


def test_facts_of_a_positive_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [_hand("b", [-0.5, 0.5], stds=[0.4, 0.3]), _hand("a", [1.0, -1.0], stds=[0.3, 0.4])]
    _patch_features(monkeypatch, features)
    facts = local.compute_local_facts(_hand_model(), {})
    prediction = facts.prediction
    logit, std = 0.25 - 0.5 + 1.0, 0.5  # the standard deviation is sqrt(0.4**2 + 0.3**2)
    assert prediction.decision == "yes"
    assert prediction.score == pytest.approx(logit)
    assert prediction.score_std == pytest.approx(std)
    assert prediction.output == pytest.approx(_sig(logit))
    assert prediction.margin == pytest.approx(logit)
    assert prediction.lower == pytest.approx(_sig(logit - Z95 * std))
    assert prediction.upper == pytest.approx(_sig(logit + Z95 * std))
    assert facts.confidence_level == pytest.approx(0.95)
    # the largest effect first
    assert facts.contributions == [local.Contribution("a", "a0", 1.0, 0.3), local.Contribution("b", "b0", -0.5, 0.4)]
    assert facts.counterfactual == local._counterfactual(
        features,
        0.25,
        1,
        Z95,
        after=lambda margin, _replaced: Outcome("no", _probability_after(margin, _replaced).output),
    )
    assert facts.sufficient_reason == local._sufficient_reason(features, 0.25, 1, Z95)


def test_facts_of_a_negative_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [_hand("a", [-1.0, 1.0], stds=[0.3, 0.4]), _hand("b", [0.5, -0.5], stds=[0.4, 0.3], bin_index=1)]
    _patch_features(monkeypatch, features)
    facts = local.compute_local_facts(_hand_model(-0.25), {}, local.LocalOptions(confidence_level=0.9, seed=3))
    prediction = facts.prediction
    logit, std, z = -0.25 - 1.0 - 0.5, math.hypot(0.3, 0.3), 1.6448536269514722
    assert prediction.decision == "no"
    assert prediction.output == pytest.approx(1 - _sig(logit))
    # the margin is the log-odds of the predicted class, so it is positive for the negative class too
    assert prediction.margin == pytest.approx(-logit)
    assert prediction.lower == pytest.approx(1 - _sig(logit + z * std))
    assert prediction.upper == pytest.approx(1 - _sig(logit - z * std))
    assert facts.counterfactual == local._counterfactual(
        features,
        -0.25,
        -1,
        z,
        after=lambda margin, _replaced: Outcome("yes", _probability_after(margin, _replaced).output),
    )


def test_facts_of_a_prediction_on_the_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    # a logit of exactly 0 belongs to the negative class
    _patch_features(monkeypatch, [_hand("a", [1.0, -1.0]), _hand("b", [-1.0, 1.0])])
    facts = local.compute_local_facts(_hand_model(0.0), {}, local.LocalOptions(confidence_level=0))
    assert (facts.prediction.decision, facts.prediction.output) == ("no", 0.5)
    # the negative class is the prediction, so only b can still move the logit up, and a cannot
    assert facts.counterfactual is not None
    assert [change.feature for change in facts.counterfactual.changes] == ["b"]
    assert facts.counterfactual.new_decision == "yes"


def test_facts_use_the_confidence_level_as_a_two_sided_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_features(monkeypatch, [_hand("a", [1.0, -1.0], stds=[1.0, 1.0])])
    prediction = local.compute_local_facts(_hand_model(), {}, local.LocalOptions(confidence_level=0.5)).prediction
    z = 0.6744897501960817  # the 75th percentile of the normal distribution
    assert prediction.lower == pytest.approx(_sig(1.25 - z))
    assert prediction.upper == pytest.approx(_sig(1.25 + z))


@pytest.mark.parametrize("confidence_level", [-0.5, 1.5])
def test_facts_reject_a_confidence_level_out_of_range(confidence_level: float) -> None:
    with pytest.raises(OptionRangeError, match="confidence_level"):
        local.compute_local_facts(_hand_model(), {}, local.LocalOptions(confidence_level=confidence_level))


# A feature for the properties below: the scores, the standard deviations and the plausible bins of its graph, and the
# bin of the instance. The strategies draw these parts and not the _Feature, because beartype samples with `random`
# when it checks the arguments of a class, and Hypothesis does not allow `random` inside a strategy.
type FeatureParts = tuple[list[float], list[float], list[bool], int]


@st.composite
def feature_parts(draw: st.DrawFn) -> FeatureParts:
    n_bins = draw(st.integers(min_value=2, max_value=3))
    scores = draw(st.lists(st.floats(min_value=-3, max_value=3), min_size=n_bins, max_size=n_bins))
    stds = draw(st.lists(st.floats(min_value=0, max_value=1), min_size=n_bins, max_size=n_bins))
    bin_index = draw(st.integers(min_value=0, max_value=n_bins - 1))
    plausible = draw(st.lists(st.booleans(), min_size=n_bins, max_size=n_bins))
    plausible[bin_index] = True  # _features always offers the bin of the instance
    return scores, stds, plausible, bin_index


def _build(parts: Sequence[FeatureParts]) -> list[local._Feature]:
    features = []
    for index, (scores, stds, plausible, bin_index) in enumerate(parts):
        name = f"f{index}"
        graph = EBMGraph(name, "nominal", [str(bin_index) for bin_index, _ in enumerate(scores)], scores, stds)
        features.append(
            local._Feature(name, graph, np.ones(len(scores)), bin_index, str(bin_index), np.array(plausible)),
        )
    return features


def _sign(features: Sequence[local._Feature], intercept: float) -> int:
    # a logit of 0 is the negative class, as in compute_local_facts
    return 1 if intercept + sum(f.current for f in features) > 0 else -1


# the properties recompute a sum in another order than the code, so they allow for the float error of that sum
TOLERANCE = 1e-9

FEATURES: st.SearchStrategy[list[FeatureParts]] = st.lists(feature_parts(), min_size=1, max_size=4)
INTERCEPT: st.SearchStrategy[float] = st.floats(min_value=-2, max_value=2)
Z: st.SearchStrategy[float] = st.floats(min_value=0, max_value=2.5)


@given(parts=FEATURES, intercept=INTERCEPT, z=Z)
def test_the_changes_of_a_counterfactual_flip_the_prediction(
    parts: list[FeatureParts],
    intercept: float,
    z: float,
) -> None:
    features = _build(parts)
    sign = _sign(features, intercept)
    counterfactual = local._counterfactual(
        features,
        intercept,
        sign,
        z,
        after=_probability_after,
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


@given(parts=FEATURES, intercept=INTERCEPT, z=Z)
def test_the_features_of_a_sufficient_reason_fix_the_outcome_for_every_completion(
    parts: list[FeatureParts],
    intercept: float,
    z: float,
) -> None:
    features = _build(parts)
    sign = _sign(features, intercept)
    reason = local._sufficient_reason(features, intercept, sign, z)
    assume(reason is not None)
    assert reason is not None
    # each feature of the reason keeps the bin of the instance; each other one takes any plausible bin
    choices = [[f.bin_index] if f.name in reason.features else np.flatnonzero(f.plausible).tolist() for f in features]
    for completion in itertools.product(*choices):
        margin = sign * intercept + sum(
            sign * float(f.scores[b]) - z * float(f.stds[b]) for f, b in zip(features, completion, strict=True)
        )
        assert margin > -TOLERANCE


def test_a_decimal_value_falls_in_the_bin_of_its_float(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    as_float = local.compute_local_facts(ebm, {"a": 0.25, "b": -0.5, "c": "x"})
    as_decimal = local.compute_local_facts(ebm, {"a": Decimal("0.25"), "b": -0.5, "c": "x"})
    assert as_decimal.prediction == as_float.prediction


@pytest.fixture(scope="module")
def categorical_regressor(
    fit_ebm: Callable[..., ExplainableBoostingClassifier | ExplainableBoostingRegressor],
) -> FittedRegressor:
    rng = np.random.default_rng(2)
    data = frame_of_columns({name: rng.choice(CATEGORIES, 900) for name in NAMES})
    target = data["f0"].map({"p": 3.0, "q": 0.0, "r": -3.0}) + data["f1"].map({"p": 1.0, "q": 0.0, "r": -1.0})
    target += data["f2"].map({"p": 0.4, "q": 0.0, "r": -0.4}) + rng.normal(size=900) * 0.2
    ebm = fit_ebm(data, target, interactions=0, random_state=0, regression=True)
    assert isinstance(ebm, ExplainableBoostingRegressor)
    return ebm, data


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
    facts = local.compute_local_facts(
        ebm,
        dict(zip(NAMES, row, strict=True)),
        local.LocalOptions(
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
        assert facts.sufficient_reason.minimum


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
    facts = local.compute_local_facts(
        ebm,
        instance,
        local.LocalOptions(confidence_level=0, min_support=0, decision=interval, search=search),
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
    if reason.minimum:
        assert len(chosen) == smallest
    if isinstance(search, MipCover) and search.stop is Stop.OPTIMAL:
        assert reason.minimum
        assert len(chosen) == smallest


def test_regressor_prediction_is_the_predicted_value(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, data = categorical_regressor
    facts = local.compute_local_facts(ebm, instance_of(data, 3), local.LocalOptions(decision=Threshold(1.0)))
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


def test_regressor_counterfactual_gives_the_new_value(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, _ = categorical_regressor
    row = {"f0": "p", "f1": "p", "f2": "p"}
    facts = local.compute_local_facts(
        ebm,
        row,
        local.LocalOptions(confidence_level=0, min_support=0, decision=Threshold(0.5)),
    )
    assert facts.counterfactual is not None
    assert facts.counterfactual.new_decision == "at most 0.5"
    changed = {**row, **{c.feature: c.to_value for c in facts.counterfactual.changes}}
    new_value = float(ebm.predict(frame_of_rows([changed]))[0])
    assert facts.counterfactual.new_output == pytest.approx(new_value)
    assert new_value <= 0.5


def test_a_classifier_rejects_a_decision(categorical_ebm: FittedClassifier) -> None:
    ebm, _ = categorical_ebm
    with pytest.raises(OptionRangeError, match="Only a regressor accepts a decision"):
        local.compute_local_facts(ebm, {"f0": "p", "f1": "p", "f2": "p"}, local.LocalOptions(decision=Threshold(0.0)))


def test_no_reason_with_several_margins_when_all_features_are_not_enough() -> None:
    features = [_feature([0.5, -2.0], 0, std=1.0), _feature([0.5, -2.0], 0, std=1.0)]
    bounds = [local._Bound(0.0, 1), local._Bound(0.0, -1)]
    assert local._reason(features, bounds, 1.96, local.LocalOptions()) is None


def test_the_reveal_order_of_several_margins_ranks_by_the_smallest_gain() -> None:
    assert local._by_smallest_gain([[1.0, 4.0, 2.0], [3.0, 4.0, 0.5]]) == [1, 0, 2]


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
) -> tuple[FittedModel, local.LocalFacts]:
    model, limits = case
    ebm = build_ebm(templates, model)
    # the completions are not under test here, so the sample is tiny
    options = local.LocalOptions(
        confidence_level=confidence_level,
        min_support=min_support,
        decision=_decision(limits),
        n_samples=8,
        max_steps=0,
    )
    return ebm, local.compute_local_facts(ebm, model.instance, options)


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
        assert counterfactual.minimum is True
        assert counterfactual.lower_bound == len(counterfactual.changes)

    reason = facts.sufficient_reason
    assert (None if reason is None else len(reason.features)) == problem.smallest_sufficient()
    if reason is not None:
        guaranteed = problem.worst_case_margin([names.index(name) for name in reason.features])
        assert guaranteed > 0
        assert reason.guaranteed_margin == pytest.approx(guaranteed)
        assert reason.minimum is True


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
        assert low.sufficient_reason.minimum is True
        assert high.sufficient_reason.minimum is True
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
    options = local.LocalOptions(confidence_level=0, min_support=0)
    facts = local.compute_local_facts(build_ebm(small_templates, model), model.instance, options)
    assert facts.prediction.margin == pytest.approx(1.0)
    assert facts.counterfactual is None
    # the worst completion has the margin zero, and a guarantee needs a margin above zero
    assert facts.sufficient_reason is not None
    assert facts.sufficient_reason.features == [feature_name(0)]


def test_a_prediction_on_a_logit_of_zero_has_no_guarantee(small_templates: Templates) -> None:
    model = _tie_model(0.0)
    ebm = build_ebm(small_templates, model)
    facts = local.compute_local_facts(ebm, model.instance, local.LocalOptions(confidence_level=0, min_support=0))
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
