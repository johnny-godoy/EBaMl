"""Tests for the deterministic local explanations."""

import itertools
import json
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import local
from ebaml.custom_types import Audience
from ebaml.graphs import EBMGraph

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


@pytest.fixture(scope="module")
def continuous_ebm(
    fit_ebm: Callable[..., ExplainableBoostingClassifier],
) -> tuple[ExplainableBoostingClassifier, pd.DataFrame]:
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600), "c": rng.choice(["x", "y"], 600)})
    noise = rng.normal(size=600)
    target = ((2 * data["a"] + 0.3 * data["b"] + (data["c"] == "x") + noise) > 0).astype(int)
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


def test_texts_hide_numbers_from_layperson(continuous_ebm: tuple) -> None:
    ebm, data = continuous_ebm
    facts = local.compute_local_facts(ebm, data.iloc[0].to_dict())
    layperson = local.facts_to_texts(facts)
    expert = local.facts_to_texts(facts, Audience.DATA_SCIENTIST)
    assert set(layperson) == {"prediction", "counterfactual", "sufficient_reason", "completions"}
    for text in layperson.values():
        for key in ("score", "std", "probability", "logit", "confidence"):
            assert key not in text
    assert "score" in expert["prediction"]
    assert "flip_holds_at_confidence" in expert["counterfactual"]
    assert "confidence_interval" in expert["prediction"]
    assert json.loads(expert["completions"])["steps"][0]["share_with_same_prediction"] <= 1


@pytest.mark.parametrize(
    ("probability", "word"),
    [
        (0.99, "almost certain"),
        (0.85, "very likely"),
        (0.7, "likely"),
        (0.5, "about as likely as not"),
        (0.3, "unlikely"),
        (0.1, "very unlikely"),
        (0.01, "almost impossible"),
    ],
)
def test_likelihood_words(probability: float, word: str) -> None:
    assert local._likelihood(probability) == word


@pytest.mark.parametrize(("score", "word"), [(-0.6, "strongly"), (0.3, "moderately"), (0.1, "slightly")])
def test_strength_words(score: float, word: str) -> None:
    assert local._strength(score) == word


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


def test_texts_without_counterfactual_or_sufficient_reason() -> None:
    prediction = local.Prediction(label=0, probability=0.7, lower=0.6, upper=0.8, logit=-0.8, logit_std=0.1)
    facts = local.LocalFacts(prediction, [], None, None, [], 0.95)
    texts = local.facts_to_texts(facts, Audience.DATA_SCIENTIST)
    assert json.loads(texts["counterfactual"]) == {"flip_possible": False}
    assert json.loads(texts["sufficient_reason"]) == {"exists": False}


def test_rejects_multiclass(fit_ebm: Callable) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    ebm = fit_ebm(frame, rng.integers(0, 3, 300), interactions=0, random_state=0)
    with pytest.raises(ValueError, match="binary"):
        local.compute_local_facts(ebm, frame.iloc[0].to_dict())
