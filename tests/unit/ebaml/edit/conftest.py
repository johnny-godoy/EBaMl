"""Fixtures, a consistency check, and the reference edits for the tests of ``ebaml.edit``.

The reference edits write each GAM Changer tool again as plain loops over lists of points. They follow the table of the
tools in ``docs/plans/model-editing.md``, and they share no code with ``ebaml.edit``. A test compares the two.
"""

import copy
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pytest
import skops.io

from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.graphs.ebm_graph import EBMGraph
from tests.unit.ebaml.conftest import PandasFrame, frame_of_columns

ROW_COUNT = 600
FEATURES = ("age", "score", "group", "grade")
# the feature types of the fixture: two continuous, one nominal, and one ordinal with its order
FEATURE_TYPES = ["continuous", "continuous", "nominal", ["low", "mid", "high"]]


@pytest.fixture(scope="module")
def edit_data() -> PandasFrame:
    """Give training data with a missing value in ``age``, a nominal ``group`` and an ordinal ``grade``.

    Returns
    -------
    PandasFrame
        The data.

    """
    random_generator = np.random.default_rng(2)
    age = random_generator.normal(size=ROW_COUNT)
    age[:21] = np.nan
    return frame_of_columns({
        "age": age,
        "score": random_generator.normal(size=ROW_COUNT),
        "group": random_generator.choice(["x", "y", "z"], ROW_COUNT),
        "grade": random_generator.choice(["low", "mid", "high"], ROW_COUNT),
    })


@pytest.fixture(scope="module")
def edit_ebm(edit_data: PandasFrame) -> ExplainableBoostingClassifier:
    """Give a fitted binary classifier with pair terms, a missing bin and an ordinal feature.

    Returns
    -------
    ExplainableBoostingClassifier
        The fitted model.

    """
    random_generator = np.random.default_rng(3)
    signal = (
        2 * np.nan_to_num(edit_data["age"].to_numpy())
        + (edit_data["group"] == "x").to_numpy()
        + (edit_data["grade"] == "high").to_numpy()
    )
    target = ((signal + random_generator.normal(size=ROW_COUNT)) > 0).astype(int)
    model = ExplainableBoostingClassifier(interactions=2, random_state=0, outer_bags=4, feature_types=FEATURE_TYPES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # interpret warns that the graphs do not show the missing values
        return model.fit(edit_data, target)


@pytest.fixture(scope="module")
def edit_regressor(edit_data: PandasFrame) -> ExplainableBoostingRegressor:
    """Give a fitted regressor without pair terms.

    Returns
    -------
    ExplainableBoostingRegressor
        The fitted model.

    """
    random_generator = np.random.default_rng(4)
    target = 2 * np.nan_to_num(edit_data["age"].to_numpy()) + random_generator.normal(size=ROW_COUNT)
    model = ExplainableBoostingRegressor(interactions=0, random_state=0, outer_bags=4, feature_types=FEATURE_TYPES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(edit_data, target)


@pytest.fixture(scope="module")
def interpret_monotonized(edit_ebm: ExplainableBoostingClassifier) -> ExplainableBoostingClassifier:
    """Give a model that ``interpret`` monotonized, so its uncertainty attributes are ``None``.

    Returns
    -------
    ExplainableBoostingClassifier
        A copy of the fitted model.

    """
    model = copy.deepcopy(edit_ebm)
    model.monotonize(1, increasing=True)
    return model


def assert_consistent(ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor, data: PandasFrame) -> None:
    """Check that an edited model works wherever a fitted model works, and that its uncertainty agrees.

    The calls are ``predict``, ``predict_with_uncertainty``, ``explain_global``, ``term_importances``,
    ``EBMGraph.from_ebm`` and a save and load with ``skops``. The prediction with uncertainty must equal the
    prediction, and no uncertainty attribute may hold ``None``.
    """
    assert all(bags is not None for bags in ebm.bagged_scores_)
    assert all(deviations is not None for deviations in ebm.standard_deviations_)
    assert ebm.bagged_intercept_ is not None
    ebm.predict(data)
    score = ebm.predict_with_uncertainty(data)[:, 0]
    assert np.allclose(
        score,
        np.ravel(ebm.decision_function(data)) if hasattr(ebm, "decision_function") else ebm.predict(data),
    )
    ebm.explain_global()
    ebm.term_importances()
    for feature_index, feature_type in enumerate(ebm.feature_types_in_):
        if feature_type in {"continuous", "nominal"} and (feature_index,) in [tuple(t) for t in ebm.term_features_]:
            EBMGraph.from_ebm(ebm, feature_index)
    saved = skops.io.dumps(ebm)
    loaded = skops.io.loads(saved, trusted=skops.io.get_untrusted_types(data=saved))
    assert np.allclose(loaded.predict(data), ebm.predict(data))


# ---- the reference edits: plain loops, written from the table of tools in the plan -----------------------------------


def reference_isotonic(values: Sequence[float], weights: Sequence[float], *, increasing: bool = True) -> list[float]:
    """Fit a monotone sequence with the pool-adjacent-violators loop, with weights.

    Returns
    -------
    list[float]
        The fitted value of each point.

    """
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # each block holds [weighted sum, weight, size]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([sign * value * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            last = blocks.pop()
            blocks[-1] = [blocks[-1][0] + last[0], blocks[-1][1] + last[1], blocks[-1][2] + last[2]]
    fitted: list[float] = []
    for weighted_sum, weight, size in blocks:
        fitted.extend([sign * weighted_sum / weight] * int(size))
    return fitted


def reference_interpolate(places: Sequence[float], scores: Sequence[float]) -> list[float]:
    """Draw the straight line between the first and the last point.

    Returns
    -------
    list[float]
        The value of the line at each place.

    """
    slope = (scores[-1] - scores[0]) / (places[-1] - places[0])
    return [scores[0] + slope * (place - places[0]) for place in places]


def reference_regression(places: Sequence[float], scores: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Fit the weighted least-squares line.

    Returns
    -------
    list[float]
        The value of the line at each place.

    """
    total = sum(weights)
    mean_place = sum(weight * place for weight, place in zip(weights, places, strict=True)) / total
    mean_score = sum(weight * score for weight, score in zip(weights, scores, strict=True)) / total
    covariance = sum(
        weight * (place - mean_place) * (score - mean_score)
        for weight, place, score in zip(weights, places, scores, strict=True)
    )
    variance = sum(weight * (place - mean_place) ** 2 for weight, place in zip(weights, places, strict=True))
    slope = covariance / variance
    return [mean_score + slope * (place - mean_place) for place in places]


def reference_term_scores(ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor, name: str) -> list[float]:
    """Read the scores of the term of a feature as a list.

    Returns
    -------
    list[float]
        The score of every row of the term.

    """
    return [float(score) for score in ebm.term_scores_[list(ebm.term_names_).index(name)]]


def fixed_clock() -> Callable[[], datetime]:
    """Give a clock that returns a new time on each call, one minute after the last.

    Returns
    -------
    Callable[[], datetime]
        The clock.

    """
    state = {"time": datetime(2026, 1, 1, tzinfo=UTC)}

    def clock() -> datetime:
        state["time"] += timedelta(minutes=1)
        return state["time"]

    return clock


def strip_uncertainty(
    ebm: ExplainableBoostingClassifier | ExplainableBoostingRegressor,
    *,
    terms: Sequence[int],
    intercept: bool,
) -> None:
    """Set the uncertainty attributes of some terms to ``None``, as ``monotonize`` of ``interpret`` leaves them.

    The attributes are typed as arrays, so the function writes them through ``vars``, as ``fake_model`` does.
    """
    attributes = vars(ebm)
    for term in terms:
        attributes["bagged_scores_"][term] = None
        attributes["standard_deviations_"][term] = None
    if intercept:
        attributes["bagged_intercept_"] = None


def move_cuts(ebm: ExplainableBoostingClassifier, feature_index: int, offset: float) -> None:
    """Move every cut of a continuous feature by ``offset``, so that the bins differ."""
    vars(ebm)["bins_"][feature_index][0] = np.asarray(ebm.bins_[feature_index][0]) + offset


def rename_categories(ebm: ExplainableBoostingClassifier, feature_index: int) -> None:
    """Add ``!`` to each category of a nominal feature."""
    names = cast("dict[str, int]", ebm.bins_[feature_index][0])
    vars(ebm)["bins_"][feature_index][0] = {f"{name}!": position for name, position in names.items()}
