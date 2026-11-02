"""Tests for the token estimator, its budget check and its learning."""

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pytest

from ebaml.graphs.ebm_graph import BudgetCheck, EBMGraph
from ebaml.graphs.token_budget import (
    DEFAULT_RELATIVE_SPREAD,
    DEFAULT_WEIGHTS,
    BudgetStatus,
    TokenEstimator,
    check_token_budget,
    estimate_tokens,
)
from ebaml.graphs.token_weights import TextFeatures, TokenSample, TokenWeights

# one word is one token, so that a text of 100 words has an estimate of 100
_ONE_PER_WORD = TokenWeights(words=1.0, bytes=0.0, digits=0.0)


def _plain(weights: TokenWeights = _ONE_PER_WORD, *, learning: bool = False) -> TokenEstimator:
    return TokenEstimator(weights=weights, relative_spread=0.0, learning=learning)


def test_estimate_gives_each_count_its_own_weight() -> None:
    estimator = _plain(TokenWeights(words=2.0, bytes=0.5, digits=3.0))
    # 2 words, 2 + 1 + 1 + 1 = 5 bytes, 2 digits: 2 * 2 + 0.5 * 5 + 3 * 2 = 12.5
    assert estimator.estimate("a1 b1") == 13


def test_estimate_of_an_empty_text_is_zero() -> None:
    assert TokenEstimator().estimate("") == 0


def test_estimate_adds_the_margin_above_the_fit_and_rounds_up() -> None:
    estimator = TokenEstimator(weights=_ONE_PER_WORD, relative_spread=0.1, margin=2.0)
    # 10 words are 10 tokens, and the margin of two spreads makes it 12
    assert estimator.estimate("w " * 10) == 12
    # 1 word is 1.2 tokens, which rounds up to 2
    assert estimator.estimate("w") == 2


@pytest.mark.parametrize(
    ("max_tokens", "overbudget_percent", "status"),
    [(100, 0, "ok"), (99, 2, "warn"), (99, 1, "over"), (99, 0, "over"), (67, 50, "warn"), (66, 50, "over")],
)
def test_check_budget_changes_status_exactly_past_the_limits(
    max_tokens: int,
    overbudget_percent: int,
    status: Literal["ok", "warn", "over"],
) -> None:
    text = "w " * 100  # 100 tokens
    estimator = _plain()
    assert estimator(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent) == (
        BudgetStatus(100, status)
    )


def test_an_estimator_is_a_budget_check() -> None:
    estimator = _plain()
    assert isinstance(estimator, BudgetCheck)
    assert estimator(text="w " * 100, max_tokens=80, overbudget_percent=50) == BudgetStatus(100, "warn")


def test_the_module_functions_use_the_default_weights_and_never_learn() -> None:
    text = "0.5 1.25 -0.75 " * 20
    default = TokenEstimator(weights=DEFAULT_WEIGHTS, relative_spread=DEFAULT_RELATIVE_SPREAD)
    assert estimate_tokens(text) == default.estimate(text)
    assert check_token_budget(text=text, max_tokens=10, overbudget_percent=0) == default(
        text=text,
        max_tokens=10,
        overbudget_percent=0,
    )
    assert default.learning is False


def test_the_default_estimate_is_not_far_below_the_tokens_of_a_numeric_graph_text() -> None:
    # the old rule of 3 characters per token gave 20 tokens for this text. Digit-heavy text takes about 2 characters.
    text = '"(0.12, 0.13)": -0.4496, ' * 10
    assert estimate_tokens(text) > len(text) / 2.5 * 0.5


def test_observe_changes_nothing_when_learning_is_off() -> None:
    estimator = _plain()
    assert estimator.observe(TokenSample(TextFeatures(words=100, bytes=500, digits=0), 400)) is False
    assert estimator.current_weights == _ONE_PER_WORD
    assert estimator.samples_seen == 0


def test_observe_starts_at_the_prior_and_moves_the_weights_a_little_after_one_sample() -> None:
    estimator = _plain(learning=True)
    # the sample says 200 tokens for 100 words, and the prior says 1 per word
    assert estimator.observe(TokenSample(TextFeatures(words=100, bytes=0, digits=0), 200)) is True
    moved = estimator.current_weights.words
    assert 1.0 < moved < 1.5
    assert estimator.samples_seen == 1


def test_many_consistent_samples_move_the_weights_to_the_true_slope() -> None:
    # a prior worth one sample leaves little bias after 400 samples. The default prior of 10 samples keeps more of it.
    estimator = TokenEstimator(weights=_ONE_PER_WORD, relative_spread=0.0, learning=True, prior_samples=1.0)
    random_generator = np.random.default_rng(0)
    for _ in range(400):
        words = int(random_generator.integers(20, 800))
        digits = int(random_generator.integers(0, 3000))
        # the bytes do not depend on the other two counts here, so that the true weights are the only fit
        bytes_ = int(random_generator.integers(0, 5000))
        estimator.observe(TokenSample(TextFeatures(words=words, bytes=bytes_, digits=digits), 2 * words + digits))
    weights = estimator.current_weights
    assert weights.as_vector() == pytest.approx((2.0, 0.0, 1.0), abs=0.1)


def test_a_sample_that_asks_for_a_negative_weight_leaves_it_at_zero() -> None:
    estimator = _plain(learning=True)
    for _ in range(20):
        estimator.observe(TokenSample(TextFeatures(words=100, bytes=1000, digits=0), 1))
    assert min(estimator.current_weights.as_vector()) >= 0.0


def test_concurrent_observations_lose_no_sample() -> None:
    estimator = _plain(learning=True)
    sample = TokenSample(TextFeatures(words=100, bytes=700, digits=10), 250)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(estimator.observe, sample) for _ in range(400)]
        assert all(future.result() for future in futures)
    assert estimator.samples_seen == 400
    # the prior keeps the estimate between its start (100) and the sample (250), but 400 samples move it far from 100
    assert 150 < estimator.current_weights.estimate(sample.features) <= 250


def test_restore_gives_the_same_estimator_as_the_one_that_made_the_state() -> None:
    first = _plain(learning=True)
    samples = [
        TokenSample(TextFeatures(words=words, bytes=7 * words, digits=words // 3), 3 * words)
        for words in (50, 120, 300)
    ]
    for sample in samples[:2]:
        first.observe(sample)
    second = _plain(learning=True).restore(first.state())
    assert second.current_weights == first.current_weights
    assert second.samples_seen == 2
    first.observe(samples[2])
    second.observe(samples[2])
    assert second.current_weights == first.current_weights
    assert len(first.state().covariance) == 9


def test_a_negative_spread_is_rejected() -> None:
    with pytest.raises(ValueError, match="relative_spread"):
        TokenEstimator(relative_spread=-0.1)


def test_a_negative_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="margin"):
        TokenEstimator(margin=-0.1)


def test_a_prior_of_no_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="prior_samples"):
        TokenEstimator(prior_samples=0.0)


def test_to_text_simplifies_a_large_graph_with_an_estimator() -> None:
    # without a check, to_text ignores max_tokens; with this one it simplifies the graph until it fits
    bins = 200
    x_vals = [(float(bin_index), float(bin_index + 1)) for bin_index in range(bins)]
    graph = EBMGraph("x", "continuous", x_vals, [bin_index * 0.001 for bin_index in range(bins)], [0.0] * bins)
    long_text = graph.to_text()
    short_text = graph.to_text(max_tokens=1000, overbudget_percent=0, check_budget=TokenEstimator())
    assert estimate_tokens(short_text) <= 1000 < estimate_tokens(long_text)
