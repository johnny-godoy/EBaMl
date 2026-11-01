"""Tests for the text counts, the token weights and their non-negative fit."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ebaml.core.exceptions import TokenFitError
from ebaml.graphs.token_weights import (
    ModelKey,
    TextFeatures,
    TokenSample,
    TokenWeights,
    WeightsEntry,
    WeightsTable,
    fit_weights,
    relative_spread,
)


def test_text_features_count_words_bytes_and_digits_apart() -> None:
    # "é" is two bytes, "²" is a digit of two bytes, and the three words are separated by spaces and a newline
    assert TextFeatures.of("ab 12\né²") == TextFeatures(words=3, bytes=10, digits=3)


def test_text_features_of_an_empty_text_are_zero() -> None:
    assert TextFeatures.of("") == TextFeatures(words=0, bytes=0, digits=0)


def test_text_features_vector_keeps_the_order_words_bytes_digits() -> None:
    assert TextFeatures(words=1, bytes=2, digits=3).as_vector() == (1.0, 2.0, 3.0)


def test_weights_vector_keeps_the_order_words_bytes_digits() -> None:
    assert TokenWeights(words=1.0, bytes=2.0, digits=3.0).as_vector() == (1.0, 2.0, 3.0)


def test_weights_estimate_gives_each_count_its_own_weight() -> None:
    weights = TokenWeights(words=2.0, bytes=0.5, digits=3.0)
    assert weights.estimate(TextFeatures(words=10, bytes=40, digits=5)) == pytest.approx(55.0)


@pytest.mark.parametrize("invalid_value", [-0.1, math.inf, math.nan])
@pytest.mark.parametrize("field", ["words", "bytes", "digits"])
def test_weights_reject_a_negative_or_non_finite_value(field: str, invalid_value: float) -> None:
    values = {"words": 1.0, "bytes": 1.0, "digits": 1.0, field: invalid_value}
    with pytest.raises(ValueError, match=field):
        TokenWeights(**values)


def test_weights_accept_zero() -> None:
    assert TokenWeights(words=0.0, bytes=0.0, digits=0.0).as_vector() == (0.0, 0.0, 0.0)


def test_sample_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError, match="tokens"):
        TokenSample(TextFeatures(words=1, bytes=1, digits=0), 0)


def _sample(words: int, bytes_: int, digits: int, tokens: float) -> TokenSample:
    return TokenSample(TextFeatures(words=words, bytes=bytes_, digits=digits), round(tokens))


def test_fit_recovers_the_exact_weights_of_clean_data() -> None:
    rows = [(10, 70, 40), (30, 200, 90), (5, 33, 12), (50, 340, 200), (20, 130, 55)]
    samples = [_sample(words, bytes_, digits, 4 * words + 1 * bytes_ + 2 * digits) for words, bytes_, digits in rows]
    fitted = fit_weights(samples)
    assert fitted.as_vector() == pytest.approx((4.0, 1.0, 2.0), abs=1e-6)


def test_fit_sets_a_weight_to_zero_where_least_squares_would_make_it_negative() -> None:
    # the true bytes weight is -0.2, which a negative weight would fit exactly. The fit must keep it at zero instead.
    rows = [(10, 70, 40), (30, 200, 90), (5, 33, 12), (50, 340, 200), (20, 130, 55), (40, 260, 150)]
    samples = [
        _sample(words, bytes_, digits, 5 * words - 0.2 * bytes_ + 0.5 * digits) for words, bytes_, digits in rows
    ]
    fitted = fit_weights(samples)
    assert fitted.bytes == pytest.approx(0.0, abs=1e-12)
    assert fitted.words > 0
    assert fitted.digits > 0


def test_fit_with_one_useful_feature_uses_only_that_feature() -> None:
    samples = [_sample(words, 0, 0, 3 * words) for words in (1, 5, 20, 80)]
    assert fit_weights(samples).words == pytest.approx(3.0, abs=1e-6)


def test_fit_of_a_single_sample_matches_it() -> None:
    sample = _sample(10, 0, 0, 40)
    assert fit_weights([sample]).estimate(sample.features) == pytest.approx(40.0)


def test_fit_rejects_no_samples() -> None:
    with pytest.raises(TokenFitError, match="at least one sample"):
        fit_weights([])


@given(
    rows=st.lists(
        st.tuples(st.integers(1, 500), st.integers(1, 5000), st.integers(0, 3000), st.integers(1, 20000)),
        min_size=1,
        max_size=12,
    ),
)
def test_fit_never_gives_a_negative_weight(rows: list[tuple[int, int, int, int]]) -> None:
    fitted = fit_weights([
        TokenSample(TextFeatures(words, bytes_, digits), tokens) for words, bytes_, digits, tokens in rows
    ])
    assert min(fitted.as_vector()) >= 0.0


def test_relative_spread_is_the_root_mean_square_of_the_relative_error() -> None:
    weights = TokenWeights(words=1.0, bytes=0.0, digits=0.0)
    samples = [_sample(1, 0, 0, 2), _sample(3, 0, 0, 2)]  # errors -0.5 and +0.5
    assert relative_spread(weights, samples) == pytest.approx(0.5)


def test_relative_spread_of_no_samples_is_zero() -> None:
    assert relative_spread(TokenWeights(words=1.0, bytes=0.0, digits=0.0), []) == pytest.approx(0.0)


def _entry(words: float = 1.0, *, update_allowed: bool = True) -> WeightsEntry:
    return WeightsEntry(
        TokenWeights(words=words, bytes=0.0, digits=0.5),
        relative_spread=0.1,
        samples=7,
        update_allowed=update_allowed,
    )


def test_entry_for_a_listed_model_is_its_entry_and_for_any_other_the_default() -> None:
    table = WeightsTable(default=_entry(1.0), models={ModelKey("p", "m"): _entry(2.0)})
    assert table.entry_for(ModelKey("p", "m")) == _entry(2.0)
    assert table.entry_for(ModelKey("p", "other")) == _entry(1.0)
    assert table.entry_for(None) == _entry(1.0)


def test_with_entry_adds_or_replaces_an_entry_and_keeps_the_table_unchanged() -> None:
    table = WeightsTable(default=_entry(1.0), models={ModelKey("p", "m"): _entry(2.0)})
    added = table.with_entry(ModelKey("p", "n"), _entry(3.0))
    replaced = table.with_entry(ModelKey("p", "m"), _entry(4.0))
    assert added.entry_for(ModelKey("p", "n")) == _entry(3.0)
    assert added.entry_for(ModelKey("p", "m")) == _entry(2.0)
    assert replaced.entry_for(ModelKey("p", "m")) == _entry(4.0)
    assert table.entry_for(ModelKey("p", "n")) == _entry(1.0)


def test_a_table_reads_back_what_it_wrote() -> None:
    table = WeightsTable(
        default=_entry(1.0),
        models={ModelKey("b", "x"): _entry(2.0, update_allowed=False), ModelKey("a", "y"): _entry(3.0)},
    )
    assert WeightsTable.from_bytes(table.to_bytes()) == table


def test_a_table_writes_its_models_in_order_so_that_the_text_is_the_same() -> None:
    first = WeightsTable(default=_entry(), models={ModelKey("b", "x"): _entry(), ModelKey("a", "y"): _entry()})
    second = WeightsTable(default=_entry(), models={ModelKey("a", "y"): _entry(), ModelKey("b", "x"): _entry()})
    assert first.to_bytes() == second.to_bytes()
    assert first.to_bytes().index(b'"a"') < first.to_bytes().index(b'"b"')


@pytest.mark.parametrize("data", [b"", b"not json", b"{}", b'{"default": {}, "models": []}', b"[]", b'{"default": 1}'])
def test_a_table_that_cannot_be_read_raises_a_fit_error(data: bytes) -> None:
    with pytest.raises(TokenFitError, match="cannot be read"):
        WeightsTable.from_bytes(data)


def test_an_entry_rejects_a_negative_spread_or_sample_count() -> None:
    weights = TokenWeights(words=1.0, bytes=0.0, digits=0.0)
    with pytest.raises(ValueError, match="relative_spread"):
        WeightsEntry(weights, relative_spread=-0.1, samples=1)
    with pytest.raises(ValueError, match="samples"):
        WeightsEntry(weights, relative_spread=0.1, samples=-1)


def test_an_entry_takes_four_bytes_of_prose_for_each_token_unless_told_otherwise() -> None:
    assert _entry().prose_bytes_per_token == pytest.approx(4.0)


@pytest.mark.parametrize("invalid_density", [0.0, -1.0])
def test_an_entry_rejects_a_density_of_prose_that_is_not_above_zero(invalid_density: float) -> None:
    with pytest.raises(ValueError, match="prose_bytes_per_token"):
        WeightsEntry(
            TokenWeights(words=1.0, bytes=0.0, digits=0.0),
            relative_spread=0.1,
            samples=1,
            prose_bytes_per_token=invalid_density,
        )


def test_a_table_keeps_the_density_of_prose_of_each_entry_when_it_is_written_and_read() -> None:
    entry = WeightsEntry(TokenWeights(words=1.0, bytes=0.0, digits=0.0), 0.1, 1, prose_bytes_per_token=4.38)
    table = WeightsTable(default=_entry(), models={ModelKey("p", "m"): entry})
    assert WeightsTable.from_bytes(table.to_bytes()).entry_for(
        ModelKey("p", "m"),
    ).prose_bytes_per_token == pytest.approx(4.38)
    assert WeightsTable.from_bytes(table.to_bytes()).default.prose_bytes_per_token == pytest.approx(4.0)
