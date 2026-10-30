"""Tests for the token budget check that needs no BAML SDK."""

from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

import baml_sdk
from ebaml.graphs.ebm_graph import BudgetCheck, EBMGraph
from ebaml.graphs.token_budget import BudgetStatus, check_token_budget, estimate_tokens


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("", 0), ("a", 1), ("ab", 1), ("abc", 1), ("abcd", 2), ("abcdef", 2), ("abcdefg", 3), ("日本語テ", 2)],
)
def test_estimate_tokens_rounds_up_at_three_characters_per_token(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


@pytest.mark.parametrize(
    ("max_tokens", "overbudget_percent", "status"),
    [(100, 0, "ok"), (99, 2, "warn"), (99, 1, "over"), (99, 0, "over"), (67, 50, "warn"), (66, 50, "over")],
)
def test_check_token_budget_changes_status_exactly_past_the_limits(
    max_tokens: int,
    overbudget_percent: int,
    status: Literal["ok", "warn", "over"],
) -> None:
    text = "a" * 300  # 100 tokens
    assert check_token_budget(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent) == BudgetStatus(
        100,
        status,
    )


@given(
    text=st.text(max_size=400),
    max_tokens=st.integers(min_value=0, max_value=200),
    overbudget_percent=st.integers(min_value=0, max_value=200),
)
def test_check_token_budget_agrees_with_the_baml_function(text: str, max_tokens: int, overbudget_percent: int) -> None:
    # the prompts of the LLM functions use the BAML rule, so the two must not drift apart
    expected = baml_sdk.check_token_budget(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent)
    actual = check_token_budget(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent)
    assert (actual.estimated_tokens, actual.status) == (expected.estimated_tokens, expected.status)
    assert estimate_tokens(text) == baml_sdk.estimate_tokens(text=text)


def test_check_token_budget_is_a_budget_check() -> None:
    assert isinstance(check_token_budget, BudgetCheck)


def test_to_text_simplifies_a_large_graph_with_the_check() -> None:
    # without a check, to_text ignores max_tokens; with this one it simplifies the graph until it fits
    bins = 200
    x_vals = [(float(i), float(i + 1)) for i in range(bins)]
    graph = EBMGraph("x", "continuous", x_vals, [i * 0.001 for i in range(bins)], [0.0] * bins)
    long_text = graph.to_text()
    short_text = graph.to_text(max_tokens=1000, overbudget_percent=0, check_budget=check_token_budget)
    assert estimate_tokens(short_text) <= 1000 < estimate_tokens(long_text)
