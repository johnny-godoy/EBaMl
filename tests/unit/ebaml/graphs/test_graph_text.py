"""Tests for the textual representation of graphs."""

import pytest

from ebaml.core.exceptions import GraphTextError
from ebaml.graphs.graph_text import parse_str_tuple_to_float_tuple, xy_to_json


def test_xy_to_jsoncontinuous() -> None:
    assert xy_to_json([(0.0, 1.0), (1.0, 2.0)], [3.0, 4.0]) == '{"(0.0, 1.0)": 3.0, "(1.0, 2.0)": 4.0}'


def test_xy_to_jsoncategorical() -> None:
    assert xy_to_json(["a", "b"], [1.0, 2.0]) == '{"a": 1.0, "b": 2.0}'


def test_parse_str_tuple_to_float_tuple() -> None:
    assert parse_str_tuple_to_float_tuple("(1.5, 2)") == (1.5, 2.0)


def test_parse_str_tuple_to_float_tuple_rejects_three_values() -> None:
    # the parser returned a tuple of any length, so "(1, 2, 3)" became an interval with three ends
    with pytest.raises(GraphTextError, match="not an interval of two numbers"):
        parse_str_tuple_to_float_tuple("(1, 2, 3)")


def test_xy_to_jsonmixed_and_empty() -> None:
    assert xy_to_json([(0.0, 1.0), "a"], [1.0, 2.0]) == '{"(0.0, 1.0)": 1.0, "a": 2.0}'
    assert xy_to_json([], []) == "{}"


@pytest.mark.parametrize("x_vals", [["a", "b"], [(0.0, 1.0), (1.0, 2.0)]])
def test_xy_to_jsonlength_mismatch(x_vals: list[str] | list[tuple[float, float]]) -> None:
    with pytest.raises(ValueError, match=r"shorter|longer"):
        xy_to_json(x_vals, [1.0])


@pytest.mark.parametrize("text", ["(a, 2)", "(1)", "()"])
def test_parse_str_tuple_to_float_tuple_rejects_a_text_that_is_not_two_numbers(text: str) -> None:
    with pytest.raises(GraphTextError, match="not an interval of two numbers"):
        parse_str_tuple_to_float_tuple(text)
