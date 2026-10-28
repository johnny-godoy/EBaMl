"""Tests for the textual representation of graphs."""

import orjson
import pytest

from ebaml.core.exceptions import GraphTextError
from ebaml.graphs.graph_text import parse_graph_text, parse_str_tuple_to_float_tuple, xy_to_json


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


def test_xy_to_json_escapes_quotes_and_backslashes_in_keys() -> None:
    # regression: a key with a quote or a backslash made the line invalid JSON
    keys = ['Master "B"', "a\\b"]
    text = xy_to_json(keys, ["0.1", "0.2"])
    assert list(orjson.loads(text)) == keys


def series(means: str, label: str | None = None) -> str:
    head = f"Class: {label}\n" if label is not None else ""
    return (
        f"{head}Means: {means}\n"
        f"Lower Bounds (95%-Confidence Interval): {means}\n"
        f"Upper Bounds (95%-Confidence Interval): {means}\n"
    )


HEADER = "Feature Name: age\nFeature Type: continuous\n"


def test_parse_graph_text_of_one_output_has_no_classes() -> None:
    parsed = parse_graph_text(HEADER + series('{"(0, 1)": 0.5}'))
    assert (parsed.scores, parsed.stds, parsed.classes) == ([0.5], [0.0], [])


def test_parse_graph_text_reads_one_column_per_class() -> None:
    text = HEADER + series('{"(0, 1)": 0.5, "(1, 2)": 1.0}', " a ") + series('{"(0, 1)": -0.5, "(1, 2)": 2.0}', "b")
    parsed = parse_graph_text(text)
    # a label keeps its spaces, as to_text writes it
    assert parsed.classes == [" a ", "b"]
    assert parsed.x_vals == [(0.0, 1.0), (1.0, 2.0)]
    assert parsed.scores == [[0.5, -0.5], [1.0, 2.0]]
    assert parsed.stds == [[0.0, 0.0], [0.0, 0.0]]


def test_parse_graph_text_reads_each_class_from_its_own_lines() -> None:
    # the second class lacks its bounds, so the bounds of the first class must not stand in for them
    text = HEADER + series('{"(0, 1)": 0.5}', "a") + 'Class: b\nMeans: {"(0, 1)": 1.0}\n'
    with pytest.raises(GraphTextError, match="no 'Lower Bounds' line"):
        parse_graph_text(text)


def test_parse_graph_text_rejects_classes_with_other_intervals() -> None:
    text = HEADER + series('{"(0, 1)": 0.5}', "a") + series('{"(0, 2)": 0.5}', "b")
    with pytest.raises(GraphTextError, match="classes of the graph text do not have the same intervals"):
        parse_graph_text(text)


def test_parse_graph_text_rejects_a_series_before_the_first_class() -> None:
    # a series of one output before the class series was dropped without a word
    text = HEADER + series('{"(0, 1)": 0.5}') + series('{"(0, 1)": 0.5}', "a")
    with pytest.raises(GraphTextError, match="a series before its first 'Class' line"):
        parse_graph_text(text)
