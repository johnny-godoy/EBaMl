"""The graph modes of the benchmark cross the real BAML runtime: what the requests of each mode carry.

Each test answers one question about the requests that reach the local server, so each has one reason to fail.
"""

import base64
from typing import cast

import orjson
import pytest
from benchmark.run import DATA_DIR, ModeAsker, question_graph, question_view

import baml_sdk
from ebaml import GraphImageOptions, GraphMode, ModelOptions, graph_png
from ebaml.benchmark import Task, load_questions, run_task
from tests.conftest import FakeLLM, RecordedRequest, Reply

# one question in the format of EBMGraph.to_text, with bounds, and one in the "Graph: {...}" format, with no bounds
JUMPS: str = load_questions(Task.JUMPS, DATA_DIR)[0].prompt
FUNCTION: str = load_questions(Task.FUNCTION_RECOGNITION, DATA_DIR)[0].prompt
VALUE_LINES = ("Means:", "Lower Bounds", "Graph: {")


def _spec(model: ModelOptions) -> baml_sdk.ModelSpec:
    return baml_sdk.ModelSpec(
        provider=model.provider,
        model=model.model,
        api_key_env=model.api_key_env,
        base_url=model.base_url,
    )


def _ask(model: ModelOptions, mode: GraphMode, question: str) -> None:
    ModeAsker(mode, _spec(model))(question=question, answer_format="a single number")


def _image_urls(request: RecordedRequest) -> list[str]:
    urls: list[str] = []
    for message in request.messages:
        content = message["content"]
        if message["role"] == "user" and isinstance(content, list):
            parts = cast("list[dict[str, dict[str, str]]]", content)
            urls.extend(part["image_url"]["url"] for part in parts if part.get("type") == "image_url")
    return urls


@pytest.mark.parametrize("question", [JUMPS, FUNCTION])
def test_each_turn_in_vision_mode_carries_one_image_and_no_value_of_the_graph(
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    question: str,
) -> None:
    _ask(llm_model, GraphMode.VISION, question)
    assert [len(_image_urls(request)) for request in fake_llm.requests] == [1, 1]
    assert not any(line in request.prompt for request in fake_llm.requests for line in VALUE_LINES)


def test_the_image_is_the_png_of_the_graph_of_the_question_with_its_bounds(
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _ask(llm_model, GraphMode.VISION, JUMPS)
    expected = graph_png(question_graph(JUMPS), GraphImageOptions(confidence_level=0.95))
    assert _image_urls(fake_llm.requests[0]) == ["data:image/png;base64," + base64.b64encode(expected).decode()]


def test_the_image_of_a_question_with_no_bounds_has_no_band(llm_model: ModelOptions, fake_llm: FakeLLM) -> None:
    _ask(llm_model, GraphMode.VISION, FUNCTION)
    expected = graph_png(question_graph(FUNCTION), GraphImageOptions(confidence_level=None))
    assert _image_urls(fake_llm.requests[0]) == ["data:image/png;base64," + base64.b64encode(expected).decode()]


def test_each_turn_in_mixed_mode_carries_the_image_and_the_whole_question(
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _ask(llm_model, GraphMode.MIXED, JUMPS)
    assert [len(_image_urls(request)) for request in fake_llm.requests] == [1, 1]
    assert all(JUMPS.strip() in request.prompt for request in fake_llm.requests)


def test_text_mode_sends_the_question_as_it_is_and_no_image(llm_model: ModelOptions, fake_llm: FakeLLM) -> None:
    _ask(llm_model, GraphMode.TEXT, FUNCTION)
    assert [_image_urls(request) for request in fake_llm.requests] == [[], []]
    assert FUNCTION.strip() in fake_llm.requests[0].prompt


def test_the_vision_question_keeps_the_feature_name_and_the_task() -> None:
    view = question_view(FUNCTION, GraphMode.VISION)
    assert "Feature Name: x\nThe values of the graph are not in this text." in view.question
    assert view.question.endswith(FUNCTION[FUNCTION.index("The graph approximately depicts") :])


@pytest.mark.parametrize("task", list(Task))
def test_every_question_of_the_data_has_a_graph_that_the_image_shows(task: Task) -> None:
    questions = load_questions(task, DATA_DIR)
    for question in questions:
        graph = question_graph(question.prompt)
        assert len(graph.x_vals) == len(graph.scores) > 1
    # the drawing takes a moment, so a few questions of each task check that vision mode removes every value
    for question in questions[:3]:
        assert not any(line in question_view(question.prompt, GraphMode.VISION).question for line in VALUE_LINES)


def test_the_graph_of_the_graph_format_has_the_values_of_the_question() -> None:
    line = next(line for line in FUNCTION.splitlines() if line.startswith("Graph: "))
    values: dict[str, float] = orjson.loads(line.removeprefix("Graph: "))
    graph = question_graph(FUNCTION)
    assert list(graph.scores) == list(values.values())
    assert graph.x_vals[0] == tuple(float(end) for end in next(iter(values))[1:-1].split(","))


def test_an_answer_that_runs_out_of_tokens_counts_as_wrong_and_the_run_goes_on(
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    # a reasoning model can use up the token limit; the run stopped at the first such answer, and lost every result
    fake_llm.handler = lambda _request: Reply(text="", finish_reason="length")
    questions = load_questions(Task.MONOTONICITY, DATA_DIR)[:2]
    result = run_task(Task.MONOTONICITY, questions, ModeAsker(GraphMode.TEXT, _spec(llm_model)))
    assert [outcome.correct for outcome in result.outcomes] == [False, False]
    assert [outcome.response for outcome in result.outcomes] == ["", ""]
