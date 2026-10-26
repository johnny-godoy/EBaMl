"""Tests for the benchmark."""

import inspect
from pathlib import Path
from typing import Protocol

import attrs
import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st
from structlog.testing import capture_logs

import baml_sdk
from ebaml import benchmark
from ebaml.benchmark import (
    AskQuestion,
    Outcome,
    Question,
    Reply,
    Task,
    TaskResult,
    Usage,
    format_results,
    is_correct,
    load_questions,
    parse_answer,
    run_task,
)
from tests.unit.ebaml.conftest import call_mismatches, protocol_mismatches, same_type, sdk_fields, sdk_signature

DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"


@attrs.frozen
class FakeUsage:
    """The tokens of a fake reply."""

    input_tokens: int
    output_tokens: int


@attrs.frozen
class FakeReply:
    """A reply that obeys the ``Reply`` protocol, with no LLM client."""

    text: str
    usage: FakeUsage


def counted(text: str, input_tokens: int = 3, output_tokens: int = 5) -> FakeReply:
    return FakeReply(text, FakeUsage(input_tokens, output_tokens))


def write_questions(tmp_path: Path, task: Task, records: list[list[str]]) -> None:
    (tmp_path / f"{task}.json").write_bytes(orjson.dumps(records))


def test_every_task_has_one_spec() -> None:
    assert set(benchmark._SPECS) == set(Task)


@pytest.mark.parametrize("task", list(Task))
def test_every_task_has_an_answer_format(task: Task) -> None:
    assert isinstance(task.answer_format, str)
    assert task.answer_format


@pytest.mark.parametrize("task", list(Task))
def test_every_solution_of_the_data_is_parsed_and_matches_itself(task: Task) -> None:
    questions = load_questions(task, DATA_DIR)
    assert questions
    for question in questions:
        assert parse_answer(task, question.solution) is not None
        assert is_correct(task, question.solution, question.solution)


def test_load_questions(tmp_path: Path) -> None:
    write_questions(tmp_path, Task.MONOTONICITY, [["p1", "Increasing"], ["p2", "Not monotone"]])
    assert load_questions(Task.MONOTONICITY, tmp_path) == [Question("p1", "Increasing"), Question("p2", "Not monotone")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a)", "a"),
        ("(B)", "b"),
        ("**c)**", "c"),
        ("d", "d"),
        ("The answer is e)", "e"),
        ("Answer: b", None),
        ("f)", None),
        ("", None),
    ],
)
def test_parse_choice(text: str, expected: str | None) -> None:
    assert parse_answer(Task.FUNCTION_RECOGNITION, text) == expected
    assert parse_answer(Task.FUNCTION_RECOGNITION_HARD, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Increasing", "increasing"),
        ("monotone decreasing.", "decreasing"),
        ("Not monotone.", "not monotone"),
        ("not monotone: it is not increasing", "not monotone"),
        ("a) monotone increasing", "increasing"),
        ("I cannot tell", None),
    ],
)
def test_parse_monotonicity(text: str, expected: str | None) -> None:
    assert parse_answer(Task.MONOTONICITY, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.0917", (0.0917,)),
        ("-0.004421", (-0.004421,)),
        ("**2.5**", (2.5,)),
        ("-1e-05", (-1e-05,)),
        ("+3", (3.0,)),
        ("17,113.4", (17113.4,)),
        ("1,234,567", (1234567.0,)),
        ("x = 2, y = 3", (2.0,)),
        ("2,3", (2.0,)),
        ("1,2345", (1.0,)),
        # a number with no digit before the decimal point is not read as the digits after it
        (".5", (0.5,)),
        ("-.25", (-0.25,)),
        ("The value is .5.", (0.5,)),
        ("no number", None),
    ],
)
def test_parse_number(text: str, expected: tuple[float, ...] | None) -> None:
    assert parse_answer(Task.JUMPS, text) == expected
    assert parse_answer(Task.READ_VALUE, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(0.5, 1.5)", (0.5, 1.5)),
        ("( -3 ,  4e2 )", (-3.0, 400.0)),
        ("The interval is (13.5, 15.0).", (13.5, 15.0)),
        ("(.5, 1.5)", (0.5, 1.5)),
        ("(1,234, 5,678)", None),
        ("(1.5)", None),
        ("[1, 2]", None),
    ],
)
def test_parse_interval(text: str, expected: tuple[float, ...] | None) -> None:
    assert parse_answer(Task.WIDE_CONFIDENCE, text) == expected


@pytest.mark.parametrize(
    ("task", "response", "solution", "expected"),
    [
        (Task.FUNCTION_RECOGNITION, "(a)", "a)", True),
        (Task.FUNCTION_RECOGNITION, "b)", "a)", False),
        (Task.MONOTONICITY, "Not monotone.", "Not monotone", True),
        (Task.MONOTONICITY, "Increasing", "Decreasing", False),
        (Task.READ_VALUE, "0.50", "0.5", True),
        (Task.READ_VALUE, "0.51", "0.5", False),
        (Task.READ_VALUE, ".5", "0.5", True),
        (Task.JUMPS, "2.5", "2.5", True),
        (Task.WIDE_CONFIDENCE, "(0.0, 0.5)", "(0.0, 0.5)", True),
        (Task.WIDE_CONFIDENCE, "(0.0, 0.6)", "(0.0, 0.5)", False),
        (Task.WIDE_CONFIDENCE, "(0.1, 0.5)", "(0.0, 0.5)", False),
        # no answer, or one that is not in the format of the task
        (Task.MONOTONICITY, "I cannot tell", "Increasing", False),
        (Task.READ_VALUE, "unknown", "0.5", False),
        # the solution itself holds no answer: nothing can match it
        (Task.READ_VALUE, "0.5", "unknown", False),
        (Task.READ_VALUE, "unknown", "unknown", False),
    ],
)
def test_is_correct(task: Task, response: str, solution: str, expected: bool) -> None:  # ruff: ignore[boolean-type-hint-positional-argument] -- parametrized expected value, pytest passes it positionally
    assert is_correct(task, response, solution) is expected


def test_task_result_scores_the_outcomes() -> None:
    question = Question("p", "a)")
    result = TaskResult(
        Task.JUMPS,
        (Outcome(question, "a", correct=True), Outcome(question, "b", correct=False), Outcome(question, "a", True)),  # ruff: ignore[boolean-positional-value-in-call] -- covers the positional construction of Outcome on purpose
        input_tokens=1,
        output_tokens=2,
    )
    assert result.correct == 2
    assert result.accuracy == pytest.approx(2 / 3)


def test_run_task_asks_every_question_and_sums_the_tokens() -> None:
    questions = [Question("q1", "Increasing"), Question("q2", "Decreasing"), Question("q3", "Not monotone")]
    replies = {"q1": "Increasing", "q2": "Increasing", "q3": "Not monotone."}
    calls = []

    def ask(*, question: str, answer_format: str) -> FakeReply:
        calls.append((question, answer_format))
        return counted(replies[question], input_tokens=10, output_tokens=2)

    with capture_logs() as logs:
        result = run_task(Task.MONOTONICITY, questions, ask)

    assert calls == [(q, Task.MONOTONICITY.answer_format) for q in ("q1", "q2", "q3")]
    assert result.task is Task.MONOTONICITY
    assert [outcome.correct for outcome in result.outcomes] == [True, False, True]
    assert [outcome.response for outcome in result.outcomes] == ["Increasing", "Increasing", "Not monotone."]
    assert result.outcomes[1].question == Question("q2", "Decreasing")
    assert (result.input_tokens, result.output_tokens) == (30, 6)
    assert [(log["task"], log["index"], log["total"], log["correct"]) for log in logs] == [
        ("monotonicity", 1, 3, True),
        ("monotonicity", 2, 3, False),
        ("monotonicity", 3, 3, True),
    ]


def test_task_result_without_outcomes_has_the_accuracy_zero() -> None:
    result = TaskResult(Task.JUMPS, (), input_tokens=0, output_tokens=0)
    assert result.correct == 0
    assert result.accuracy == pytest.approx(0.0)


def test_load_questions_reads_the_directory_from_a_string(tmp_path: Path) -> None:
    write_questions(tmp_path, Task.JUMPS, [["q1", "1"]])
    assert load_questions(Task.JUMPS, str(tmp_path)) == [Question("q1", "1")]


def test_run_task_without_questions_asks_nothing() -> None:
    def ask(*, question: str, answer_format: str) -> FakeReply:
        pytest.fail(f"no question is expected: {question}, {answer_format}")

    result = run_task(Task.JUMPS, [], ask)
    assert result.outcomes == ()
    assert result.accuracy == pytest.approx(0.0)


def test_baml_function_satisfies_the_benchmark_protocols() -> None:
    assert isinstance(baml_sdk.answer_benchmark_question, AskQuestion)
    reply = baml_sdk.Counted(text="a", usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=2))
    assert isinstance(reply, Reply)
    assert isinstance(reply.usage, Usage)
    assert isinstance(counted("a"), Reply)


def test_the_ask_question_protocol_matches_the_sdk_function() -> None:
    assert call_mismatches(AskQuestion, "answer_benchmark_question") == []
    assert sdk_signature("answer_benchmark_question").return_annotation is baml_sdk.Counted


def test_the_reply_protocol_matches_the_sdk_class() -> None:
    assert protocol_mismatches(Reply, "Counted", {Usage: "TokenUsage"}) == []


@pytest.mark.parametrize(("fake", "sdk_class"), [(FakeReply, "Counted"), (FakeUsage, "TokenUsage")])
def test_a_fake_reply_has_the_fields_of_the_sdk_class(fake: type, sdk_class: str) -> None:
    # a fake stands for an SDK class, so a field that `baml generate` renames or retypes must fail this test
    nested = {FakeUsage: baml_sdk.TokenUsage}
    ours = {field.name: nested.get(field.type, field.type) for field in attrs.fields(fake)}
    theirs = sdk_fields(sdk_class)
    assert ours.keys() == theirs.keys()
    assert all(same_type(ours[name], theirs[name]) for name in theirs)


def test_ask_question_protocol_takes_keyword_arguments_only() -> None:
    kinds = {name: parameter.kind for name, parameter in inspect.signature(AskQuestion.__call__).parameters.items()}
    assert kinds == {
        "self": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "question": inspect.Parameter.KEYWORD_ONLY,
        "answer_format": inspect.Parameter.KEYWORD_ONLY,
    }


def test_reply_protocol_members_are_read_only_properties() -> None:
    assert isinstance(Usage.input_tokens, property)
    assert isinstance(Usage.output_tokens, property)
    assert isinstance(Reply.text, property)
    assert isinstance(Reply.usage, property)


def test_format_results() -> None:
    question = Question("p", "1")
    results = [
        TaskResult(Task.JUMPS, (Outcome(question, "1", correct=True), Outcome(question, "2", correct=False)), 100, 20),
        TaskResult(Task.FUNCTION_RECOGNITION_HARD, (Outcome(question, "1", correct=True),), 7, 3),
    ]
    assert format_results(results).splitlines() == [
        "task                          correct  accuracy  in tokens  out tokens",
        "jumps                             1/2     50.0%        100          20",
        "function-recognition-hard         1/1    100.0%          7           3",
    ]


# The solutions of each task, in the format of the data files: Python writes the numbers with repr().
_NUMBERS = st.floats(allow_nan=False, allow_infinity=False).map(repr)
_SOLUTIONS = {
    Task.FUNCTION_RECOGNITION: st.sampled_from("abcde").map(lambda letter: f"{letter})"),
    Task.FUNCTION_RECOGNITION_HARD: st.sampled_from("abcde").map(lambda letter: f"{letter})"),
    Task.JUMPS: _NUMBERS,
    Task.MONOTONICITY: st.sampled_from(["Increasing", "Decreasing", "Not monotone"]),
    Task.READ_VALUE: _NUMBERS,
    Task.WIDE_CONFIDENCE: st.tuples(_NUMBERS, _NUMBERS).map(lambda bounds: f"({bounds[0]}, {bounds[1]})"),
}


def test_every_task_has_a_solution_strategy() -> None:
    assert set(_SOLUTIONS) == set(Task)


@given(data=st.data(), task=st.sampled_from(Task))
def test_every_well_formed_solution_matches_itself(data: st.DataObject, task: Task) -> None:
    solution = data.draw(_SOLUTIONS[task], label="solution")
    assert is_correct(task, solution, solution)


class WrongAsk(Protocol):
    """An AskQuestion that drifted from the SDK: a retyped parameter and a renamed one."""

    def __call__(self, *, question: int, answer: str) -> Reply:
        """Ask the question."""
        ...


class WrongReply(Protocol):
    """A Reply that drifted from the SDK: a retyped property and one that the SDK class lacks."""

    @property
    def text(self) -> int:
        """The answer."""
        ...

    @property
    def cost(self) -> float:
        """The price."""
        ...


def test_call_mismatches_reports_each_difference() -> None:
    assert call_mismatches(WrongAsk, "answer_benchmark_question") == [
        "answer_benchmark_question has no parameter answer",
        "the protocol lacks the parameter answer_format",
        "answer_benchmark_question(question) is <class 'str'>, the protocol passes <class 'int'>",
    ]


def test_protocol_mismatches_reports_each_difference() -> None:
    assert isinstance(WrongReply.cost, property)
    assert protocol_mismatches(WrongReply, "Counted", {}) == [
        "Counted.text is <class 'str'>, the protocol reads <class 'int'>",
        "Counted has no field cost",
    ]
