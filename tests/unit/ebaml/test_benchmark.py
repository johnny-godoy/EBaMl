"""Tests for the benchmark."""

from pathlib import Path

import orjson
import pytest
from structlog.testing import capture_logs

import baml_sdk
from ebaml.benchmark import (
    Outcome,
    Question,
    Task,
    TaskResult,
    format_results,
    is_correct,
    load_questions,
    parse_answer,
    run_task,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"


def counted(text: str, input_tokens: int = 3, output_tokens: int = 5) -> baml_sdk.Counted:
    usage = baml_sdk.TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return baml_sdk.Counted(text=text, usage=usage)


def write_questions(tmp_path: Path, task: Task, records: list[list[str]]) -> None:
    (tmp_path / f"{task}.json").write_bytes(orjson.dumps(records))


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


def test_run_task_asks_every_question_and_sums_the_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_questions(tmp_path, Task.MONOTONICITY, [["q1", "Increasing"], ["q2", "Decreasing"], ["q3", "Not monotone"]])
    replies = {"q1": "Increasing", "q2": "Increasing", "q3": "Not monotone."}
    calls = []

    def fake(*, question: str, answer_format: str) -> baml_sdk.Counted:
        calls.append((question, answer_format))
        return counted(replies[question], input_tokens=10, output_tokens=2)

    monkeypatch.setattr(baml_sdk, "answer_benchmark_question", fake)
    with capture_logs() as logs:
        result = run_task(Task.MONOTONICITY, tmp_path)

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


def test_run_task_reads_the_directory_from_a_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_questions(tmp_path, Task.JUMPS, [["q1", "1"]])
    monkeypatch.setattr(baml_sdk, "answer_benchmark_question", lambda **_kw: counted("1"))
    assert run_task(Task.JUMPS, str(tmp_path)).correct == 1


def test_run_task_limit_zero_asks_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_questions(tmp_path, Task.JUMPS, [["q1", "1"]])
    monkeypatch.setattr(baml_sdk, "answer_benchmark_question", lambda **_kw: pytest.fail("no question is expected"))
    result = run_task(Task.JUMPS, tmp_path, limit=0)
    assert result.outcomes == ()
    assert result.accuracy == pytest.approx(0.0)


def test_run_task_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_questions(tmp_path, Task.JUMPS, [["q1", "1"], ["q2", "2"], ["q3", "3"]])
    asked = []

    def fake(*, question: str, answer_format: str) -> baml_sdk.Counted:  # ruff: ignore[unused-function-argument] -- fake must accept the keywords of baml_sdk.answer_benchmark_question
        asked.append(question)
        return counted("1")

    monkeypatch.setattr(baml_sdk, "answer_benchmark_question", fake)
    result = run_task(Task.JUMPS, tmp_path, limit=2)
    assert asked == ["q1", "q2"]
    assert len(result.outcomes) == 2


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
