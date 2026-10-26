"""Benchmark of how well an LLM reads the graphs of Explainable Boosting Machines.

Port of the ``benchmark/`` of TalkToEBM. The questions are stored in ``benchmark/data`` as JSON lists of
``[prompt, solution]`` pairs. Each question is answered in two LLM turns (see ``answer_benchmark_question`` in
``baml_src/talk_to_ebm.baml``): the model reasons step by step, then states its final answer in a format that this
module parses and compares with the solution. TalkToEBM's notebooks printed the answers next to the solutions and
compared them by eye; here the comparison is automatic. The module imports no LLM client: the caller passes the
function that asks the question, so the parsing and the scoring run without the generated ``baml_sdk``.
"""

import enum
import os
import re
from collections.abc import Callable, Sequence
from itertools import starmap
from pathlib import Path
from typing import Protocol, runtime_checkable

import attrs
import orjson
import structlog

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)


@runtime_checkable
class Usage(Protocol):
    """The tokens that the provider reported for a reply."""

    @property
    def input_tokens(self) -> int:
        """The tokens of the prompt."""
        ...

    @property
    def output_tokens(self) -> int:
        """The tokens of the reply."""
        ...


@runtime_checkable
class Reply(Protocol):
    """The reply of the LLM to a question, with the tokens that it used."""

    @property
    def text(self) -> str:
        """The final answer of the LLM."""
        ...

    @property
    def usage(self) -> Usage:
        """The tokens that the provider reported."""
        ...


@runtime_checkable
class AskQuestion(Protocol):
    """A function that asks the LLM a question of the benchmark.

    ``baml_sdk.answer_benchmark_question`` is such a function.
    """

    def __call__(self, *, question: str, answer_format: str) -> Reply:
        """Ask the question and ask for the final answer in the format of ``answer_format``."""
        ...


# a parsed answer: a label, or the numbers of a value (one) or an interval (two)
Answer = str | tuple[float, ...]

# ".5" is a number too, so the digits before the decimal point are optional when digits follow it
_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_PATTERN = re.compile(_NUMBER)
_INTERVAL_PATTERN = re.compile(rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")
# "17,113.4" is one number, "17,113" in "(17,113)" too, but the "," of "2, 3" is not a separator
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# "a)", wherever it is, or a bare "a" at the start of the answer
_CHOICE_PATTERNS = (re.compile(r"\b([a-e])\)"), re.compile(r"^\W*([a-e])\b"))
# "not monotone" goes first: it contains none of the others, but an answer may well say "not increasing"
_MONOTONICITY_LABELS = ("not monotone", "increasing", "decreasing")


def _parse_choice(text: str) -> Answer | None:
    """Parse the letter of the chosen option.

    Parameters
    ----------
    text: str
        The answer, such as ``a)``, ``(b)`` or ``**c)**``.

    Returns
    -------
    Answer | None
        The lowercase letter, or None if the answer names no option.

    """
    lowered = text.lower()
    for pattern in _CHOICE_PATTERNS:
        if match := pattern.search(lowered):
            return match.group(1)
    return None


def _parse_monotonicity(text: str) -> Answer | None:
    """Parse the monotonicity of a graph.

    Parameters
    ----------
    text: str
        The answer, such as ``Increasing``, ``monotone decreasing`` or ``Not monotone.``.

    Returns
    -------
    Answer | None
        The lowercase label, or None if the answer states no monotonicity.

    """
    lowered = text.lower()
    return next((label for label in _MONOTONICITY_LABELS if label in lowered), None)


def _parse_number(text: str) -> Answer | None:
    """Parse the first number of an answer.

    Parameters
    ----------
    text: str
        The answer, such as ``0.0917``, ``-1e-05`` or ``17,113.4``.

    Returns
    -------
    Answer | None
        A tuple with the number, or None if the answer has no number.

    """
    match = _NUMBER_PATTERN.search(_THOUSANDS_SEPARATOR.sub("", text))
    return None if match is None else (float(match.group()),)


def _parse_interval(text: str) -> Answer | None:
    """Parse the first interval of an answer.

    Parameters
    ----------
    text: str
        The answer, such as ``(0.5, 1.5)``.

    Returns
    -------
    Answer | None
        A tuple with the lower and the upper bound, or None if the answer has no interval.

    """
    match = _INTERVAL_PATTERN.search(text)
    return None if match is None else (float(match.group(1)), float(match.group(2)))


class Task(enum.StrEnum):
    """The tasks of the benchmark. The value is the name of the data file."""

    FUNCTION_RECOGNITION = "function-recognition"
    FUNCTION_RECOGNITION_HARD = "function-recognition-hard"
    JUMPS = "jumps"
    MONOTONICITY = "monotonicity"
    READ_VALUE = "read-value"
    WIDE_CONFIDENCE = "wide-confidence"

    @property
    def answer_format(self) -> str:
        """How the model is asked to state its final answer, to complete "answering with ...".

        Returns
        -------
        str
            The description of the answer format of the task.

        """
        return _SPECS[self].answer_format


@attrs.frozen
class _TaskSpec:
    """What a task needs besides its name: the answer format that the LLM is asked for, and the parser of an answer."""

    answer_format: str
    parser: Callable[[str], Answer | None]


# a new task adds a member to Task and a row here
_SPECS = {
    Task.FUNCTION_RECOGNITION: _TaskSpec("the letter of the function, for example 'a)'", _parse_choice),
    Task.FUNCTION_RECOGNITION_HARD: _TaskSpec("the letter of the function, for example 'a)'", _parse_choice),
    Task.JUMPS: _TaskSpec("the x-axis position of the largest jump, as a single number", _parse_number),
    Task.MONOTONICITY: _TaskSpec("'Increasing', 'Decreasing' or 'Not monotone'", _parse_monotonicity),
    Task.READ_VALUE: _TaskSpec("the mean value of the graph, as a single number", _parse_number),
    Task.WIDE_CONFIDENCE: _TaskSpec("the interval, in the form '(lower, upper)'", _parse_interval),
}


@attrs.frozen
class Question:
    """A question of the benchmark."""

    prompt: str
    solution: str


@attrs.frozen
class Outcome:
    """The answer of the LLM to a question."""

    question: Question
    response: str
    correct: bool


@attrs.frozen
class TaskResult:
    """The outcomes of a task, with the tokens that the provider reported for them."""

    task: Task
    outcomes: tuple[Outcome, ...]
    input_tokens: int
    output_tokens: int

    @property
    def correct(self) -> int:
        """The number of questions that were answered correctly.

        Returns
        -------
        int
            The number of correct outcomes.

        """
        return sum(outcome.correct for outcome in self.outcomes)

    @property
    def accuracy(self) -> float:
        """The fraction of the questions that were answered correctly.

        Returns
        -------
        float
            The accuracy, between 0 and 1. A task with no outcomes has the accuracy 0.

        """
        if not self.outcomes:
            return 0.0
        return self.correct / len(self.outcomes)


def load_questions(task: Task, data_dir: str | os.PathLike[str]) -> list[Question]:
    """Load the questions of a task.

    Parameters
    ----------
    task: Task
        The task.
    data_dir: str | os.PathLike[str]
        The directory with the data files of the benchmark.

    Returns
    -------
    list[Question]
        The questions, in the order of the file.

    """
    records = orjson.loads((Path(data_dir) / f"{task}.json").read_bytes())
    return list(starmap(Question, records))


def parse_answer(task: Task, text: str) -> Answer | None:
    """Parse the answer to a question of a task, or its solution.

    Parameters
    ----------
    task: Task
        The task, which sets the format of the answer.
    text: str
        The text of the answer.

    Returns
    -------
    Answer | None
        The parsed answer, or None if the text does not hold an answer in the format of the task.

    """
    return _SPECS[task].parser(text)


def is_correct(task: Task, response: str, solution: str) -> bool:
    """Whether the response of the LLM states the solution.

    The comparison is made on the parsed answers, so the formatting of the response (``a`` or ``(a)``, ``0.50`` or
    ``.5``, markdown, a trailing period) does not matter, but the values must be exactly the same.

    Parameters
    ----------
    task: Task
        The task, which sets the format of the answer.
    response: str
        The final answer of the LLM.
    solution: str
        The solution of the question.

    Returns
    -------
    bool
        True if the response holds an answer, and it is the solution.

    """
    answer = parse_answer(task, response)
    return answer is not None and answer == parse_answer(task, solution)


def run_task(task: Task, questions: Sequence[Question], ask: AskQuestion) -> TaskResult:
    """Ask the LLM the questions of a task, one at a time, and score its answers.

    The questions are asked sequentially, because the free models are rate limited.

    Parameters
    ----------
    task: Task
        The task, which sets the answer format and the scoring.
    questions: Sequence[Question]
        The questions to ask, for example from ``load_questions``.
    ask: AskQuestion
        Asks the LLM one question, for example ``baml_sdk.answer_benchmark_question``. It chooses the model.

    Returns
    -------
    TaskResult
        The outcomes and the token usage. If there are no questions, there are no outcomes and the accuracy is 0.

    """
    outcomes = []
    input_tokens = output_tokens = 0
    for index, question in enumerate(questions, start=1):
        counted = ask(question=question.prompt, answer_format=task.answer_format)
        input_tokens += counted.usage.input_tokens
        output_tokens += counted.usage.output_tokens
        outcome = Outcome(question, counted.text, is_correct(task, counted.text, question.solution))
        logger.info("answered", task=str(task), index=index, total=len(questions), correct=outcome.correct)
        outcomes.append(outcome)
    return TaskResult(task, tuple(outcomes), input_tokens, output_tokens)


def format_results(results: Sequence[TaskResult]) -> str:
    """Format the results of the tasks as a table.

    Parameters
    ----------
    results: Sequence[TaskResult]
        The results, one per task.

    Returns
    -------
    str
        A table with a row per task: the correct answers, the accuracy and the tokens used.

    """
    lines = [f"{'task':<28}{'correct':>9}{'accuracy':>10}{'in tokens':>11}{'out tokens':>12}"]
    lines.extend(
        f"{result.task:<28}{f'{result.correct}/{len(result.outcomes)}':>9}{result.accuracy:>10.1%}"
        f"{result.input_tokens:>11}{result.output_tokens:>12}"
        for result in results
    )
    return "\n".join(lines)
