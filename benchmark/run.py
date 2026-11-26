"""Run the benchmark against an LLM.

Usage: ``uv run python benchmark/run.py [--task TASK ...] [--limit N] [--model PROVIDER/MODEL] [--mode MODE]
[--output FILE] [--no-progress]``. Makes real requests. The default model is OpenRouter's ``openrouter/free``, which
needs ``OPENROUTER_API_KEY`` (see the documentation page "Setup"). ``--model`` selects another model, and the key
variable of its provider.

``--mode`` chooses how each graph reaches the model. The questions hold their graph in one of two formats: the function
recognition tasks write ``Graph: {...}`` with no bounds, and the other tasks write the graph text of
``EBMGraph.to_text``. ``text`` sends the question as it is. ``vision`` replaces the values of the graph with a note and
sends the image of the graph. ``mixed`` sends the question with the image. The image needs the extra ``vision`` and a
model that reads images.
"""

import base64
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

import attrs
import cyclopts
import orjson
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml import EBMGraph, GraphImageOptions, GraphMode, graph_png
from ebaml.benchmark import Task, format_results, load_questions, run_task
from ebaml.core.exceptions import InvalidOptionError
from ebaml.graphs.graph_text import parse_str_tuple_to_float_tuple
from ebaml.llm.options import ModelOptions

DATA_DIR = Path(__file__).resolve().parent / "data"
# the lines of a question that hold the values of its graph: from the feature name to the last line of values
_DATA_BLOCK = re.compile(
    r"^Feature Name: (?P<name>.*)\n(?:Feature Type: .*\n)?(?:(?:Graph|Means|Lower Bounds|Upper Bounds)\b.*(?:\n|$))+",
    re.MULTILINE,
)
_GRAPH_LINE = re.compile(r"^Graph: (\{.*\})$", re.MULTILINE)
_LEVEL = re.compile(r"Bounds \((\d+(?:\.\d+)?)%-Confidence Interval\)")


@attrs.frozen
class QuestionView:
    """What one benchmark request sends.

    Attributes
    ----------
    question: str
        The text of the question.
    png_base64: str | None
        The PNG image of the graph of the question in base64, or ``None`` in text mode.

    """

    question: str
    png_base64: str | None


def question_graph(question: str) -> EBMGraph:
    """Read the graph of a benchmark question.

    Parameters
    ----------
    question: str
        The question, with its graph in one of the two formats.

    Returns
    -------
    EBMGraph
        The graph. A graph with no bounds in the question has standard deviations of zero.

    Raises
    ------
    ValueError
        If the question holds no graph.

    """
    graph_line = _GRAPH_LINE.search(question)
    if graph_line is None:
        return EBMGraph.from_text(question)
    values: dict[str, float] = orjson.loads(graph_line.group(1))
    block = _DATA_BLOCK.search(question)
    if block is None:
        msg = "The question has a graph line but no feature name."
        raise ValueError(msg)
    intervals = [parse_str_tuple_to_float_tuple(key) for key in values]
    return EBMGraph(block.group("name"), "continuous", intervals, list(values.values()), [0.0] * len(values))


def _confidence_level(question: str) -> float | None:
    """Read the level of the bounds of a question, or ``None`` for a question with no bounds.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    float | None
        The level, for example 0.95.

    """
    level = _LEVEL.search(question)
    return None if level is None else float(level.group(1)) / 100


def _png_base64(question: str) -> str:
    """Draw the graph of a question as a PNG image, with its bounds if the question has them.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    str
        The PNG image in base64.

    """
    options = GraphImageOptions(confidence_level=_confidence_level(question))
    return base64.b64encode(graph_png(question_graph(question), options)).decode("ascii")


def _without_values(question: str) -> str:
    """Replace the values of the graph of a question with a note that the image shows them.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    str
        The question with the feature name, and with no value of the graph.

    """
    return _DATA_BLOCK.sub(
        lambda block: (
            f"Feature Name: {block.group('name')}\n"
            "The values of the graph are not in this text. The image above shows the graph: read the values from its"
            " axes.\n"
        ),
        question,
        count=1,
    )


def _text_view(question: str) -> QuestionView:
    return QuestionView(question, None)


def _vision_view(question: str) -> QuestionView:
    return QuestionView(_without_values(question), _png_base64(question))


def _mixed_view(question: str) -> QuestionView:
    return QuestionView(question, _png_base64(question))


# the request of each mode; a new mode is a new function and a new row
_VIEWS: Mapping[GraphMode, Callable[[str], QuestionView]] = {
    GraphMode.TEXT: _text_view,
    GraphMode.VISION: _vision_view,
    GraphMode.MIXED: _mixed_view,
}


def question_view(question: str, mode: GraphMode) -> QuestionView:
    """Build the request of a benchmark question in a graph mode.

    Parameters
    ----------
    question: str
        The question, as the data file holds it.
    mode: GraphMode
        The mode.

    Returns
    -------
    QuestionView
        The question as the mode sends it, and the image of its graph in vision and mixed mode.

    """
    return _VIEWS[mode](question)


def model_spec(_type: type, tokens: Sequence[cyclopts.Token]) -> baml_sdk.ModelSpec:
    """Parse the ``provider/model`` argument and check that BAML knows the provider.

    Parameters
    ----------
    _type: type
        The annotated type. Cyclopts passes it and this converter ignores it.
    tokens: Sequence[cyclopts.Token]
        The one token of the argument, for example ``anthropic/claude-haiku-4-5``.

    Returns
    -------
    baml_sdk.ModelSpec
        The model, with the temperature and the output limit of the benchmark.

    Raises
    ------
    ValueError
        If the value is not ``provider/model``, or BAML does not know the provider.

    """
    try:
        options = ModelOptions.parse(tokens[0].value)
        spec = baml_sdk.ModelSpec(provider=options.provider, model=options.model)
        baml_sdk.check_model_sync(spec=spec)
    except (InvalidOptionError, BamlError) as error:
        raise ValueError(str(error)) from error
    return spec


@attrs.frozen
class ModeAsker:
    """Ask a benchmark question in a graph mode: the text, the image of its graph, or both.

    Attributes
    ----------
    mode: GraphMode
        The mode of every question.
    model: baml_sdk.ModelSpec | None
        The model, or ``None`` for ``openrouter/free``.

    """

    mode: GraphMode
    model: baml_sdk.ModelSpec | None

    def __call__(self, *, question: str, answer_format: str) -> baml_sdk.Counted:
        """Send the question in the mode of the asker.

        Parameters
        ----------
        question: str
            The question, as the data file holds it.
        answer_format: str
            How the model states its final answer.

        Returns
        -------
        baml_sdk.Counted
            The answer and its token usage.

        """
        view = question_view(question, self.mode)
        return baml_sdk.answer_benchmark_question_sync(
            question=view.question,
            answer_format=answer_format,
            model=self.model,
            png_base64=view.png_base64,
        )


app: cyclopts.App = cyclopts.App(name="run.py", help=__doc__.splitlines()[0], result_action="return_value")


@cyclopts.Parameter(name="*")
@attrs.frozen(kw_only=True)
class RunOptions:
    """The options of a run of the benchmark. Each field is one flag of the command.

    Attributes
    ----------
    task: list[Task], optional
        A task to run. Repeat the flag for more tasks. The default is all tasks.
    limit: int, optional
        Ask only the first N questions of each task.
    model: baml_sdk.ModelSpec, optional
        The model as provider/model, for example anthropic/claude-haiku-4-5. The default is openrouter/free.
    mode: GraphMode
        How each graph reaches the model: ``text`` (the question as it is), ``vision`` (the values of the graph replaced
        by its image) or ``mixed`` (the question with the image of its graph).
    output: Path, optional
        Write every question, response and verdict to this JSON file.
    progress: bool
        Show a progress bar on the error stream. If that stream is not a terminal, the bar writes one line for each
        question. Use ``--no-progress`` to turn it off.

    """

    task: Annotated[list[Task], cyclopts.Parameter(negative_iterable=())] | None = None
    limit: Annotated[int, cyclopts.Parameter(validator=cyclopts.validators.Number(gte=1))] | None = None
    model: (
        Annotated[
            baml_sdk.ModelSpec,
            cyclopts.Parameter(converter=model_spec, accepts_keys=False, n_tokens=1, metavar="PROVIDER/MODEL"),
        ]
        | None
    ) = None
    mode: GraphMode = GraphMode.TEXT
    output: Path | None = None
    progress: bool = True


@app.default
def run(options: RunOptions | None = None) -> int:
    """Run the tasks, print the table of results, and optionally save every response.

    Parameters
    ----------
    options: RunOptions, optional
        The flags of the command.

    Returns
    -------
    int
        The exit code.

    """
    options = RunOptions() if options is None else options
    ask = ModeAsker(options.mode, options.model)
    tasks = options.task or list(Task)
    results = [
        run_task(each, load_questions(each, DATA_DIR)[: options.limit], ask, progress=options.progress)
        for each in tasks
    ]
    sys.stdout.write(format_results(results) + "\n")
    if options.output:
        output = options.output
        output.write_bytes(orjson.dumps([attrs.asdict(result) for result in results], option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    sys.exit(app())
