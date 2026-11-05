"""Run the benchmark against an LLM.

Usage: ``uv run python benchmark/run.py [--task TASK ...] [--limit N] [--model PROVIDER/MODEL] [--output FILE]``. Makes
real requests. The default model is OpenRouter's ``openrouter/free``, which needs ``OPENROUTER_API_KEY`` (see the
README). ``--model`` selects another model, and the key variable of its provider.
"""

import functools
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import attrs
import cyclopts
import orjson
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.benchmark import Task, format_results, load_questions, run_task
from ebaml.core.exceptions import InvalidOptionError
from ebaml.llm.options import ModelOptions

DATA_DIR = Path(__file__).resolve().parent / "data"


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
        baml_sdk.check_model(spec=spec)
    except (InvalidOptionError, BamlError) as error:
        raise ValueError(str(error)) from error
    return spec


app: cyclopts.App = cyclopts.App(name="run.py", help=__doc__.splitlines()[0], result_action="return_value")


@app.default
def run(
    *,
    task: Annotated[list[Task], cyclopts.Parameter(negative_iterable=())] | None = None,
    limit: Annotated[int, cyclopts.Parameter(validator=cyclopts.validators.Number(gte=1))] | None = None,
    model: Annotated[
        baml_sdk.ModelSpec,
        cyclopts.Parameter(converter=model_spec, accepts_keys=False, n_tokens=1, metavar="PROVIDER/MODEL"),
    ]
    | None = None,
    output: Path | None = None,
) -> int:
    """Run the tasks, print the table of results, and optionally save every response.

    Parameters
    ----------
    task: list[Task], optional
        A task to run. Repeat the flag for more tasks. The default is all tasks.
    limit: int, optional
        Ask only the first N questions of each task.
    model: baml_sdk.ModelSpec, optional
        The model as provider/model, for example anthropic/claude-haiku-4-5. The default is openrouter/free.
    output: Path, optional
        Write every question, response and verdict to this JSON file.

    Returns
    -------
    int
        The exit code.

    """
    ask = functools.partial(baml_sdk.answer_benchmark_question, model=model)
    results = [run_task(each, load_questions(each, DATA_DIR)[:limit], ask) for each in task or list(Task)]
    sys.stdout.write(format_results(results) + "\n")
    if output:
        output.write_bytes(orjson.dumps([attrs.asdict(result) for result in results], option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    sys.exit(app())
