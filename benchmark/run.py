"""Run the benchmark against an LLM.

Usage: ``uv run python benchmark/run.py [--task TASK ...] [--limit N] [--model PROVIDER/MODEL] [--output FILE]``. Makes
real requests. The default model is OpenRouter's ``openrouter/free``, which needs ``OPENROUTER_API_KEY`` (see the
README). ``--model`` selects another model, and the key variable of its provider.
"""

import argparse
import functools
import sys
from pathlib import Path

import attrs
import orjson
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.benchmark import Task, format_results, load_questions, run_task
from ebaml.core.exceptions import InvalidOptionError
from ebaml.llm.options import ModelOptions

DATA_DIR = Path(__file__).resolve().parent / "data"


def positive_int(value: str) -> int:
    """Parse a positive integer argument.

    Parameters
    ----------
    value: str
        The text of the argument.

    Returns
    -------
    int
        The integer.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a positive integer.

    """
    number = int(value)
    if number < 1:
        msg = f"{value} is not a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return number


def model_spec(value: str) -> baml_sdk.ModelSpec:
    """Parse the ``provider/model`` argument and check that BAML knows the provider.

    Parameters
    ----------
    value: str
        The text of the argument, for example ``anthropic/claude-haiku-4-5``.

    Returns
    -------
    baml_sdk.ModelSpec
        The model, with the temperature and the output limit of the benchmark.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not ``provider/model``, or BAML does not know the provider.

    """
    try:
        options = ModelOptions.parse(value)
        spec = baml_sdk.ModelSpec(provider=options.provider, model=options.model)
        baml_sdk.check_model(spec=spec)
    except (InvalidOptionError, BamlError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return spec


def main(argv: list[str] | None = None) -> int:
    """Run the tasks, print the table of results, and optionally save every response.

    Parameters
    ----------
    argv: list[str], optional
        The command line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", type=Task, help="a task to run, repeatable (default: all)")
    parser.add_argument("--limit", type=positive_int, help="only ask the first N questions of each task")
    parser.add_argument(
        "--model",
        type=model_spec,
        help="the model as provider/model, for example anthropic/claude-haiku-4-5 (default: openrouter/free)",
    )
    parser.add_argument("--output", type=Path, help="write every question, response and verdict to this JSON file")
    args = parser.parse_args(argv)

    ask = functools.partial(baml_sdk.answer_benchmark_question, model=args.model)
    results = [run_task(task, load_questions(task, DATA_DIR)[: args.limit], ask) for task in args.task or list(Task)]
    sys.stdout.write(format_results(results) + "\n")
    if args.output:
        args.output.write_bytes(orjson.dumps([attrs.asdict(result) for result in results], option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
