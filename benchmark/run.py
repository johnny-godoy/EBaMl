"""Run the benchmark against the LLM configured in ``baml_src/talk_to_ebm.baml``.

Usage: ``uv run python benchmark/run.py [--task TASK ...] [--limit N] [--output FILE]``. Makes real requests, so it
needs ``OPENROUTER_API_KEY`` (see the README).
"""

import argparse
import sys
from pathlib import Path

import attrs
import orjson

import baml_sdk
from ebaml.benchmark import Task, format_results, load_questions, run_task

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
    parser.add_argument("--output", type=Path, help="write every question, response and verdict to this JSON file")
    args = parser.parse_args(argv)

    results = [
        run_task(task, load_questions(task, DATA_DIR)[: args.limit], baml_sdk.answer_benchmark_question)
        for task in args.task or list(Task)
    ]
    sys.stdout.write(format_results(results) + "\n")
    if args.output:
        args.output.write_bytes(orjson.dumps([attrs.asdict(result) for result in results], option=orjson.OPT_INDENT_2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
