"""Run a benchmark of EBaMl.

Usage: ``uv run python benchmark/run.py KIND [FLAGS]``. Each kind is one subcommand:

- ``accuracy`` measures how well an LLM reads the graphs of an EBM. It sends real requests.
- ``performance`` measures the time of the work of EBaMl around the LLM requests. It sends no real request.

``uv run python benchmark/run.py KIND --help`` lists the flags of a kind. ``benchmark/README.md`` has the steps to add a
kind.
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import cyclopts

# The command runs by path, so Python puts benchmark/ on the path and not the root of the repository. The kinds are
# modules of the package benchmark, so the root goes on the path before cyclopts imports a kind.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Each kind of benchmark, as the import path of its cyclopts app. A new kind is a new module and a new row. Cyclopts
# imports a kind only when the command runs it, so a kind does not pay for the imports of the others.
KINDS: Mapping[str, str] = {
    "accuracy": "benchmark.accuracy.command:app",
    "performance": "benchmark.performance.command:app",
}

app: cyclopts.App = cyclopts.App(name="run.py", help=__doc__.splitlines()[0], result_action="return_value")
for kind, import_path in KINDS.items():
    app.command(import_path, name=kind)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Parameters
    ----------
    argv: Sequence[str], optional
        The arguments, for example ``["accuracy", "--task", "jumps"]``. The default reads them from the command line.

    Returns
    -------
    int
        The exit code.

    """
    result = app(sys.argv[1:] if argv is None else list(argv))
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
