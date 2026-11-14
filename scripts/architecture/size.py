"""Fail when a file, a function or a class grows past a size limit, so that code stays easy to read as a whole.

A reader can keep about seven things in mind, and a function that needs more than two screens of scrolling makes the
reader lose the start. The limits below are rough, and they follow the logical structure of the code, not a taste for
short files:

1. A source file has at most ``SOURCE_FILE_LINES`` lines. A test file has no limit: a test is a list of independent
   cases, and the layout rule ties it to one source module, so a long list of cases is no sign of a messy module.
2. A function has at most ``FUNCTION_LINES`` lines, from its ``def`` line to its last line, without its docstring.
   Documentation is not logic, so a limit that counts it pushes authors to write less of it.
3. A class has at most ``PUBLIC_METHODS_PER_CLASS`` public methods and at most ``PRIVATE_METHODS_PER_CLASS`` private
   methods. A dunder method counts as neither.

Run it as a module from the repo root: ``uv run python -m scripts.architecture.size``. ``.claude/rules/size.md``
says how to split a unit that is too big.
"""

import ast
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs

ROOT = Path(__file__).resolve().parents[2]
# The directories whose files the project writes. ``src/baml_sdk`` is generated, so it is not here.
SOURCE_ROOTS = ("src/ebaml", "scripts", "benchmark")
TEST_ROOTS = ("tests",)
SOURCE_FILE_LINES = 800
FUNCTION_LINES = 100
PUBLIC_METHODS_PER_CLASS = 15
PRIVATE_METHODS_PER_CLASS = 10


@attrs.frozen
class Measure:
    """The size of one unit: a file, a function or a class, with the limit that applies to it."""

    key: str
    size: int
    limit: int
    what: str

    @property
    def is_over(self) -> bool:
        """Say whether the unit is over its limit."""
        return self.size > self.limit


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _code_lines(node: FunctionNode) -> int:
    # from the ``def`` line to the last line, without the docstring
    total = (node.end_lineno or node.lineno) - node.lineno + 1
    first = node.body[0]
    is_docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    return total - ((first.end_lineno or first.lineno) - first.lineno + 1 if is_docstring else 0)


def _method_counts(node: ast.ClassDef) -> tuple[int, int]:
    names = [member.name for member in node.body if isinstance(member, FunctionNode)]
    methods = [name for name in names if not (name.startswith("__") and name.endswith("__"))]
    private = sum(name.startswith("_") for name in methods)
    return len(methods) - private, private


def _unit_measures(parent: ast.AST, prefix: str) -> Iterator[Measure]:
    for node in ast.iter_child_nodes(parent):
        if isinstance(node, ast.ClassDef):
            key = f"{prefix}::{node.name}" if "::" not in prefix else f"{prefix}.{node.name}"
            public, private = _method_counts(node)
            yield Measure(f"{key}:public", public, PUBLIC_METHODS_PER_CLASS, "public methods")
            yield Measure(f"{key}:private", private, PRIVATE_METHODS_PER_CLASS, "private methods")
            yield from _unit_measures(node, key)
        elif isinstance(node, FunctionNode):
            key = f"{prefix}::{node.name}" if "::" not in prefix else f"{prefix}.{node.name}"
            lines = _code_lines(node)
            yield Measure(key, lines, FUNCTION_LINES, "lines")
            yield from _unit_measures(node, key)
        else:
            yield from _unit_measures(node, prefix)


def measure_file(path: Path, relative: str, *, is_test: bool) -> list[Measure]:
    """Measure a Python file and every function and class in it.

    Parameters
    ----------
    path: Path
        The file to read.
    relative: str
        The path from the repo root, with ``/``, which names the file in a message.
    is_test: bool
        Whether the file is a test, whose length has no limit.

    Returns
    -------
    list[Measure]
        One measure for a source file, then one for each function and one for each side of each class.

    """
    source = path.read_text(encoding="utf-8")
    units = list(_unit_measures(ast.parse(source), relative))
    if is_test:
        return units
    return [Measure(relative, len(source.splitlines()), SOURCE_FILE_LINES, "lines"), *units]


def project_measures(root: Path = ROOT) -> list[Measure]:
    """Measure every Python file under the source and test roots.

    Parameters
    ----------
    root: Path
        The repo root.

    Returns
    -------
    list[Measure]
        The measures of all files, in path order.

    """
    measures: list[Measure] = []
    for roots, is_test in ((SOURCE_ROOTS, False), (TEST_ROOTS, True)):
        paths = sorted(path for directory in roots for path in (root / directory).rglob("*.py"))
        measures.extend(
            measure
            for path in paths
            for measure in measure_file(path, path.relative_to(root).as_posix(), is_test=is_test)
        )
    return measures


def size_problems(measures: Iterable[Measure]) -> list[str]:
    """Compare the measures with the limits.

    Parameters
    ----------
    measures: Iterable[Measure]
        The sizes of the units.

    Returns
    -------
    list[str]
        One message for each unit that is over its limit.

    """
    return [
        f"{measure.key}: {measure.size} {measure.what}, and the limit is {measure.limit}. "
        "Split the unit. See .claude/rules/size.md."
        for measure in measures
        if measure.is_over
    ]


def main() -> int:
    """Report every unit that is over its limit.

    Returns
    -------
    int
        The process exit code: 0 if no unit is over a limit, 1 otherwise.

    """
    problems = size_problems(project_measures())
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
