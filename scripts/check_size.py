"""Fail when a file, a function or a class grows past a size limit, so that code stays easy to read as a whole.

A reader can keep about seven things in mind, and a function that needs more than two screens of scrolling makes the
reader lose the start. The limits below are rough, and they follow the logical structure of the code, not a taste for
short files:

1. A source file has at most ``SOURCE_FILE_LINES`` lines, and a test file has at most ``TEST_FILE_LINES`` lines.
2. A function has at most ``FUNCTION_LINES`` lines, from its ``def`` line to its last line.
3. A class has at most ``METHODS_PER_CLASS`` public methods and at most ``METHODS_PER_CLASS`` private methods. A
   dunder method counts as neither.

``scripts/size_baseline.toml`` holds the units that are over a limit today, each with its current size and a reason. The
size of such a unit may not grow, and the entry goes when the unit is under the limit again. A shrunk unit must lower
its entry, so the baseline only moves down.

Run it as a module from the repo root: ``uv run python -m scripts.check_size``. ``.claude/rules/size.md`` says how to
split a unit that is too big.
"""

import ast
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import attrs

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "size_baseline.toml"
# The directories whose files the project writes. ``src/baml_sdk`` is generated, so it is not here.
SOURCE_ROOTS = ("src/ebaml", "scripts", "benchmark")
TEST_ROOTS = ("tests",)
SOURCE_FILE_LINES = 800
TEST_FILE_LINES = 1200
FUNCTION_LINES = 100
METHODS_PER_CLASS = 10


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
            yield Measure(f"{key}:public", public, METHODS_PER_CLASS, "public methods")
            yield Measure(f"{key}:private", private, METHODS_PER_CLASS, "private methods")
            yield from _unit_measures(node, key)
        elif isinstance(node, FunctionNode):
            key = f"{prefix}::{node.name}" if "::" not in prefix else f"{prefix}.{node.name}"
            lines = (node.end_lineno or node.lineno) - node.lineno + 1
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
        The path from the repo root, with ``/``, which names the file in the baseline.
    is_test: bool
        Whether the file is a test, which has a higher limit for its length.

    Returns
    -------
    list[Measure]
        One measure for the file, then one for each function and one for each side of each class.

    """
    source = path.read_text(encoding="utf-8")
    limit = TEST_FILE_LINES if is_test else SOURCE_FILE_LINES
    file_measure = Measure(relative, len(source.splitlines()), limit, "lines")
    return [file_measure, *_unit_measures(ast.parse(source), relative)]


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


def size_problems(measures: Iterable[Measure], baseline: Mapping[str, int]) -> list[str]:
    """Compare the measures with the limits and with the baseline.

    Parameters
    ----------
    measures: Iterable[Measure]
        The sizes of the units.
    baseline: Mapping[str, int]
        The allowed size of each unit that is over its limit today.

    Returns
    -------
    list[str]
        One message for each unit that grew past its limit or its entry, each entry that is stale, and each entry that
        names no unit.

    """
    by_key = {measure.key: measure for measure in measures}
    problems: list[str] = []
    for measure in by_key.values():
        allowed = baseline.get(measure.key, measure.limit)
        if measure.size > allowed:
            problems.append(
                f"{measure.key}: {measure.size} {measure.what}, and the limit is {allowed}. "
                "Split the unit. See .claude/rules/size.md.",
            )
    for key, allowed in baseline.items():
        measure = by_key.get(key)
        if measure is None:
            problems.append(f"{key}: the baseline names a unit that does not exist. Delete the entry.")
        elif not measure.is_over:
            problems.append(f"{key}: now {measure.size}, under the limit {measure.limit}. Delete the baseline entry.")
        elif measure.size < allowed:
            problems.append(f"{key}: now {measure.size}, and the entry says {allowed}. Lower the entry.")
    return problems


def main(argv: Sequence[str] = ()) -> int:
    """Report every unit that is over its limit and not in the baseline, and every stale entry.

    Parameters
    ----------
    argv: Sequence[str]
        With ``--baseline``, print the entries for every unit that is over its limit, for a new baseline file.

    Returns
    -------
    int
        The process exit code: 0 if no unit is over a limit, 1 otherwise.

    """
    measures = project_measures()
    if "--baseline" in argv:
        for measure in measures:
            if measure.is_over:
                sys.stdout.write(f'"{measure.key}" = {measure.size}\n')
        return 0
    baseline: Mapping[str, int] = tomllib.loads(BASELINE.read_text(encoding="utf-8"))["allowed"]
    problems = size_problems(measures, baseline)
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
