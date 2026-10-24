"""Fail when the directory layout of ``src/ebaml`` and ``tests/unit/ebaml`` breaks the project rules.

import-linter limits what a module imports. This check limits where a module lives. The rules are:

1. Each package of ``src/ebaml`` is one of the layers in the ``[tool.importlinter]`` layers contract, so no package
   escapes the import limits.
2. Each package has an ``__init__.py`` with a docstring that says what the package holds.
3. Only the modules in ``TOP_LEVEL_MODULES`` sit directly in ``src/ebaml``. A new module goes into a layer.
4. Each module name is unique in ``src/ebaml``, because ``scripts/mutation.py`` finds the tests of a module by name.
5. The test tree mirrors the source tree. Each module has a test file at the same path under ``tests/unit/ebaml``, with
   ``test_`` before the name. Each test file and test directory has a source module or package at the same path.

Run it from the repo root: ``uv run python -m scripts.check_layout``.
"""

import ast
import sys
import tomllib
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# The benchmark has no import of another part of ebaml, so it is a layer of its own.
TOP_LEVEL_MODULES = frozenset({"__init__.py", "benchmark.py"})
LAYERS_CONTRACT = "layers"
# The tests of a package file ``__init__.py`` cannot be named ``test___init__.py``, so they have a short name.
INIT_TESTS = {"test_init.py": "__init__.py"}


def layer_names(pyproject: dict[str, Any]) -> list[str]:
    """Read the layer packages from the import-linter layers contract.

    Parameters
    ----------
    pyproject: dict[str, Any]
        The parsed ``pyproject.toml``.

    Returns
    -------
    list[str]
        The last part of each layer name, for example ``llm`` for ``ebaml.llm``.

    """
    contracts = pyproject.get("tool", {}).get("importlinter", {}).get("contracts", [])
    return [
        layer.rsplit(".", 1)[-1]
        for contract in contracts
        if contract.get("type") == LAYERS_CONTRACT
        for layer in contract["layers"]
    ]


def _has_docstring(path: Path) -> bool:
    """Tell whether a Python file starts with a docstring.

    Parameters
    ----------
    path: Path
        The Python file.

    Returns
    -------
    bool
        True if the file has a non-empty docstring.

    """
    return bool(ast.get_docstring(ast.parse(path.read_bytes())))


def layout_problems(src: Path, tests: Path, layers: Collection[str], top_level: Collection[str]) -> list[str]:
    """List the ways in which the layout breaks the rules.

    Parameters
    ----------
    src: Path
        The package directory, for example ``src/ebaml``.
    tests: Path
        The test directory of the package, for example ``tests/unit/ebaml``.
    layers: Collection[str]
        The names of the packages that the import limits cover.
    top_level: Collection[str]
        The file names that may sit directly in ``src``.

    Returns
    -------
    list[str]
        One message for each problem. The list is empty when the layout is right.

    """
    problems = []
    packages = sorted(path for path in src.iterdir() if path.is_dir() and path.name != "__pycache__")
    problems.extend(
        f"{package.name}: not a layer in the import-linter layers contract - add it there or remove it"
        for package in packages
        if package.name not in layers
    )
    problems.extend(
        f"{layer}: a layer of the import-linter layers contract has no package in {src}"
        for layer in layers
        if not (src / layer).is_dir()
    )
    problems.extend(
        f"{package.name}/__init__.py: missing, or it has no docstring"
        for package in packages
        if not (package / "__init__.py").is_file() or not _has_docstring(package / "__init__.py")
    )
    problems.extend(
        f"{path.name}: a new module goes into a layer package, not directly into {src.name}"
        for path in sorted(src.glob("*.py"))
        if path.name not in top_level
    )
    modules = sorted(path for path in src.rglob("*.py") if path.name != "__init__.py")
    counts = Counter(path.stem for path in modules)
    problems.extend(
        f"{path.relative_to(src)}: the module name is not unique in {src.name}"
        for path in modules
        if counts[path.stem] > 1
    )
    problems.extend(
        f"{path.relative_to(src)}: no test file {test.relative_to(tests.parent)}"
        for path in modules
        if not (test := tests / path.relative_to(src).with_name(f"test_{path.name}")).is_file()
    )
    problems.extend(_orphan_tests(src, tests))
    return problems


def _orphan_tests(src: Path, tests: Path) -> list[str]:
    """List the test files and test directories that have no source at the mirrored path.

    Parameters
    ----------
    src: Path
        The package directory.
    tests: Path
        The test directory of the package.

    Returns
    -------
    list[str]
        One message for each test file or directory without a source.

    """
    problems = [
        f"{path.relative_to(tests.parent)}: no package {path.relative_to(tests).as_posix()} in {src.name}"
        for path in sorted(tests.rglob("*"))
        if path.is_dir() and path.name != "__pycache__" and not (src / path.relative_to(tests)).is_dir()
    ]
    problems.extend(
        f"{path.relative_to(tests.parent)}: no module {source.relative_to(src.parent)}"
        for path in sorted(tests.rglob("test_*.py"))
        if not (
            source := src
            / path.relative_to(tests).with_name(INIT_TESTS.get(path.name, path.name.removeprefix("test_")))
        ).is_file()
    )
    return problems


def main(argv: Sequence[str] = ()) -> int:
    """Report every layout problem.

    Parameters
    ----------
    argv: Sequence[str]
        The root of the project. The default is the root of this repository.

    Returns
    -------
    int
        The process exit code: 0 if the layout is right, 1 otherwise.

    """
    root = Path(argv[0]) if argv else ROOT
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    problems = layout_problems(
        root / "src" / "ebaml",
        root / "tests" / "unit" / "ebaml",
        layer_names(pyproject),
        TOP_LEVEL_MODULES,
    )
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
