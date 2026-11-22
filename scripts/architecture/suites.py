"""Fail when the test suites break the rules of ``.claude/rules/testing.md``.

The suites under ``tests/`` are ``unit``, ``integration``, ``api``, ``contract``, ``site`` and ``docs``. The unit suite
has its own guards (the mirrored layout of ``check_layout``, 100 % coverage and mutation testing). This check guards the
others that run with a marker, so that they stay thorough:

1. ``tests/`` holds the known suites only, and each suite is a package with a docstring.
2. Each of the suites ``integration``, ``api``, ``contract`` and ``site`` holds at least one test module, and each of
   its Python modules is a test module, a ``conftest.py`` or the ``__init__.py``.
3. An integration test uses the real ``baml_sdk``. A test of that suite that replaces a function of ``baml_sdk`` with a
   fake tests nothing that a unit test does not, so the check fails on ``fake_sdk`` and on ``setattr(baml_sdk, ...)``.
4. ``pyproject.toml`` registers the marker of each suite and runs with ``--strict-markers``. ``testpaths`` lists each
   suite that ``uv run pytest`` runs. The ``site`` suite builds the docs, so it is not in ``testpaths``.
5. A CI workflow runs each guarded suite: ``ci.yml`` the first three, and ``docs.yml`` the ``site`` suite.

Run it from the repo root: ``uv run python -m scripts.architecture.suites``.
"""

import ast
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KNOWN_SUITES = ("unit", "integration", "api", "contract", "site", "docs")
# The suites that ``uv run pytest`` runs, so ``testpaths`` lists them.
DEFAULT_SUITES = ("unit", "integration", "api", "contract")
# The suites that run through pytest with a marker.
MARKED_SUITES = (*DEFAULT_SUITES, "site")
HELPER_FILES = frozenset({"__init__.py", "conftest.py"})
WORKFLOWS = Path(".github") / "workflows"
# The suites that this check guards beyond the layout of the unit suite, with the workflow that runs each one.
SUITE_WORKFLOWS = {
    "integration": WORKFLOWS / "ci.yml",
    "api": WORKFLOWS / "ci.yml",
    "contract": WORKFLOWS / "ci.yml",
    "site": WORKFLOWS / "docs.yml",
}
GUARDED_SUITES: tuple[str, ...] = tuple(SUITE_WORKFLOWS)


def _has_docstring(path: Path) -> bool:
    return path.is_file() and bool(ast.get_docstring(ast.parse(path.read_bytes())))


def _patches_the_sdk(node: ast.AST) -> bool:
    """Tell whether a node is a call such as ``monkeypatch.setattr(baml_sdk, ...)``.

    Parameters
    ----------
    node: ast.AST
        Any node of a module.

    Returns
    -------
    bool
        True if the node is a call of ``setattr`` or ``setitem`` whose first argument is ``baml_sdk``.

    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
        return False
    first = node.args[0]
    return node.func.attr in {"setattr", "setitem"} and isinstance(first, ast.Name) and first.id == "baml_sdk"


def _fakes_the_sdk(tree: ast.AST) -> bool:
    """Tell whether a module replaces a function of ``baml_sdk`` with a fake.

    Parameters
    ----------
    tree: ast.AST
        The parsed module.

    Returns
    -------
    bool
        True if the module names ``fake_sdk`` or patches ``baml_sdk``.

    """
    return any(
        (isinstance(node, ast.Name) and node.id == "fake_sdk")
        or (isinstance(node, ast.alias) and node.name == "fake_sdk")
        or _patches_the_sdk(node)
        for node in ast.walk(tree)
    )


def suite_problems(tests: Path) -> list[str]:
    """List the ways in which the suite directories break the rules.

    Parameters
    ----------
    tests: Path
        The ``tests`` directory.

    Returns
    -------
    list[str]
        One message for each problem. The list is empty when the suites are right.

    """
    problems = [
        f"tests/{path.name}: not a known suite ({', '.join(KNOWN_SUITES)}) - add it to KNOWN_SUITES and to testing.md"
        for path in sorted(tests.iterdir())
        if path.is_dir() and path.name != "__pycache__" and path.name not in KNOWN_SUITES
    ]
    for name in KNOWN_SUITES:
        suite = tests / name
        if not suite.is_dir():
            problems.append(f"tests/{name}: the suite is missing")
        elif not _has_docstring(suite / "__init__.py"):
            problems.append(f"tests/{name}/__init__.py: missing, or it has no docstring")
    for name in GUARDED_SUITES:
        problems.extend(_guarded_suite_problems(tests, name))
    return problems


def _guarded_suite_problems(tests: Path, name: str) -> list[str]:
    suite = tests / name
    if not suite.is_dir():
        return []
    modules = sorted(path for path in suite.rglob("*.py") if "__pycache__" not in path.parts)
    problems = [
        f"tests/{name}/{path.relative_to(suite).as_posix()}: "
        "a module of a test suite is test_*.py, conftest.py or __init__.py"
        for path in modules
        if path.name not in HELPER_FILES and not path.name.startswith("test_")
    ]
    if not any(path.name.startswith("test_") for path in modules):
        problems.append(f"tests/{name}: the suite has no test module")
    if name == "integration":
        problems.extend(
            f"tests/{name}/{path.relative_to(suite).as_posix()}: replaces a function of baml_sdk with a fake; "
            "an integration test runs the real SDK and fakes the server (tests/conftest.py)"
            for path in modules
            if _fakes_the_sdk(ast.parse(path.read_bytes()))
        )
    return problems


def configuration_problems(pyproject: dict[str, Any], workflows: Mapping[Path, str]) -> list[str]:
    """List the ways in which the pytest configuration and the CI workflows leave a suite out.

    Parameters
    ----------
    pyproject: dict[str, Any]
        The parsed ``pyproject.toml``.
    workflows: Mapping[Path, str]
        The text of each workflow of ``SUITE_WORKFLOWS``, by its path. A missing workflow counts as empty.

    Returns
    -------
    list[str]
        One message for each problem.

    """
    pytest_options = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
    markers = [str(marker).split(":")[0] for marker in pytest_options.get("markers", [])]
    testpaths = pytest_options.get("testpaths", [])
    problems = []
    if "--strict-markers" not in str(pytest_options.get("addopts", "")):
        problems.append("pyproject.toml: addopts lacks --strict-markers, so a misspelled marker passes")
    for suite in MARKED_SUITES:
        if suite not in markers:
            problems.append(f"pyproject.toml: the marker {suite!r} is not registered")
        if suite in DEFAULT_SUITES and f"tests/{suite}" not in testpaths:
            problems.append(f"pyproject.toml: testpaths lacks tests/{suite}")
    problems.extend(
        f"{workflow.as_posix()}: no step runs tests/{suite}"
        for suite, workflow in SUITE_WORKFLOWS.items()
        if f"tests/{suite}" not in workflows.get(workflow, "")
    )
    return problems


def main(argv: Sequence[str] = ()) -> int:
    """Report every problem of the test suites.

    Parameters
    ----------
    argv: Sequence[str]
        The root of the project. The default is the root of this repository.

    Returns
    -------
    int
        The process exit code: 0 if the suites are right, 1 otherwise.

    """
    root = Path(argv[0]) if argv else ROOT
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workflows = {
        workflow: (root / workflow).read_text(encoding="utf-8")
        for workflow in set(SUITE_WORKFLOWS.values())
        if (root / workflow).is_file()
    }
    problems = [*suite_problems(root / "tests"), *configuration_problems(pyproject, workflows)]
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
