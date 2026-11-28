"""Tests of the table of workloads of the performance benchmark."""

import ast
from pathlib import Path

from benchmark.performance.workloads import WORKLOADS

import ebaml


def test_each_workload_has_a_name_of_its_own() -> None:
    names = [workload.name for workload in WORKLOADS]
    assert len(names) == len(set(names))


def test_each_name_that_a_workload_requires_is_a_public_name_of_ebaml() -> None:
    required = {workload.requires for workload in WORKLOADS if workload.requires is not None}
    assert required <= set(ebaml.__all__)
    assert "RenderTiming" in required


# The modules of ebaml, other than the package itself, that the workloads may import. The base of a pull request runs
# the workloads of its head, so a module that moves or goes away breaks the base run at import (as render_graph did).
# Add a module only if it has a stable place, and prefer a public name of ebaml.
STABLE_MODULES = frozenset(
    {
        "ebaml.core.custom_types",
        "ebaml.core.ebm_classes",
        "ebaml.graphs.token_budget",
        "ebaml.mcp.model_store",
        "ebaml.mcp.server",
    },
)
PERFORMANCE = Path(__file__).resolve().parents[4] / "benchmark" / "performance"


def _ebaml_imports() -> list[tuple[str, str, str]]:
    """List each import from ebaml of the modules of the performance benchmark.

    Returns
    -------
    list[tuple[str, str, str]]
        The file, the module and the imported name of each import.

    """
    found: list[tuple[str, str, str]] = []
    for path in sorted(PERFORMANCE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ebaml":
                found += [(path.name, node.module or "", alias.name) for alias in node.names]
    return found


def test_the_workloads_import_public_names_of_ebaml_or_stable_modules_only() -> None:
    imports = _ebaml_imports()
    assert imports, "the scan must find the imports of the workloads"
    for file_name, module, name in imports:
        if module == "ebaml":
            assert name in ebaml.__all__, f"{file_name} imports {name}, which ebaml does not export"
        else:
            assert module in STABLE_MODULES, f"{file_name} imports {module}; use a public name of ebaml"
