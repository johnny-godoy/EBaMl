"""Each call of ``ebaml`` into ``baml_sdk``, read from the source, against the stub that ``baml generate`` wrote.

The unit tests check each fake against the stub, so a fake cannot drift. These tests check the real call sites, so a
call cannot drift from the stub either, even on a branch that no unit test runs.
"""

import ast
from pathlib import Path
from typing import override

import pytest

from tests.unit.ebaml.conftest import sdk_fields, sdk_function_names, sdk_signature

ROOT = Path(__file__).resolve().parents[2]
SOURCES: list[Path] = sorted([*(ROOT / "src" / "ebaml").rglob("*.py"), *(ROOT / "benchmark").rglob("*.py")])
# The functions of baml_sdk that ebaml calls. The BAML code has many more, and they run inside the BAML runtime. A new
# call of the sources needs a row here, so that a reviewer sees each new crossing of the boundary. The benchmark
# passes answer_benchmark_question on as a value, so it is a reference and no call.
CALLED = frozenset({
    "chat_with_mcp",
    "check_model",
    "default_graph_options",
    "describe_graph_cot",
    "explain_local",
    "sample_graph_tokens",
    "summarize_ebm",
    "supported_providers",
})
REFERENCED_ONLY = frozenset({"answer_benchmark_question"})


class _Call(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ast.Call]] = []
        self.references: set[str] = set()

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "baml_sdk":
            self.references.add(node.attr)
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "baml_sdk"
        ):
            self.calls.append((function.attr, node))
        self.generic_visit(node)


def _scan() -> _Call:
    scanner = _Call()
    for path in SOURCES:
        scanner.visit(ast.parse(path.read_bytes()))
    return scanner


def _class_names() -> set[str]:
    stub = ast.parse((ROOT / "src" / "baml_sdk" / "__init__.pyi").read_text(encoding="utf-8"))
    return {node.name for node in stub.body if isinstance(node, ast.ClassDef)}


SCAN: _Call = _scan()
FUNCTION_CALLS: list[tuple[str, ast.Call]] = [(name, call) for name, call in SCAN.calls if name in sdk_function_names()]
CLASS_CALLS: list[tuple[str, ast.Call]] = [(name, call) for name, call in SCAN.calls if name in _class_names()]


def test_the_sources_call_exactly_the_functions_of_the_table() -> None:
    assert {name for name, _ in FUNCTION_CALLS} == CALLED
    assert SCAN.references >= REFERENCED_ONLY


def test_every_name_of_baml_sdk_that_the_sources_use_exists_in_the_generated_stub() -> None:
    known = sdk_function_names() | _class_names()
    assert SCAN.references <= known, f"unknown names of baml_sdk: {sorted(SCAN.references - known)}"


def _arguments(call: ast.Call) -> tuple[set[str], int, bool]:
    keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    spreads = any(keyword.arg is None for keyword in call.keywords) or any(
        isinstance(argument, ast.Starred) for argument in call.args
    )
    return keywords, len(call.args), spreads


@pytest.mark.parametrize(
    ("name", "call"),
    FUNCTION_CALLS,
    ids=[f"{name}:{call.lineno}" for name, call in FUNCTION_CALLS],
)
def test_a_call_of_a_function_names_only_parameters_and_leaves_no_required_one_out(name: str, call: ast.Call) -> None:
    parameters = sdk_signature(name).parameters
    keywords, positional, spreads = _arguments(call)
    assert keywords <= set(parameters), f"{name} has no parameters {sorted(keywords - set(parameters))}"
    if spreads:
        return
    given = keywords | set(list(parameters)[:positional])
    required = {parameter.name for parameter in parameters.values() if parameter.default is parameter.empty}
    assert required <= given, f"{name} lacks the required parameters {sorted(required - given)}"


@pytest.mark.parametrize(("name", "call"), CLASS_CALLS, ids=[f"{name}:{call.lineno}" for name, call in CLASS_CALLS])
def test_a_construction_of_an_sdk_class_names_only_fields(name: str, call: ast.Call) -> None:
    keywords, _positional, _spreads = _arguments(call)
    assert keywords <= set(sdk_fields(name)), f"{name} has no fields {sorted(keywords - set(sdk_fields(name)))}"


def test_no_call_site_uses_the_async_variant_of_a_function() -> None:
    assert not [name for name, _ in SCAN.calls if name.endswith("_async")]
    assert not [name for name in SCAN.references if name.endswith(("_async", "_stream", "_spec"))]
