"""Tests for the check of the test suites."""

from pathlib import Path
from typing import Any

import pytest
from scripts.architecture import suites

PYPROJECT: dict[str, Any] = {
    "tool": {
        "pytest": {
            "ini_options": {
                "addopts": "--cov=ebaml --strict-markers",
                "markers": ["unit: x", "integration: x", "api: x", "contract: x"],
                "testpaths": ["tests/unit", "tests/integration", "tests/api", "tests/contract"],
            },
        },
    },
}
WORKFLOW = "uv run pytest tests/integration tests/api tests/contract"
DOCSTRING = '"""Docstring."""\n'


def build(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root / "tests"


GOOD = {
    **{f"{suite}/__init__.py": DOCSTRING for suite in suites.KNOWN_SUITES},
    "integration/test_flow.py": "def test_flow(): ...\n",
    "api/test_names.py": "def test_names(): ...\n",
    "contract/test_wire.py": "def test_wire(): ...\n",
}


def test_suites_without_problems(tmp_path: Path) -> None:
    assert suites.suite_problems(build(tmp_path, GOOD)) == []


def test_unknown_suite(tmp_path: Path) -> None:
    found = suites.suite_problems(build(tmp_path, {**GOOD, "smoke/__init__.py": DOCSTRING}))
    assert [message.split(":")[0] for message in found] == ["tests/smoke"]


def test_missing_suite(tmp_path: Path) -> None:
    files = {name: text for name, text in GOOD.items() if not name.startswith("api/")}
    assert suites.suite_problems(build(tmp_path, files)) == ["tests/api: the suite is missing"]


def test_suite_without_docstring(tmp_path: Path) -> None:
    found = suites.suite_problems(build(tmp_path, {**GOOD, "contract/__init__.py": "x = 1\n"}))
    assert found == ["tests/contract/__init__.py: missing, or it has no docstring"]


def test_suite_without_test_module(tmp_path: Path) -> None:
    found = suites.suite_problems(
        build(tmp_path, {**GOOD, "api/test_names.py": ""} | {"api/conftest.py": ""}),
    )
    assert found == []
    files = {name: text for name, text in GOOD.items() if name != "api/test_names.py"}
    assert suites.suite_problems(build(tmp_path / "second", files)) == [
        "tests/api: the suite has no test module",
    ]


def test_helper_module_that_pytest_does_not_collect(tmp_path: Path) -> None:
    found = suites.suite_problems(build(tmp_path, {**GOOD, "api/surface.py": "x = 1\n"}))
    assert len(found) == 1
    assert found[0].startswith("tests/api/surface.py: a module of a test suite is test_*.py")


@pytest.mark.parametrize(
    "line",
    [
        "from tests.unit.ebaml.conftest import fake_sdk\n",
        "def test_x(monkeypatch):\n    monkeypatch.setattr(baml_sdk, 'summarize_ebm', None)\n",
    ],
)
def test_integration_test_that_fakes_the_sdk(tmp_path: Path, line: str) -> None:
    found = suites.suite_problems(build(tmp_path, {**GOOD, "integration/test_flow.py": line}))
    assert len(found) == 1
    assert "replaces a function of baml_sdk with a fake" in found[0]


def test_the_other_suites_may_use_the_fake_of_the_sdk(tmp_path: Path) -> None:
    files = {**GOOD, "contract/test_wire.py": "from tests.unit.ebaml.conftest import fake_sdk\n"}
    assert suites.suite_problems(build(tmp_path, files)) == []


def test_configuration_without_problems() -> None:
    assert suites.configuration_problems(PYPROJECT, WORKFLOW) == []


def test_configuration_without_strict_markers() -> None:
    options = {"addopts": "--cov=ebaml", "markers": PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]}
    found = suites.configuration_problems(
        {"tool": {"pytest": {"ini_options": {**PYPROJECT["tool"]["pytest"]["ini_options"], **options}}}},
        WORKFLOW,
    )
    assert found == ["pyproject.toml: addopts lacks --strict-markers, so a misspelled marker passes"]


def test_configuration_without_marker_and_path() -> None:
    options = {
        "addopts": "--strict-markers",
        "markers": ["unit: x", "api: x", "contract: x"],
        "testpaths": ["tests/unit", "tests/api", "tests/contract"],
    }
    found = suites.configuration_problems({"tool": {"pytest": {"ini_options": options}}}, WORKFLOW)
    assert found == [
        "pyproject.toml: the marker 'integration' is not registered",
        "pyproject.toml: testpaths lacks tests/integration",
    ]


def test_configuration_of_a_project_without_pytest_settings() -> None:
    found = suites.configuration_problems({}, WORKFLOW)
    assert len(found) == 1 + 2 * 4


def test_workflow_that_leaves_a_suite_out() -> None:
    found = suites.configuration_problems(PYPROJECT, "uv run pytest tests/integration tests/api")
    assert found == [".github/workflows/ci.yml: no step runs tests/contract"]


def test_main_reports_the_problems_of_a_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build(tmp_path, {**GOOD, "smoke/__init__.py": DOCSTRING})
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(WORKFLOW, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--strict-markers"\n',
        encoding="utf-8",
    )
    assert suites.main([str(tmp_path)]) == 1
    assert "tests/smoke: not a known suite" in capsys.readouterr().err


def test_main_passes_on_this_repository() -> None:
    assert suites.main() == 0
