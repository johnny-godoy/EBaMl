"""What the package promises to an installer: the files it ships, its extras and the imports that it avoids."""

import importlib.metadata
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from scripts.fixed_command import run_fixed

import ebaml

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT: dict[str, Any] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
# The packages of the extra `mcp`.
EXTRA_PACKAGES = ("mcp", "cyclopts", "joblib", "polars", "pydantic")
# A plain `import ebaml` must not load these, because a user without the extra has none. joblib and pydantic are left
# out: scikit-learn and the BAML SDK need them, so they are present and loaded without the extra.
SERVER_ONLY_PACKAGES = ("mcp", "cyclopts", "polars")


def test_the_version_of_the_installed_metadata_is_the_version_of_the_project_file() -> None:
    assert ebaml.__version__ == PYPROJECT["project"]["version"] == importlib.metadata.version("ebaml")


def test_the_package_is_typed() -> None:
    assert (Path(ebaml.__file__).parent / "py.typed").is_file()


def test_the_wheel_ships_the_generated_sdk_and_the_bundled_token_weights() -> None:
    assert PYPROJECT["tool"]["uv"]["build-backend"]["module-name"] == ["ebaml", "baml_sdk"]
    assert (Path(ebaml.__file__).parent / "llm" / "token_weights.json").is_file()


def test_the_extra_for_the_server_names_the_packages_that_only_the_server_imports() -> None:
    extra = PYPROJECT["project"]["optional-dependencies"]["mcp"]
    names = {requirement.split(">")[0].split("=")[0].strip() for requirement in extra}
    assert names == set(EXTRA_PACKAGES)


@pytest.mark.parametrize("package", SERVER_ONLY_PACKAGES)
def test_importing_ebaml_and_loading_the_llm_functions_does_not_load_a_package_of_the_extra(package: str) -> None:
    code = f"import sys, ebaml; ebaml.describe_ebm; ebaml.explain_instance; print({package!r} in sys.modules)"
    result = run_fixed([sys.executable, "-c", code], check=True)
    assert result.stdout.split() == ["False"]


def test_every_runtime_dependency_has_a_lower_bound_and_the_bridge_an_exact_pin() -> None:
    for requirement in PYPROJECT["project"]["dependencies"]:
        assert any(sign in requirement for sign in (">=", "==")), f"{requirement} has no lower bound"
    assert any(requirement.startswith("baml-bridge==") for requirement in PYPROJECT["project"]["dependencies"])


def test_the_scripts_of_the_project_are_the_commands_that_the_docs_name() -> None:
    documented = (ROOT / "README.md").read_text(encoding="utf-8") + (ROOT / "docs" / "index.md").read_text(
        encoding="utf-8",
    )
    for name in PYPROJECT["project"]["scripts"]:
        assert name in documented or name in "".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "docs").glob("*.md")
        )


def test_the_python_requirement_is_the_version_that_the_classifier_names() -> None:
    assert PYPROJECT["project"]["requires-python"] == ">=3.13"
    assert "Programming Language :: Python :: 3.13" in PYPROJECT["project"]["classifiers"]
    assert sys.version_info >= (3, 13)
