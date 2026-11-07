"""Tests for the check that the toolchain, ``baml-bridge`` and the lockfile name one version."""

from pathlib import Path

import pytest
from scripts import check_toolchain_pin

BAML_TOML = '[package]\nname = "x"\n\n[toolchain]\nversion = "0.23.0"\n'
PYPROJECT = '[project]\ndependencies = ["attrs>=26", "baml-bridge==0.23.0", "numpy>=2"]\n'
LOCKFILE = '[[package]]\nname = "attrs"\nversion = "26.1.0"\n\n[[package]]\nname = "baml-bridge"\nversion = "0.23.0"\n'


def test_toolchain_version_reads_the_pin() -> None:
    assert check_toolchain_pin.toolchain_version(BAML_TOML) == "0.23.0"


@pytest.mark.parametrize(
    "text",
    ['[package]\nname = "x"\n', "[toolchain]\nchannel = 1\n", "[toolchain]\nversion = 5\n"],
)
def test_toolchain_version_is_none_without_a_string_pin(text: str) -> None:
    assert check_toolchain_pin.toolchain_version(text) is None


def test_required_bridge_version_reads_the_exact_requirement() -> None:
    assert check_toolchain_pin.required_bridge_version(PYPROJECT) == "0.23.0"


@pytest.mark.parametrize("requirement", ["baml-bridge>=0.23.0", "baml-bridge>=0.23.0,<0.24", "baml-bridges==1.0"])
def test_required_bridge_version_ignores_a_range(requirement: str) -> None:
    assert check_toolchain_pin.required_bridge_version(f'[project]\ndependencies = ["{requirement}"]\n') is None


def test_locked_bridge_version_reads_the_package() -> None:
    assert check_toolchain_pin.locked_bridge_version(LOCKFILE) == "0.23.0"


def test_locked_bridge_version_is_none_without_the_package() -> None:
    assert check_toolchain_pin.locked_bridge_version('[[package]]\nname = "attrs"\nversion = "1"\n') is None


def test_equal_versions_give_no_problem() -> None:
    assert check_toolchain_pin.pin_problems(baml_toml=BAML_TOML, pyproject=PYPROJECT, lockfile=LOCKFILE) == []


def test_a_range_instead_of_an_exact_requirement_lets_a_user_install_a_skewed_bridge() -> None:
    pyproject = PYPROJECT.replace("==0.23.0", ">=0.23.0")
    [problem] = check_toolchain_pin.pin_problems(baml_toml=BAML_TOML, pyproject=pyproject, lockfile=LOCKFILE)
    assert problem == "pyproject.toml baml-bridge==<version> is missing"


def test_a_missing_toolchain_pin_and_a_missing_lock_entry_are_reported() -> None:
    problems = check_toolchain_pin.pin_problems(baml_toml="", pyproject=PYPROJECT, lockfile="package = []\n")
    assert problems == ["baml.toml [toolchain] version is missing", "uv.lock baml-bridge is missing"]


def test_different_versions_are_reported_with_each_place() -> None:
    [problem] = check_toolchain_pin.pin_problems(
        baml_toml=BAML_TOML.replace("0.23.0", "0.25.2"),
        pyproject=PYPROJECT,
        lockfile=LOCKFILE,
    )
    assert problem == (
        "the toolchain versions differ: baml.toml [toolchain] version = 0.25.2, "
        "pyproject.toml baml-bridge==<version> = 0.23.0, uv.lock baml-bridge = 0.23.0"
    )


def test_main_passes_on_this_repository(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_toolchain_pin.main() == 0
    assert not capsys.readouterr().err


def test_main_reports_a_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "baml.toml").write_text(BAML_TOML.replace("0.23.0", "0.25.2"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(LOCKFILE, encoding="utf-8")
    monkeypatch.setattr(check_toolchain_pin, "ROOT", tmp_path)
    assert check_toolchain_pin.main() == 1
    assert "the toolchain versions differ" in capsys.readouterr().err
