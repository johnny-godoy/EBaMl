"""Tests for the script that names the BAML toolchain of CI."""

import re
import tomllib
from pathlib import Path

import pytest
from scripts.release import baml_toolchain

ROOT = Path(__file__).parents[4]

LOCK = """\
version = 1

[[package]]
name = "attrs"
version = "26.1.0"

[[package]]
name = "baml-bridge"
version = "0.23.0"
"""


def test_the_version_is_the_one_that_the_lock_pins_for_baml_bridge() -> None:
    assert baml_toolchain.locked_bridge_version(LOCK) == "0.23.0"


def test_a_lock_without_baml_bridge_is_an_error() -> None:
    with pytest.raises(LookupError, match="pins no version of baml-bridge"):
        baml_toolchain.locked_bridge_version('[[package]]\nname = "attrs"\nversion = "1.0"\n')


def test_the_command_prints_the_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(LOCK, encoding="utf-8")
    assert baml_toolchain.main([str(lock)]) == 0
    assert capsys.readouterr().out == "0.23.0\n"


def test_the_command_fails_for_a_lock_without_baml_bridge(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "attrs"\nversion = "1.0"\n', encoding="utf-8")
    assert baml_toolchain.main([str(lock)]) == 1
    assert "pins no version of baml-bridge" in capsys.readouterr().err


def test_the_lock_of_the_repository_pins_baml_bridge() -> None:
    # the setup action of CI reads this lock file, so it must name a plain version that the installer accepts
    version = baml_toolchain.locked_bridge_version((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)


def test_the_setup_action_installs_the_toolchain_of_the_lock() -> None:
    # an installer with no version installs the newest toolchain, which writes an SDK that baml-bridge cannot read
    action = (ROOT / ".github/actions/setup/action.yml").read_text(encoding="utf-8")
    assert "python3 scripts/release/baml_toolchain.py" in action
    assert re.search(r'install\.sh \| sh -s -- --version "\$\{?version\}?"', action)


def test_the_project_requires_the_baml_bridge_that_the_lock_pins() -> None:
    # the toolchain follows the lock, so the lock must stay inside the range that the project declares
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(requirement.startswith("baml-bridge") for requirement in project["dependencies"])
