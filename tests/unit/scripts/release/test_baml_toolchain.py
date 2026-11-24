"""Tests for the script that names the BAML toolchain of CI."""

import os
import re
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml
from scripts.fixed_command import CommandResult, run_fixed
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


INSTALL_STEP = "Install BAML toolchain"


def install_script() -> str:
    action = yaml.safe_load((ROOT / ".github/actions/setup/action.yml").read_text(encoding="utf-8"))
    (step,) = [step for step in action["runs"]["steps"] if step.get("name") == INSTALL_STEP]
    return str(step["run"])


def run_install_step(tmp_path: Path, failure_count: int) -> CommandResult:
    """Run the install step of the setup action with an installer that fails ``failure_count`` times first.

    Returns
    -------
    CommandResult
        The exit code and the output of the step.

    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    attempts = tmp_path / "attempts"
    fakes = {
        "curl": "#!/bin/bash\necho 'echo install'\n",
        # the installer counts its attempts and fails as the server error of pkg.boundaryml.com did
        "sh": (
            '#!/bin/bash\ncat > /dev/null\necho x >> "$ATTEMPTS"\n'
            f'[ "$(wc -l < "$ATTEMPTS")" -gt {failure_count} ] || {{ echo "HTTP 520" >&2; exit 7; }}\n'
        ),
        "sleep": "#!/bin/bash\nexit 0\n",
    }
    for name, text in fakes.items():
        fake = fake_bin / name
        fake.write_text(text, encoding="utf-8")
        fake.chmod(0o755)
    environment = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "GITHUB_PATH": str(tmp_path / "github_path"),
        "ATTEMPTS": str(attempts),
    }
    bash = shutil.which("bash") or "bash"
    return run_fixed([bash, "-e", "-o", "pipefail", "-c", install_script()], cwd=ROOT, env=environment)


def test_the_setup_action_retries_a_failed_toolchain_install(tmp_path: Path) -> None:
    # one server error of pkg.boundaryml.com (HTTP 520) failed a whole job of the weekly mutation run
    result = run_install_step(tmp_path, failure_count=2)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "attempts").read_text(encoding="utf-8").count("x") == 3


def test_the_setup_action_fails_when_every_toolchain_install_fails(tmp_path: Path) -> None:
    result = run_install_step(tmp_path, failure_count=99)
    assert result.returncode != 0
    assert "did not install after 4 attempts" in result.stdout
    assert not (tmp_path / "github_path").exists()


def test_the_project_requires_the_baml_bridge_that_the_lock_pins() -> None:
    # the toolchain follows the lock, so the lock must stay inside the range that the project declares
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(requirement.startswith("baml-bridge") for requirement in project["dependencies"])
