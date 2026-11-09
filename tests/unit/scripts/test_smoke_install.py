"""Tests for the clean-environment smoke test of a distribution."""

import tomllib
from collections.abc import Sequence
from pathlib import Path

import attrs
import pytest
from scripts import smoke_install
from scripts.fixed_command import CommandResult


@attrs.define
class Call:
    """One recorded call of the fake ``run_fixed``."""

    command: list[str]
    env: dict[str, str] | None
    cwd: Path


@attrs.define
class FakeRun:
    """Record each command and give an exit code for each."""

    calls: list[Call] = attrs.field(factory=list)
    exit_codes: list[int] = attrs.field(factory=list)

    def __call__(self, command: Sequence[str], *, cwd: Path, env: dict[str, str] | None) -> CommandResult:
        """Record the call and report the next exit code, or 0.

        Returns
        -------
        CommandResult
            A finished command with the exit code, some output and a reason.

        """
        self.calls.append(Call(list(command), env, cwd))
        code = self.exit_codes.pop(0) if self.exit_codes else 0
        return CommandResult(returncode=code, stdout="out\n", stderr="the reason")


def _uv_path(_name: str) -> str | None:
    return "/bin/uv"


def _no_path(_name: str) -> str | None:
    return None


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    run = FakeRun()
    monkeypatch.setattr(smoke_install, "run_fixed", run)
    monkeypatch.setattr(smoke_install.shutil, "which", _uv_path)
    return run


def test_smoke_program_is_valid_python() -> None:
    compile(smoke_install.SMOKE_PROGRAM, "smoke program", "exec")


def test_clean_environment_has_no_baml_and_no_key(tmp_path: Path) -> None:
    assert smoke_install.clean_environment(tmp_path) == {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
    }


def test_install_command_passes_a_requirement_as_it_is(tmp_path: Path) -> None:
    command = smoke_install.install_command("/bin/uv", tmp_path / "venv", "ebaml==1.2.3", [])
    assert command == [
        "/bin/uv",
        "pip",
        "install",
        "--refresh",
        "--resolution",
        "highest",
        "--python",
        str(tmp_path / "venv" / "bin" / "python"),
        "ebaml==1.2.3",
    ]


def test_install_command_adds_the_extras_to_a_requirement(tmp_path: Path) -> None:
    command = smoke_install.install_command("/bin/uv", tmp_path, "ebaml==1.2.3", ["mcp", "other"])
    assert command[-1] == "ebaml[mcp,other]==1.2.3"


def test_install_command_adds_the_extras_to_a_file(tmp_path: Path) -> None:
    wheel = tmp_path / "ebaml-1.2.3-py3-none-any.whl"
    wheel.touch()
    command = smoke_install.install_command("/bin/uv", tmp_path, str(wheel), ["mcp"])
    assert command[-1] == f"ebaml[mcp] @ {wheel.resolve().as_uri()}"


def test_install_command_installs_a_file_without_extras(tmp_path: Path) -> None:
    wheel = tmp_path / "ebaml-1.2.3-py3-none-any.whl"
    wheel.touch()
    assert smoke_install.install_command("/bin/uv", tmp_path, str(wheel), [])[-1] == str(wheel)


def test_the_steps_run_in_order_and_the_smoke_test_gets_a_clean_environment(fake_run: FakeRun) -> None:
    assert smoke_install.main(["ebaml==1.2.3", "1.2.3"]) == 0
    create, install, smoke = fake_run.calls
    assert create.command[:4] == ["/bin/uv", "venv", "--python", "3.13"]
    assert install.command[:3] == ["/bin/uv", "pip", "install"]
    assert smoke.command[1:3] == ["-I", "-c"]
    assert smoke.command[-1] == "1.2.3"
    assert smoke.env is not None
    assert smoke.env["PATH"] == "/usr/bin:/bin"
    assert smoke.cwd == Path(smoke.env["HOME"])
    assert create.env is None
    assert install.env is None


def test_a_failed_step_stops_with_its_output(fake_run: FakeRun) -> None:
    fake_run.exit_codes = [0, 1]
    with pytest.raises(SystemExit) as stopped:
        smoke_install.main(["ebaml==1.2.3", "1.2.3"])
    assert "exit code 1" in str(stopped.value)
    assert "the reason" in str(stopped.value)
    assert len(fake_run.calls) == 2


def test_missing_uv_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke_install.shutil, "which", _no_path)
    with pytest.raises(SystemExit, match="uv command is not on the path"):
        smoke_install.main(["ebaml==1.2.3", "1.2.3"])


def test_extras_reach_the_install_step(fake_run: FakeRun) -> None:
    smoke_install.main(["ebaml==1.2.3", "1.2.3", "--extras", "mcp"])
    install, smoke = fake_run.calls[1:]
    assert install.command[-1] == "ebaml[mcp]==1.2.3"
    assert smoke.command[-2:] == ["1.2.3", "mcp"]


def test_mcp_extra_requires_mcp_2() -> None:
    """`ebaml-mcp` imports `MCPServer`, which mcp 1.x lacks, so the extra must not allow mcp 1.x."""
    pyproject = tomllib.loads((Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(encoding="utf-8"))
    [requirement] = [item for item in pyproject["project"]["optional-dependencies"]["mcp"] if item.startswith("mcp>")]
    assert requirement.startswith("mcp>=2.")


def test_the_lowest_resolution_reaches_the_install_step(fake_run: FakeRun) -> None:
    smoke_install.main(["ebaml==1.2.3", "1.2.3", "--resolution", "lowest-direct"])
    assert fake_run.calls[1].command[4:6] == ["--resolution", "lowest-direct"]
