"""Tests for the helper that runs a command from a fixed argument list."""

import sys
from pathlib import Path

import pytest
from scripts.fixed_command import CommandFailedError, CommandResult, run_fixed

PRINT_AND_FAIL = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"


def test_captures_the_output_and_the_exit_code() -> None:
    result = run_fixed([sys.executable, "-c", PRINT_AND_FAIL])
    assert result == CommandResult(returncode=3, stdout="out\n", stderr="err\n")


def test_a_failed_command_raises_with_its_result_when_checked() -> None:
    with pytest.raises(CommandFailedError, match="exit code 3") as raised:
        run_fixed([sys.executable, "-c", PRINT_AND_FAIL], check=True)
    assert raised.value.result.stderr == "err\n"


def test_a_successful_command_passes_the_check() -> None:
    assert run_fixed([sys.executable, "-c", "pass"], check=True).returncode == 0


def test_a_command_without_capture_returns_empty_output(capfd: pytest.CaptureFixture[str]) -> None:
    result = run_fixed([sys.executable, "-c", "print('shown')"], capture_output=False)
    assert result == CommandResult(returncode=0, stdout="", stderr="")
    assert capfd.readouterr().out == "shown\n"


def test_runs_in_the_given_directory_with_the_given_environment(tmp_path: Path) -> None:
    code = "import os; print(os.getcwd()); print(os.environ['FIXED_COMMAND_TEST'])"
    result = run_fixed([sys.executable, "-c", code], cwd=tmp_path, env={"FIXED_COMMAND_TEST": "set"})
    assert result.stdout.split() == [str(tmp_path.resolve()), "set"]
