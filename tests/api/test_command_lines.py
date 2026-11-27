"""The commands of ``ebaml``: the flags of each, the entry points of the package, and how a wrong call fails."""

import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import orjson
import pytest
from benchmark.run import KINDS
from benchmark.run import main as benchmark_main
from scripts.fixed_command import run_fixed

import ebaml
from ebaml.mcp import chat, server

ROOT = Path(__file__).resolve().parents[2]
# Each command with its entry point, and its flags. A new flag needs a row here and a line in the docs.
COMMANDS: dict[str, tuple[str, set[str]]] = {
    "ebaml-mcp": (
        "ebaml.mcp.server",
        {"--model", "--data", "--llm", "--transport", "--host", "--port", "--output-dir", "--edits", "--labels"},
    ),
    "ebaml-chat": ("ebaml.mcp.chat", {"--model", "--data", "--llm", "--max-steps"}),
}
# literal imports: the module of a command is a fixed name, so no test loads a module by a computed name
MODULES: dict[str, ModuleType] = {"ebaml-mcp": server, "ebaml-chat": chat}
REQUIRED: dict[str, set[str]] = {"ebaml-mcp": {"--model"}, "ebaml-chat": {"--model"}}


def _flags(help_text: str) -> set[str]:
    return set(re.findall(r"--[a-z][a-z-]*", help_text)) - {"--help", "--version", "--no-progress"}


def test_the_project_declares_exactly_the_commands_of_the_table() -> None:
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert set(declared) == set(COMMANDS)
    for name, (module, _flags_of) in COMMANDS.items():
        assert declared[name] == f"{module}:main"


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_the_entry_point_of_each_command_is_a_callable_that_takes_the_arguments(name: str) -> None:
    module = MODULES[name]
    assert callable(module.main)


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_the_help_of_each_command_lists_its_flags_and_marks_the_required_ones(
    name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = MODULES[name]
    module.main(["--help"])
    help_text = capsys.readouterr().out
    assert _flags(help_text) == COMMANDS[name][1]
    for flag in REQUIRED[name]:
        assert re.search(rf"\*\s+{flag}\b", help_text), f"{flag} of {name} must be marked as required"


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_a_call_without_the_model_fails_with_a_usage_error_and_not_a_traceback(
    name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = MODULES[name]
    with pytest.raises(SystemExit) as stopped:
        module.main([])
    assert stopped.value.code != 0
    captured = capsys.readouterr()
    assert "--model" in captured.err + captured.out
    assert "Traceback" not in captured.err


# Each kind of the benchmark command, with its flags. A new kind or flag needs a row here and a line in its README.
BENCHMARK_FLAGS: dict[str, set[str]] = {
    "accuracy": {"--task", "--limit", "--model", "--mode", "--output", "--progress"},
    "performance": {"--output", "--workload", "--models-dir", "--progress"},
    "performance compare": {"--threshold", "--floor-seconds", "--markdown", "--gate"},
}


def test_the_benchmark_command_has_one_subcommand_for_each_kind_of_the_table() -> None:
    assert {command.split()[0] for command in BENCHMARK_FLAGS} == set(KINDS)


@pytest.mark.parametrize("command", sorted(BENCHMARK_FLAGS))
def test_each_subcommand_of_the_benchmark_lists_its_flags(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_main([*command.split(), "--help"])
    assert _flags(capsys.readouterr().out) == BENCHMARK_FLAGS[command]


def test_the_performance_benchmark_runs_by_path_and_writes_the_summary_of_its_workloads(tmp_path: Path) -> None:
    output = tmp_path / "performance.json"
    result = run_fixed(
        [
            sys.executable,
            ROOT / "benchmark" / "run.py",
            "performance",
            "--workload",
            "local.facts.diabetes",
            "--models-dir",
            tmp_path / "models",
            "--output",
            output,
            "--no-progress",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = orjson.loads(output.read_bytes())
    (timing,) = summary["workloads"]
    assert timing["name"] == "local.facts.diabetes"
    assert timing["median_seconds"] > 0


def test_the_chat_and_server_commands_print_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    for name, module in MODULES.items():
        module.main(["--version"])
        assert ebaml.__version__ in capsys.readouterr().out, name
