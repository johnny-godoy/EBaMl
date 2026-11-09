"""Tests of the ``ebaml-chat`` command, with the call into ``ebaml.llm`` faked."""

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from ebaml.llm.options import ModelOptions
from ebaml.mcp import chat
from ebaml.mcp.chat import main, server_arguments

Call = tuple[str, Sequence[str], ModelOptions | None, int]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Fake ``chat_with_mcp`` and record its arguments.

    Returns
    -------
    list[Call]
        The list that the fake fills with the arguments of each call.

    """
    recorded: list[Call] = []

    def fake_chat_with_mcp(
        command: str,
        arguments: Sequence[str],
        model: ModelOptions | None = None,
        *,
        max_steps: int = 12,
    ) -> int:
        recorded.append((command, arguments, model, max_steps))
        return 2

    monkeypatch.setattr(chat, "chat_with_mcp", fake_chat_with_mcp)
    return recorded


def test_server_arguments_run_the_server_entry_point_with_the_model_and_the_data(tmp_path: Path) -> None:
    arguments = server_arguments(tmp_path / "model.skops", tmp_path / "train.csv")
    assert arguments[:2] == ["-c", "from ebaml.mcp.server import main; main()"]
    assert arguments[2:] == ["--model", str(tmp_path / "model.skops"), "--data", str(tmp_path / "train.csv")]


def test_server_arguments_leave_out_the_data_when_there_is_none(tmp_path: Path) -> None:
    assert server_arguments(tmp_path / "model.skops", None)[2:] == ["--model", str(tmp_path / "model.skops")]


def test_the_chat_starts_the_server_with_the_python_of_the_chat(tmp_path: Path, calls: list[Call]) -> None:
    (tmp_path / "model.skops").write_bytes(b"")
    main(["--model", str(tmp_path / "model.skops")])
    command, arguments, model, max_steps = calls[0]
    assert command == sys.executable
    assert arguments == server_arguments(tmp_path / "model.skops", None)
    assert (model, max_steps) == (None, 12)


def test_the_chat_passes_the_data_the_llm_and_the_step_limit(tmp_path: Path, calls: list[Call]) -> None:
    (tmp_path / "model.skops").write_bytes(b"")
    (tmp_path / "train.csv").write_text("a\n1\n", encoding="utf-8")
    main(
        [
            "--model",
            str(tmp_path / "model.skops"),
            "--data",
            str(tmp_path / "train.csv"),
            "--llm",
            "ollama/llama3",
            "--max-steps",
            "5",
        ],
    )
    _, arguments, model, max_steps = calls[0]
    assert arguments[-2:] == ["--data", str(tmp_path / "train.csv")]
    assert model == ModelOptions(provider="ollama", model="llama3")
    assert max_steps == 5


@pytest.mark.parametrize("missing", ["--model", "--data"])
def test_a_missing_file_is_a_usage_error_and_starts_no_chat(
    missing: str,
    tmp_path: Path,
    calls: list[Call],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug guard: the server failed to start behind BAML, and the user saw an error about the connection only."""
    present = tmp_path / "present.file"
    present.write_bytes(b"")
    gone = tmp_path / "gone.file"
    arguments = {"--model": str(present), "--data": str(present), missing: str(gone)}
    with pytest.raises(SystemExit) as exit_info:
        main([option for pair in arguments.items() for option in pair])
    assert exit_info.value.code != 0
    assert "gone.file" in capsys.readouterr().err
    assert not calls
