"""Run a command with a fixed argument list, and return its output.

The scripts of this repository start tools such as ``git``, ``uv``, marimo and Zensical. Each call passes an argument
list that the script builds, and never a shell line. This module holds the only ``subprocess`` import of ``scripts/``,
so the ignores for it stand in one place.
"""

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the one module that starts a command from a fixed argument list
from collections.abc import Mapping, Sequence

import attrs


@attrs.frozen
class CommandResult:
    """The outcome of a command that ended."""

    returncode: int
    stdout: str
    stderr: str


class CommandFailedError(Exception):
    """A command ended with a non-zero exit code, and the caller asked for a check.

    Attributes
    ----------
    result : CommandResult
        The exit code and the output of the command.

    """

    def __init__(self, argv: Sequence[str | os.PathLike[str]], result: CommandResult) -> None:
        """Keep the result and name the command in the message.

        Parameters
        ----------
        argv : Sequence[str | os.PathLike[str]]
            The command that failed.
        result : CommandResult
            Its exit code and output.

        """
        words = [os.fspath(word) for word in argv]
        super().__init__(f"command failed with exit code {result.returncode}: {' '.join(words)}")
        self.result: CommandResult = result


def run_fixed(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = True,
    check: bool = False,
) -> CommandResult:
    """Run a command from a fixed argument list, with no shell.

    Parameters
    ----------
    argv : Sequence[str | os.PathLike[str]]
        The program and its arguments. The caller builds the list, so a value of a user never becomes a shell line.
    cwd : str | os.PathLike[str] | None
        The directory to run in, or ``None`` for the directory of this process.
    env : Mapping[str, str] | None
        The environment variables, or ``None`` for those of this process.
    capture_output : bool
        If ``True``, the result holds the output as text. If ``False``, the command writes to the streams of this
        process, and the output of the result is empty.
    check : bool
        If ``True``, a non-zero exit code raises ``CommandFailedError``.

    Returns
    -------
    CommandResult
        The exit code and the output of the command.

    Raises
    ------
    CommandFailedError
        If ``check`` is ``True`` and the command ends with a non-zero exit code.

    """
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argument list, run without a shell
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=capture_output,
        text=capture_output,
        check=False,
    )
    result = CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout if capture_output else "",
        stderr=completed.stderr if capture_output else "",
    )
    if check and result.returncode:
        raise CommandFailedError(argv, result)
    return result
