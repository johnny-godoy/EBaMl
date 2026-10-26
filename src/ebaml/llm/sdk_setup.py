"""Make sure that the generated ``baml_sdk`` package exists, and generate it if it does not.

``baml_sdk`` is gitignored. ``baml generate`` writes it from ``baml_src/``. In a source checkout, ``ensure_sdk`` runs
that command when the package is missing. If it cannot, it raises ``SDKUnavailableError`` with the steps that fix the
cause.
"""

import importlib
import importlib.util
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the only call runs the baml binary with a fixed argument
from pathlib import Path

import structlog

from ebaml.core.exceptions import SDKUnavailableError

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

SDK_NAME = "baml_sdk"
_INSTALL_COMMAND = "curl -fsSL https://pkg.boundaryml.com/install.sh | sh -s"
_GENERATE_TIMEOUT_SECONDS = 300
_OUTPUT_LINES = 15


def _project_root(start: Path) -> Path | None:
    """Find the source checkout that holds ``baml.toml`` and ``baml_src/``.

    Parameters
    ----------
    start : Path
        A file or directory inside the checkout.

    Returns
    -------
    Path | None
        The root of the checkout, or ``None`` if ``start`` is not inside one (an installed package).

    """
    for directory in start.resolve().parents:
        if (directory / "baml.toml").is_file() and (directory / "baml_src").is_dir():
            return directory
    return None


def _find_baml(home: Path) -> str | None:
    """Find the ``baml`` command.

    The installer puts ``baml`` in ``~/.baml/bin``. A shell that did not load the profile line of the installer does not
    have that directory on ``PATH``, so the search looks there too.

    Parameters
    ----------
    home : Path
        The home directory of the user.

    Returns
    -------
    str | None
        The path of the command, or ``None`` if there is none.

    """
    on_path = shutil.which("baml")
    if on_path is not None:
        return on_path
    installed = home / ".baml" / "bin" / "baml"
    return str(installed) if installed.is_file() else None


def _tail(text: str) -> str:
    """Keep the last lines of the output of a command, with an indent.

    Parameters
    ----------
    text : str
        The output.

    Returns
    -------
    str
        The last lines, one indented line each, or ``"  (no output)"`` for an empty output.

    """
    lines = text.strip().splitlines()[-_OUTPUT_LINES:]
    return "\n".join(f"  {line}" for line in lines) or "  (no output)"


def _steps(root: Path | None, *, with_install: bool) -> str:
    """Write the steps that make ``baml_sdk`` importable.

    Parameters
    ----------
    root : Path | None
        The source checkout, or ``None`` for an installed package.
    with_install : bool
        Whether the BAML toolchain is missing, so the steps start with its installation.

    Returns
    -------
    str
        A numbered list of steps.

    """
    if root is None:
        return (
            "This copy of ebaml is installed without its source checkout, and a release must contain baml_sdk.\n"
            "  1. Reinstall ebaml from a release file.\n"
            "  2. To build from source, run `baml generate` in the checkout before `uv build`."
        )
    steps = []
    if with_install:
        steps += [
            f"Install the BAML toolchain: {_INSTALL_COMMAND}",
            'Open a new shell, or run `. "$HOME/.baml/env"`, so that `baml` is on PATH.',
        ]
    steps += [
        "Check that `baml --version` prints 0.23 or later. If it is older, put ~/.baml/bin first on PATH.",
        f"Run `baml generate` in {root}.",
    ]
    return "\n".join(f"  {number}. {step}" for number, step in enumerate(steps, start=1))


def _generate(baml: str, root: Path) -> None:
    """Run ``baml generate`` in the checkout.

    Parameters
    ----------
    baml : str
        The path of the ``baml`` command.
    root : Path
        The source checkout.

    Raises
    ------
    SDKUnavailableError
        If the command cannot start, times out, or exits with an error. The message has the end of its output.

    """
    logger.info("sdk_generate", command=baml, cwd=str(root))
    try:
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- the command is the baml binary that the lookup found, with a fixed argument
            [baml, "generate"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GENERATE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        message = (
            f"{SDK_NAME} is missing, and `{baml} generate` did not finish: {error}.\n"
            f"To fix it:\n{_steps(root, with_install=False)}"
        )
        raise SDKUnavailableError(message) from error
    if done.returncode != 0:
        message = (
            f"{SDK_NAME} is missing, and `{baml} generate` failed with exit code {done.returncode}. Its output ends:\n"
            f"{_tail(done.stderr or done.stdout)}\n"
            f"To fix it:\n{_steps(root, with_install=False)}"
        )
        raise SDKUnavailableError(message)


def ensure_sdk(*, start: Path | None = None, home: Path | None = None) -> None:
    """Make ``baml_sdk`` importable, and run ``baml generate`` if the package is missing.

    The function does nothing when the package exists. Call it before the first ``import baml_sdk``.

    Parameters
    ----------
    start : Path | None
        A path inside the checkout to search from. ``None`` selects this module.
    home : Path | None
        The home directory in which to look for the ``baml`` command. ``None`` selects the home directory of the user.

    Raises
    ------
    SDKUnavailableError
        If the package is missing and cannot be generated. The message lists the steps that fix the cause.

    """
    if importlib.util.find_spec(SDK_NAME) is not None:
        return
    root = _project_root(start or Path(__file__))
    if root is None:
        message = f"{SDK_NAME} is missing.\n{_steps(None, with_install=False)}"
        raise SDKUnavailableError(message)
    baml = _find_baml(home or Path.home())
    if baml is None:
        message = (
            f"{SDK_NAME} is missing, and ebaml cannot generate it because the `baml` command was not found.\n"
            f"To fix it:\n{_steps(root, with_install=True)}"
        )
        raise SDKUnavailableError(message)
    _generate(baml, root)
    importlib.invalidate_caches()
    if importlib.util.find_spec(SDK_NAME) is None:
        message = (
            f"`{baml} generate` finished, but {SDK_NAME} is still not importable. `baml generate` writes it to the "
            f"`output_dir` of {root / 'baml.toml'}, and that directory must be on sys.path.\n"
            f"To fix it, install the project in this environment with `uv sync` in {root}."
        )
        raise SDKUnavailableError(message)
    logger.info("sdk_generated", root=str(root))
