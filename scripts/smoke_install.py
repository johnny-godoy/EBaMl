"""Install a distribution into a clean virtual environment and run a smoke test of the installed package.

A green test suite of the source checkout does not show that a user can install the release. This script installs a
wheel, a source distribution or a published version in a new environment, and it runs the smoke test with no source
checkout and no ``baml`` command on the path. The release workflow runs it on the built files before the upload, and on
the published version after the upload.

The smoke test checks that the installed version is the expected one, that ``ebaml`` comes from the environment and
not from a checkout, that the generated ``baml_sdk`` imports, and that the graph of a small EBM converts to text and
the text parses. With the ``mcp`` extra, it also runs ``ebaml-mcp --help``. It sends no request to an LLM provider.

Run it from the repo root: ``uv run python -m scripts.smoke_install dist/ebaml-1.2.3-py3-none-any.whl 1.2.3``.
"""

import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the calls run uv and the new interpreter with fixed argv
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import cyclopts

SMOKE_PROGRAM = """
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier

import ebaml
import baml_sdk
from ebaml import EBMGraph, describe_graph

expected_version = sys.argv[1]
extras = sys.argv[2:]
installed_version = importlib.metadata.version("ebaml")
assert installed_version == expected_version, f"installed {installed_version}, expected {expected_version}"
assert "site-packages" in ebaml.__file__, f"ebaml was not imported from the environment: {ebaml.__file__}"
assert "site-packages" in baml_sdk.__file__, f"baml_sdk was not imported from the environment: {baml_sdk.__file__}"
assert shutil.which("baml") is None, "the baml command is on the path, so the test does not show that it is not needed"
assert callable(describe_graph)

random_generator = np.random.default_rng(0)
features = random_generator.normal(size=(300, 2))
labels = (features[:, 0] + 0.3 * features[:, 1] + random_generator.normal(scale=0.5, size=300) > 0).astype(int)
ebm = ExplainableBoostingClassifier(interactions=0, feature_names=["first", "second"]).fit(features, labels)
graph = EBMGraph.from_ebm(ebm, feature_index=0)
graph_text = graph.to_text()
assert graph_text.strip(), "the text of the graph is empty"
EBMGraph.from_text(graph_text)
if "mcp" in extras:
    server = Path(sys.executable).parent / "ebaml-mcp"
    result = subprocess.run([server, "--help"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"ebaml-mcp --help exited with {result.returncode}: {result.stderr[-800:]}"
    assert "ebaml" in result.stdout.lower(), "the ebaml-mcp command printed no help"
print(f"smoke test passed: ebaml {installed_version}")
"""


def clean_environment(home: Path) -> dict[str, str]:
    """Build the environment of the smoke test: no ``baml`` command, no key, and an empty home directory.

    Parameters
    ----------
    home : Path
        An empty directory to use as the home directory.

    Returns
    -------
    dict[str, str]
        The environment variables for the new interpreter.

    """
    return {"HOME": str(home), "PATH": os.pathsep.join(["/usr/bin", "/bin"]), "PYTHONNOUSERSITE": "1"}


def install_command(
    uv: str,
    environment: Path,
    requirement: str,
    extras: Sequence[str],
    resolution: Literal["highest", "lowest-direct"] = "highest",
) -> list[str]:
    """Write the command that installs the distribution into the environment.

    Parameters
    ----------
    uv : str
        The path of the ``uv`` command.
    environment : Path
        The virtual environment.
    requirement : str
        A path of a wheel or a source distribution, or a requirement such as ``ebaml==1.2.3``.
    extras : Sequence[str]
        The names of the optional dependency groups to install with it.
    resolution : Literal["highest", "lowest-direct"]
        ``"highest"`` installs the newest release of each dependency that the requirements allow. ``"lowest-direct"``
        installs the oldest release that each direct requirement allows, to test the lower bounds.

    Returns
    -------
    list[str]
        The arguments of the command.

    """
    target = requirement
    if extras and Path(requirement).exists():
        target = f"ebaml[{','.join(extras)}] @ {Path(requirement).resolve().as_uri()}"
    elif extras:
        name, separator, version = requirement.partition("==")
        target = f"{name}[{','.join(extras)}]{separator}{version}"
    python = str(environment / "bin" / "python")
    return [uv, "pip", "install", "--refresh", "--resolution", resolution, "--python", python, target]


def _run(command: Sequence[str], *, environment: Mapping[str, str] | None, working_directory: Path) -> None:
    """Run a command and stop with its output if it fails.

    Parameters
    ----------
    command : Sequence[str]
        The command and its arguments.
    environment : Mapping[str, str] | None
        The environment variables, or ``None`` for those of this process.
    working_directory : Path
        The directory to run in. It holds no source of ``ebaml``.

    Raises
    ------
    SystemExit
        If the command fails. The message holds its output.

    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv: the uv path from shutil.which and the paths of this script
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=working_directory,
        env=None if environment is None else dict(environment),
    )
    sys.stdout.write(result.stdout)
    if result.returncode:
        message = f"command failed with exit code {result.returncode}: {' '.join(command[:4])} ...\n{result.stderr}"
        raise SystemExit(message)


app: cyclopts.App = cyclopts.App(
    name="smoke_install",
    help=__doc__.splitlines()[0] if __doc__ else None,
    result_action="return_value",
)


@app.default
def smoke_install(
    requirement: str,
    expected_version: str,
    *,
    extras: Sequence[str] = (),
    resolution: Literal["highest", "lowest-direct"] = "highest",
) -> int:
    """Install a distribution in a clean environment and test the installed package.

    Parameters
    ----------
    requirement : str
        A path of a wheel or a source distribution, or a requirement such as ``ebaml==1.2.3``.
    expected_version : str
        The version that the installed package must report.
    extras : Sequence[str]
        The optional dependency groups to install with the package, for example ``mcp``.
    resolution : Literal["highest", "lowest-direct"]
        Which release of each dependency to install. ``"lowest-direct"`` tests the lower bounds of the requirements.

    Returns
    -------
    int
        0 if the smoke test passes.

    Raises
    ------
    SystemExit
        If ``uv`` is missing or a step fails. The message holds the output of the step.

    """
    uv = shutil.which("uv")
    if uv is None:
        message = "the uv command is not on the path"
        raise SystemExit(message)
    with tempfile.TemporaryDirectory(prefix="ebaml-smoke-") as directory:
        workspace = Path(directory)
        environment = workspace / "venv"
        home = workspace / "home"
        home.mkdir()
        _run([uv, "venv", "--python", "3.13", str(environment)], environment=None, working_directory=workspace)
        _run(
            install_command(uv, environment, requirement, extras, resolution),
            environment=None,
            working_directory=workspace,
        )
        _run(
            [str(environment / "bin" / "python"), "-I", "-c", SMOKE_PROGRAM, expected_version, *extras],
            environment=clean_environment(home),
            working_directory=home,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments of the command line, without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit code of the command.

    """
    return int(app(list(argv) if argv is not None else None))


if __name__ == "__main__":
    sys.exit(main())
