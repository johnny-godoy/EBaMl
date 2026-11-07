"""Fail unless the BAML toolchain, ``baml-bridge`` and the lockfile name one version.

``baml generate`` writes the ``baml_sdk`` package that the wheel ships. The package records the toolchain that wrote it,
and ``baml-bridge`` refuses to start with a package from another toolchain (a version skew). A user installs the wheel
with the newest ``baml-bridge`` that the requirement allows, so the requirement must name the version of the toolchain
that generated the package. Three places hold that version:

- ``[toolchain] version`` in ``baml.toml`` selects the toolchain for ``baml generate``,
- ``baml-bridge==<version>`` in ``pyproject.toml`` selects the bridge that a user installs,
- the ``baml-bridge`` entry of ``uv.lock`` selects the bridge of the development environment.

Run it from the repo root: ``uv run python -m scripts.check_toolchain_pin``.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = "baml-bridge"
_REQUIREMENT = re.compile(rf"^{BRIDGE}\s*==\s*(?P<version>[^\s;,]+)$")


def toolchain_version(baml_toml: str) -> str | None:
    """Read the pinned toolchain version of ``baml.toml``.

    Parameters
    ----------
    baml_toml : str
        The text of ``baml.toml``.

    Returns
    -------
    str | None
        The version, or ``None`` if the file pins no toolchain.

    """
    version = tomllib.loads(baml_toml).get("toolchain", {}).get("version")
    return version if isinstance(version, str) else None


def required_bridge_version(pyproject: str) -> str | None:
    """Read the exact ``baml-bridge`` version that ``pyproject.toml`` requires.

    Parameters
    ----------
    pyproject : str
        The text of ``pyproject.toml``.

    Returns
    -------
    str | None
        The version of the requirement ``baml-bridge==<version>``, or ``None`` if there is no such requirement.

    """
    for requirement in tomllib.loads(pyproject)["project"]["dependencies"]:
        match = _REQUIREMENT.match(requirement)
        if match:
            return match["version"]
    return None


def locked_bridge_version(lockfile: str) -> str | None:
    """Read the ``baml-bridge`` version of ``uv.lock``.

    Parameters
    ----------
    lockfile : str
        The text of ``uv.lock``.

    Returns
    -------
    str | None
        The locked version, or ``None`` if the lockfile has no ``baml-bridge`` package.

    """
    for package in tomllib.loads(lockfile)["package"]:
        if package["name"] == BRIDGE:
            return str(package["version"])
    return None


def pin_problems(*, baml_toml: str, pyproject: str, lockfile: str) -> list[str]:
    """List the places that do not name the same toolchain version.

    Parameters
    ----------
    baml_toml : str
        The text of ``baml.toml``.
    pyproject : str
        The text of ``pyproject.toml``.
    lockfile : str
        The text of ``uv.lock``.

    Returns
    -------
    list[str]
        One message for each missing or different version. The list is empty if all three agree.

    """
    versions = {
        "baml.toml [toolchain] version": toolchain_version(baml_toml),
        f"pyproject.toml {BRIDGE}==<version>": required_bridge_version(pyproject),
        f"uv.lock {BRIDGE}": locked_bridge_version(lockfile),
    }
    problems = [f"{place} is missing" for place, version in versions.items() if version is None]
    named = {version for version in versions.values() if version is not None}
    if len(named) > 1:
        listing = ", ".join(f"{place} = {version}" for place, version in versions.items())
        problems.append(f"the toolchain versions differ: {listing}")
    return problems


def main() -> int:
    """Report every place that does not name the toolchain version of the others.

    Returns
    -------
    int
        The process exit code: 0 if the three versions agree, 1 otherwise.

    """
    problems = pin_problems(
        baml_toml=(ROOT / "baml.toml").read_text(encoding="utf-8"),
        pyproject=(ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        lockfile=(ROOT / "uv.lock").read_text(encoding="utf-8"),
    )
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
