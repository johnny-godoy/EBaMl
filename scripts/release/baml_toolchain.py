"""Print the version of the BAML toolchain that CI installs: the version that ``baml-bridge`` expects.

``baml generate`` writes ``baml_sdk`` for one toolchain, and ``baml-bridge`` (the Python runtime that ``baml_sdk``
imports) works with the SDK of the same toolchain only. The installer of BAML installs the newest toolchain unless it
gets a version. A newer toolchain than the one that ``uv.lock`` pins for ``baml-bridge`` then writes an SDK that does
not match: the toolchain 0.25 writes ``async`` functions and a type map that ``baml-bridge`` 0.23 cannot read. So the
setup action of CI runs ``python3 scripts/release/baml_toolchain.py`` and passes the output to the installer as
``--version``.
A Dependabot update of ``baml-bridge`` then moves the toolchain of CI with it. Run it from the repo root.
"""

import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

LOCK_FILE: Path = Path("uv.lock")
BRIDGE_PACKAGE: str = "baml-bridge"


def locked_bridge_version(lock_text: str) -> str:
    """Read the version of ``baml-bridge`` that a ``uv.lock`` pins.

    Parameters
    ----------
    lock_text: str
        The text of ``uv.lock``.

    Returns
    -------
    str
        The version, for example ``0.23.0``.

    Raises
    ------
    LookupError
        If the lock file has no entry for ``baml-bridge``.

    """
    for package in tomllib.loads(lock_text)["package"]:
        if package["name"] == BRIDGE_PACKAGE:
            return str(package["version"])
    msg = f"{LOCK_FILE} pins no version of {BRIDGE_PACKAGE}."
    raise LookupError(msg)


def main(argv: Sequence[str]) -> int:
    """Print the toolchain version of the lock file.

    Parameters
    ----------
    argv: Sequence[str]
        The command line arguments: none, or the path of the lock file.

    Returns
    -------
    int
        The process exit code: 0 if the version was found, 1 otherwise.

    """
    path = Path(argv[0]) if argv else LOCK_FILE
    try:
        version = locked_bridge_version(path.read_text(encoding="utf-8"))
    except LookupError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(f"{version}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
