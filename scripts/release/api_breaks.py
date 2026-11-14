"""Fail when a pull request breaks the public API and no changelog fragment says so.

The public API of ``ebaml`` is the names that ``ebaml.__all__`` lists. This check loads the package at a base revision
and at the working tree with griffe, and it lists each change that can break the code of a user: a removed name, a
removed or renamed parameter, a changed default, a changed type and so on. Before 1.0.0 such a change is allowed, but a
reader of the changelog must see it, so the check passes only if a fragment in ``changelog.d/`` has an entry that starts
with ``**BREAKING:**``.

Run it from the repo root with a base revision: ``uv run python -m scripts.release.api_breaks --base origin/main``.
"""

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import cyclopts
import griffe

from scripts.release.changelog import read_fragments
from scripts.release.version import BREAKING_MARK

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ebaml"
SEARCH_PATHS = [ROOT / "src"]
FRAGMENTS = ROOT / "changelog.d"


def as_module(loaded: griffe.Object | griffe.Alias) -> griffe.Module:
    """Narrow what griffe loads for a package name to a module.

    Parameters
    ----------
    loaded : griffe.Object | griffe.Alias
        The result of ``griffe.load`` or ``griffe.load_git``.

    Returns
    -------
    griffe.Module
        The same object.

    Raises
    ------
    TypeError
        If the name is not a module, for example if a search path holds another kind of object of that name.

    """
    if not isinstance(loaded, griffe.Module):
        message = f"{PACKAGE} is not a module, but {type(loaded).__name__}"
        raise TypeError(message)
    return loaded


def exported_paths(package: griffe.Module) -> dict[str, str]:
    """Map each exported name of a package to the path of the object that defines it.

    Parameters
    ----------
    package : griffe.Module
        The loaded package.

    Returns
    -------
    dict[str, str]
        The path of the defining object for each name in ``__all__``. A name that cannot be resolved is left out.

    """
    paths: dict[str, str] = {}
    for export in package.exports or []:
        member = package.members.get(str(export))
        if member is not None:
            paths[str(export)] = member.final_target.path if isinstance(member, griffe.Alias) else member.path
    return paths


def api_breaks(old: griffe.Module, new: griffe.Module) -> list[str]:
    """List the breaking changes of the public API between two revisions of a package.

    Parameters
    ----------
    old : griffe.Module
        The package at the base revision.
    new : griffe.Module
        The package at the working tree.

    Returns
    -------
    list[str]
        One line for each name that left ``__all__`` and each breaking change of an exported object.

    """
    old_paths = exported_paths(old)
    new_names = {str(export) for export in new.exports or []}
    breaks = [f"{old.name}.{name}: removed from __all__" for name in old_paths if name not in new_names]
    public = tuple(old_paths.values())
    for change in griffe.find_breaking_changes(old, new):
        path = change.obj.path
        if any(path == prefix or path.startswith(f"{prefix}.") for prefix in public):
            breaks.append(f"{path}: {change.kind.value}")
    return sorted(set(breaks))


def has_breaking_entry(fragments: Iterable[str]) -> bool:
    """Tell whether a changelog fragment has a breaking entry.

    Parameters
    ----------
    fragments : Iterable[str]
        The text of each pending fragment.

    Returns
    -------
    bool
        ``True`` if an entry starts with ``**BREAKING:**``.

    """
    return any(BREAKING_MARK.search(fragment) for fragment in fragments)


app: cyclopts.App = cyclopts.App(
    name="check_api_breaks",
    help=__doc__.splitlines()[0] if __doc__ else None,
    result_action="return_value",
)


@app.default
def check_api_breaks(*, base: str) -> int:
    """Compare the public API with a base revision.

    Parameters
    ----------
    base : str
        The git revision to compare with, for example ``origin/main``.

    Returns
    -------
    int
        0 if the API has no break or a fragment announces it, 1 otherwise.

    """
    old = as_module(griffe.load_git(PACKAGE, ref=base, repo=ROOT, search_paths=["src"]))
    new = as_module(griffe.load(PACKAGE, search_paths=SEARCH_PATHS))
    breaks = api_breaks(old, new)
    if not breaks:
        return 0
    announced = has_breaking_entry(read_fragments(FRAGMENTS).values())
    listing = "\n".join(f"  {line}" for line in breaks)
    if announced:
        sys.stdout.write(f"The public API breaks against {base}, and a fragment announces it:\n{listing}\n")
        return 0
    sys.stderr.write(
        f"The public API breaks against {base}:\n{listing}\n"
        "Add an entry that starts with **BREAKING:** to a changelog fragment, or restore the API.\n",
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments of the command line, without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit code of the command.

    """
    return int(app(None if argv is None else list(argv)))


if __name__ == "__main__":
    sys.exit(main())
