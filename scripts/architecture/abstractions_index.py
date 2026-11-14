"""Index the classes and functions of ``src/ebaml`` with ``ast``, without importing the source."""

import ast
from collections.abc import Iterable
from pathlib import Path

import attrs

from scripts.architecture.abstractions_registry import SOURCE_PATH


@attrs.frozen
class Symbol:
    """A public top-level name of a module of ``src/ebaml``.

    Parameters
    ----------
    module: str
        The dotted module, for example ``ebaml.graphs.terms``.
    name: str
        The name in the module.
    path: str
        The file, relative to the repo root.
    line: int
        The line of the definition.
    kind: str
        ``class``, ``function`` or ``alias``.

    """

    module: str
    name: str
    path: str
    line: int
    kind: str

    @property
    def reference(self) -> str:
        """Give the symbol as ``module:Name``.

        Returns
        -------
        str
            The reference as the registry writes it.

        """
        return f"{self.module}:{self.name}"


@attrs.frozen
class SourceIndex:
    """The modules and the public symbols of ``src/ebaml``.

    Parameters
    ----------
    modules: frozenset[str]
        Every module and package, as a dotted name.
    symbols: dict[str, Symbol]
        The public symbols by ``module:Name``.

    """

    modules: frozenset[str]
    symbols: dict[str, Symbol]


def module_name(source: Path, root: Path) -> str:
    """Turn a source path into a dotted module name.

    Parameters
    ----------
    source: Path
        A Python file under ``root / src``.
    root: Path
        The repo root.

    Returns
    -------
    str
        The dotted name. A package ``__init__.py`` gives the name of the package.

    """
    parts = list(source.relative_to(root / "src").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _top_level_symbols(tree: ast.Module, module: str, path: str) -> Iterable[Symbol]:
    """List the public top-level names that a module defines.

    Parameters
    ----------
    tree: ast.Module
        The parsed module.
    module: str
        The dotted name of the module.
    path: str
        The file, relative to the repo root.

    Yields
    ------
    Symbol
        A class, a function or a module-level alias whose name has no leading underscore.

    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found = [(node.name, "class")]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found = [(node.name, "function")]
        elif isinstance(node, ast.Assign):
            found = [(target.id, "alias") for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found = [(node.target.id, "alias")]
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            found = [(node.name.id, "alias")]
        else:
            continue
        for name, kind in found:
            if not name.startswith("_"):
                yield Symbol(module=module, name=name, path=path, line=node.lineno, kind=kind)


def build_index(root: Path) -> SourceIndex:
    """Index the modules and the public symbols of ``src/ebaml`` without importing them.

    Parameters
    ----------
    root: Path
        The repo root.

    Returns
    -------
    SourceIndex
        The modules and symbols. The generated ``src/baml_sdk`` is outside the scan.

    """
    modules: set[str] = set()
    symbols: dict[str, Symbol] = {}
    for source in sorted((root / SOURCE_PATH).rglob("*.py")):
        module = module_name(source, root)
        modules.add(module)
        path = source.relative_to(root).as_posix()
        for symbol in _top_level_symbols(ast.parse(source.read_bytes()), module, path):
            symbols[symbol.reference] = symbol
    return SourceIndex(modules=frozenset(modules), symbols=symbols)
