"""Tests for the beartype import hook module."""

import ast
from pathlib import Path

import ebaml
from ebaml.core import beartype_hook


def test_package_init_imports_the_hook_module_before_any_other_ebaml_module() -> None:
    # A module that loads before the hook stays unchecked by beartype, so the order is the contract.
    tree = ast.parse(Path(ebaml.__file__).read_text(encoding="utf-8"))
    imported_modules = [
        node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    first_ebaml_module = next(name for name in imported_modules if name and name.startswith("ebaml"))
    assert first_ebaml_module == beartype_hook.__name__
