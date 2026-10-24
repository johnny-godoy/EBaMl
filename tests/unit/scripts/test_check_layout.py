"""Tests for the check of the directory layout."""

from pathlib import Path

import pytest
from scripts import check_layout

TOP = {"__init__.py", "benchmark.py"}


def build(root: Path, files: list[str]) -> tuple[Path, Path]:
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Docstring."""\n' if name.endswith("__init__.py") else "x = 1\n", encoding="utf-8")
    return root / "src" / "ebaml", root / "tests" / "unit" / "ebaml"


GOOD = [
    "src/ebaml/__init__.py",
    "src/ebaml/benchmark.py",
    "src/ebaml/core/__init__.py",
    "src/ebaml/core/errors.py",
    "tests/unit/ebaml/test_benchmark.py",
    "tests/unit/ebaml/core/test_errors.py",
]


def problems(tmp_path: Path, files: list[str], layers: tuple[str, ...] = ("core",)) -> list[str]:
    src, tests = build(tmp_path, files)
    return check_layout.layout_problems(src, tests, layers, TOP)


def test_layout_without_problems(tmp_path: Path) -> None:
    assert problems(tmp_path, GOOD) == []


def test_package_that_is_no_layer(tmp_path: Path) -> None:
    found = problems(
        tmp_path,
        [*GOOD, "src/ebaml/misc/__init__.py", "src/ebaml/misc/tool.py", "tests/unit/ebaml/misc/test_tool.py"],
    )
    assert found == ["misc: not a layer in the import-linter layers contract - add it there or remove it"]


def test_layer_without_package(tmp_path: Path) -> None:
    found = problems(tmp_path, GOOD, layers=("core", "llm"))
    assert [message.split(":")[0] for message in found] == ["llm"]


def test_package_without_docstring(tmp_path: Path) -> None:
    src, tests = build(tmp_path, GOOD)
    (src / "core" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    assert check_layout.layout_problems(src, tests, ["core"], TOP) == [
        "core/__init__.py: missing, or it has no docstring",
    ]


def test_package_without_init(tmp_path: Path) -> None:
    src, tests = build(tmp_path, GOOD)
    (src / "core" / "__init__.py").unlink()
    assert check_layout.layout_problems(src, tests, ["core"], TOP) == [
        "core/__init__.py: missing, or it has no docstring",
    ]


def test_module_at_top_level(tmp_path: Path) -> None:
    found = problems(tmp_path, [*GOOD, "src/ebaml/extra.py", "tests/unit/ebaml/test_extra.py"])
    assert found == ["extra.py: a new module goes into a layer package, not directly into ebaml"]


def test_module_name_twice(tmp_path: Path) -> None:
    files = [*GOOD, "src/ebaml/llm/__init__.py", "src/ebaml/llm/errors.py", "tests/unit/ebaml/llm/test_errors.py"]
    found = problems(tmp_path, files, layers=("core", "llm"))
    assert sorted(message.split(":")[0] for message in found) == ["core/errors.py", "llm/errors.py"]


def test_module_without_test(tmp_path: Path) -> None:
    (tmp_path / "tests" / "unit" / "ebaml" / "core").mkdir(parents=True)
    found = problems(tmp_path, [file for file in GOOD if "test_errors" not in file])
    assert found == ["core/errors.py: no test file ebaml/core/test_errors.py"]


def test_test_file_without_module(tmp_path: Path) -> None:
    found = problems(tmp_path, [*GOOD, "tests/unit/ebaml/core/test_stray.py"])
    assert found == ["ebaml/core/test_stray.py: no module ebaml/core/stray.py"]


def test_test_directory_without_package(tmp_path: Path) -> None:
    found = problems(tmp_path, [*GOOD, "tests/unit/ebaml/other/__init__.py"])
    assert found == ["ebaml/other: no package other in ebaml"]


def test_pycache_is_no_package(tmp_path: Path) -> None:
    src, _ = build(tmp_path, GOOD)
    (src / "__pycache__").mkdir()
    assert problems(tmp_path, GOOD) == []


def test_layer_names_read_the_layers_contract() -> None:
    pyproject = {
        "tool": {
            "importlinter": {
                "contracts": [
                    {"type": "layers", "layers": ["ebaml.llm", "ebaml.core"]},
                    {"type": "forbidden", "layers": ["ebaml.other"]},
                ],
            },
        },
    }
    assert check_layout.layer_names(pyproject) == ["llm", "core"]
    assert check_layout.layer_names({}) == []


@pytest.mark.parametrize(("files", "code"), [(GOOD, 0), (GOOD[:-1], 1)])
def test_main(tmp_path: Path, files: list[str], code: int, capsys: pytest.CaptureFixture[str]) -> None:
    build(tmp_path, files)
    (tmp_path / "pyproject.toml").write_text(
        '[[tool.importlinter.contracts]]\ntype = "layers"\nlayers = ["ebaml.core"]\n',
        encoding="utf-8",
    )
    assert check_layout.main([str(tmp_path)]) == code
    assert bool(capsys.readouterr().err) == bool(code)


def test_main_checks_this_repository() -> None:
    assert check_layout.main() == 0
