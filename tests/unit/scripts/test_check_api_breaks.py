"""Tests for the check that a break of the public API has a changelog fragment that announces it."""

from pathlib import Path

import attrs
import griffe
import pytest
from scripts import check_api_breaks

OLD = """
from pkg.core import Keep, Gone, drop_me
__all__ = ["Keep", "Gone", "drop_me", "stay"]


def stay(first: int, second: int = 1) -> int: ...
"""
NEW = """
from pkg.core import Keep
__all__ = ["Keep", "stay"]


def stay(first: int) -> int: ...
"""
CORE_OLD = "class Keep:\n    pass\n\n\nclass Gone:\n    pass\n\n\ndef drop_me() -> None: ...\n"
CORE_NEW = "class Keep:\n    pass\n\n\nclass Internal:\n    pass\n\n\ndef helper(value: int) -> None: ...\n"
HIDDEN_OLD = "def hidden(first: int) -> None: ...\n"
HIDDEN_NEW = "def hidden() -> None: ...\n"


@attrs.frozen
class FakeLoad:
    """A stand-in for ``griffe.load`` that returns one package whatever the arguments."""

    package: griffe.Module

    def __call__(self, _name: str, **_options: str | Path | list[str] | list[Path]) -> griffe.Module:
        """Return the package.

        Returns
        -------
        griffe.Module
            The package that the object holds.

        """
        return self.package


def load(root: Path, init: str, core: str, hidden: str) -> griffe.Module:
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(init, encoding="utf-8")
    (package / "core.py").write_text(core, encoding="utf-8")
    (package / "hidden.py").write_text(hidden, encoding="utf-8")
    return check_api_breaks.as_module(griffe.load("pkg", search_paths=[root]))


@pytest.fixture
def old_package(tmp_path: Path) -> griffe.Module:
    return load(tmp_path / "old", OLD, CORE_OLD, HIDDEN_OLD)


@pytest.fixture
def new_package(tmp_path: Path) -> griffe.Module:
    return load(tmp_path / "new", NEW, CORE_NEW, HIDDEN_NEW)


def test_exported_paths_resolve_to_the_defining_objects(old_package: griffe.Module) -> None:
    assert check_api_breaks.exported_paths(old_package) == {
        "Keep": "pkg.core.Keep",
        "Gone": "pkg.core.Gone",
        "drop_me": "pkg.core.drop_me",
        "stay": "pkg.stay",
    }


def test_exported_paths_skip_a_name_that_does_not_resolve(tmp_path: Path) -> None:
    package = load(tmp_path, '__all__ = ["missing"]\n', "", "")
    assert check_api_breaks.exported_paths(package) == {}


def test_api_breaks_list_removed_names_and_changed_signatures(
    old_package: griffe.Module,
    new_package: griffe.Module,
) -> None:
    assert check_api_breaks.api_breaks(old_package, new_package) == [
        "pkg.Gone: removed from __all__",
        "pkg.core.Gone: Public object was removed",
        "pkg.core.drop_me: Public object was removed",
        "pkg.drop_me: removed from __all__",
        "pkg.stay: Parameter was removed",
    ]


def test_a_break_of_an_object_that_is_not_exported_is_not_a_break_of_the_api(
    old_package: griffe.Module,
    new_package: griffe.Module,
) -> None:
    assert not any("hidden" in line for line in check_api_breaks.api_breaks(old_package, new_package))


def test_equal_packages_have_no_break(old_package: griffe.Module) -> None:
    assert check_api_breaks.api_breaks(old_package, old_package) == []


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        (["### Changed\n\n- **BREAKING:** the name is gone.\n"], True),
        (["### Changed\n\n- **BREAKING** the name is gone.\n"], True),
        (["### Fixed\n\n- A fix.\n"], False),
        ([], False),
    ],
)
def test_has_breaking_entry(fragments: list[str], *, expected: bool) -> None:
    assert check_api_breaks.has_breaking_entry(fragments) is expected


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_package: griffe.Module,
    new_package: griffe.Module,
    fragment: str,
) -> int:
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d" / "README.md").write_text("Mark a break with **BREAKING:**.\n", encoding="utf-8")
    (tmp_path / "changelog.d" / "a.md").write_text(fragment, encoding="utf-8")
    monkeypatch.setattr(check_api_breaks, "FRAGMENTS", tmp_path / "changelog.d")
    monkeypatch.setattr(check_api_breaks.griffe, "load_git", FakeLoad(old_package))
    monkeypatch.setattr(check_api_breaks.griffe, "load", FakeLoad(new_package))
    return check_api_breaks.main(["--base", "origin/main"])


def test_main_fails_on_a_break_without_a_fragment_and_ignores_the_readme_of_the_fragments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_package: griffe.Module,
    new_package: griffe.Module,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_main(monkeypatch, tmp_path, old_package, new_package, "### Fixed\n\n- A fix.\n") == 1
    assert "Add an entry that starts with **BREAKING:**" in capsys.readouterr().err


def test_main_passes_when_a_fragment_announces_the_break(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_package: griffe.Module,
    new_package: griffe.Module,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fragment = "### Removed\n\n- **BREAKING:** `Gone` and `drop_me`.\n"
    assert run_main(monkeypatch, tmp_path, old_package, new_package, fragment) == 0
    assert "a fragment announces it" in capsys.readouterr().out


def test_main_passes_without_a_break(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_package: griffe.Module,
) -> None:
    assert run_main(monkeypatch, tmp_path, old_package, old_package, "### Fixed\n\n- A fix.\n") == 0


def test_as_module_rejects_another_kind_of_object() -> None:
    with pytest.raises(TypeError, match="ebaml is not a module, but Class"):
        check_api_breaks.as_module(griffe.Class("pkg"))
