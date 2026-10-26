"""Tests for the setup of the generated BAML SDK."""

import importlib
import importlib.util
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the tests replace subprocess.run with a fake
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from ebaml.core.exceptions import SDKUnavailableError
from ebaml.llm import sdk_setup

if TYPE_CHECKING:
    from collections.abc import Callable

# every test needs the fake lookup of the SDK, or it would see the real package
pytestmark: pytest.MarkDecorator = pytest.mark.usefixtures("specs")

SPEC: ModuleSpec = ModuleSpec("baml_sdk", None)
INSTALL_COMMAND = "curl -fsSL https://pkg.boundaryml.com/install.sh | sh -s"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Build a source checkout with ``baml.toml`` and ``baml_src/`` at the root, and return its root.

    Returns
    -------
    Path
        The root.

    """
    (tmp_path / "baml.toml").write_text("[package]\n")
    (tmp_path / "baml_src").mkdir()
    (tmp_path / "src" / "ebaml" / "llm").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """Build a home directory without the BAML toolchain.

    Returns
    -------
    Path
        The directory.

    """
    directory = tmp_path / "home"
    directory.mkdir()
    return directory


class Runs:
    """A fake of ``subprocess.run`` that records the calls and returns one result or raises one error."""

    def __init__(
        self,
        *,
        result: subprocess.CompletedProcess[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Set the outcome of every call.

        Parameters
        ----------
        result : subprocess.CompletedProcess[str] | None
            The result to return.
        error : Exception | None
            The error to raise instead.

        """
        self.result: subprocess.CompletedProcess[str] | None = result
        self.error: Exception | None = error
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.after_run: Callable[[], None] = lambda: None

    def __call__(self, args: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        """Record the call, run the hook, then return or raise the outcome.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The result.

        """
        self.calls.append((args, options))
        self.after_run()
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def finished(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["baml", "generate"], returncode, stdout, stderr)


class Specs:
    """A fake of ``importlib.util.find_spec`` for ``baml_sdk``: the SDK exists once ``generate()`` has run."""

    def __init__(self, *, present: bool) -> None:
        """Set whether the SDK exists.

        Parameters
        ----------
        present : bool
            The first answer.

        """
        self.present: bool = present
        self.names: list[str] = []

    def generate(self) -> None:
        """Make the SDK exist."""
        self.present = True

    def __call__(self, name: str) -> ModuleSpec | None:
        """Record the name and answer.

        Returns
        -------
        ModuleSpec | None
            The spec, or ``None`` if the SDK does not exist.

        """
        self.names.append(name)
        return SPEC if self.present else None


@pytest.fixture
def specs(monkeypatch: pytest.MonkeyPatch) -> Specs:
    fake = Specs(present=False)
    monkeypatch.setattr(importlib.util, "find_spec", fake)
    return fake


def fake_which(name: str) -> str | None:
    return f"/usr/bin/{name}"


def fake_which_nothing(_name: str) -> str | None:
    return None


@pytest.fixture
def on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put ``baml`` on ``PATH`` for the lookup."""
    monkeypatch.setattr(shutil, "which", fake_which)


@pytest.fixture
def off_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", fake_which_nothing)


def install_baml(home: Path) -> Path:
    binary = home / ".baml" / "bin" / "baml"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    return binary


def test_a_present_sdk_needs_no_command(specs: Specs, monkeypatch: pytest.MonkeyPatch, checkout: Path) -> None:
    specs.present = True
    runs = Runs(error=AssertionError("no command is expected"))
    monkeypatch.setattr(subprocess, "run", runs)
    sdk_setup.ensure_sdk(start=checkout / "src")
    assert specs.names == ["baml_sdk"]
    assert not runs.calls


def test_a_missing_sdk_outside_a_checkout_names_the_reinstall(specs: Specs, tmp_path: Path, home: Path) -> None:
    assert not specs.present
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=tmp_path / "site-packages" / "ebaml", home=home)
    message = str(raised.value)
    assert message.startswith("baml_sdk is missing.\n")
    assert "Reinstall ebaml from a release file." in message
    assert "baml generate" in message
    assert INSTALL_COMMAND not in message


@pytest.mark.usefixtures("off_path")
def test_a_missing_baml_command_names_the_install_steps(
    specs: Specs,
    checkout: Path,
    home: Path,
) -> None:
    assert not specs.present
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    message = str(raised.value)
    assert message.startswith(
        "baml_sdk is missing, and ebaml cannot generate it because the `baml` command was not found.",
    )
    assert f"  1. Install the BAML toolchain: {INSTALL_COMMAND}\n" in message
    assert '  2. Open a new shell, or run `. "$HOME/.baml/env"`, so that `baml` is on PATH.\n' in message
    assert "  3. Check that `baml --version` prints 0.23 or later." in message
    assert f"  4. Run `baml generate` in {checkout.resolve()}." in message


@pytest.mark.usefixtures("on_path")
def test_generate_runs_the_command_of_the_path_in_the_checkout(
    specs: Specs,
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    runs = Runs(result=finished())
    runs.after_run = specs.generate
    monkeypatch.setattr(subprocess, "run", runs)
    sdk_setup.ensure_sdk(start=checkout / "src" / "ebaml", home=home)
    assert runs.calls == [
        (
            ["/usr/bin/baml", "generate"],
            {"cwd": checkout.resolve(), "capture_output": True, "text": True, "timeout": 300, "check": False},
        ),
    ]


@pytest.mark.usefixtures("off_path")
def test_the_command_in_the_home_directory_runs_when_the_path_has_none(
    specs: Specs,
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    binary = install_baml(home)
    runs = Runs(result=finished())
    runs.after_run = specs.generate
    monkeypatch.setattr(subprocess, "run", runs)
    sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    assert [args for args, _ in runs.calls] == [[str(binary), "generate"]]


@pytest.mark.usefixtures("on_path")
def test_the_command_of_the_path_wins_over_the_home_directory(
    specs: Specs,
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    install_baml(home)
    runs = Runs(result=finished())
    runs.after_run = specs.generate
    monkeypatch.setattr(subprocess, "run", runs)
    sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    assert runs.calls[0][0] == ["/usr/bin/baml", "generate"]


@pytest.mark.usefixtures("on_path")
def test_a_generated_sdk_clears_the_import_caches_and_logs(
    specs: Specs,
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    runs = Runs(result=finished())
    runs.after_run = specs.generate
    monkeypatch.setattr(subprocess, "run", runs)
    cleared: list[bool] = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: cleared.append(True))
    with capture_logs() as logs:
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    assert cleared == [True]
    assert [(log["event"], log["log_level"]) for log in logs] == [("sdk_generate", "info"), ("sdk_generated", "info")]
    assert logs[0]["command"] == "/usr/bin/baml"
    assert logs[0]["cwd"] == str(checkout.resolve())
    assert logs[1]["root"] == str(checkout.resolve())


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_shows_the_end_of_its_error_output(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    stderr = "\n".join(f"line {number}" for number in range(40))
    monkeypatch.setattr(subprocess, "run", Runs(result=finished(2, stdout="ignored", stderr=stderr)))
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    message = str(raised.value)
    assert "`/usr/bin/baml generate` failed with exit code 2. Its output ends:\n" in message
    assert "  line 39\n" in message
    assert "  line 25\n" in message
    assert "line 24" not in message
    assert "ignored" not in message
    assert "  2. Run `baml generate` in" in message
    assert INSTALL_COMMAND not in message


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_with_only_standard_output_shows_that(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    monkeypatch.setattr(subprocess, "run", Runs(result=finished(1, stdout="unknown command: generate")))
    with pytest.raises(SDKUnavailableError, match=r"ends:\n  unknown command: generate\n"):
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_without_output_says_so(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    monkeypatch.setattr(subprocess, "run", Runs(result=finished(1)))
    with pytest.raises(SDKUnavailableError, match=r"ends:\n  \(no output\)\n"):
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)


@pytest.mark.parametrize(
    "error",
    [PermissionError("denied"), subprocess.TimeoutExpired(["baml", "generate"], 300)],
    ids=["cannot start", "times out"],
)
@pytest.mark.usefixtures("on_path")
def test_a_command_that_does_not_finish_gives_the_cause(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
    error: Exception,
) -> None:
    monkeypatch.setattr(subprocess, "run", Runs(error=error))
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    assert "`/usr/bin/baml generate` did not finish: " in str(raised.value)
    assert str(error) in str(raised.value)
    assert raised.value.__cause__ is error


@pytest.mark.usefixtures("on_path")
def test_an_sdk_that_stays_missing_after_the_command_names_the_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    home: Path,
) -> None:
    monkeypatch.setattr(subprocess, "run", Runs(result=finished()))
    with pytest.raises(SDKUnavailableError) as raised, capture_logs() as logs:
        sdk_setup.ensure_sdk(start=checkout / "src", home=home)
    message = str(raised.value)
    assert message.startswith("`/usr/bin/baml generate` finished, but baml_sdk is still not importable.")
    assert str(checkout.resolve() / "baml.toml") in message
    assert "`uv sync`" in message
    assert [log["event"] for log in logs] == ["sdk_generate"]


@pytest.mark.parametrize("present", [["baml.toml"], ["baml_src"], []])
def test_the_project_root_needs_both_marks(tmp_path: Path, present: list[str]) -> None:
    (tmp_path / "baml.toml").unlink(missing_ok=True)
    if "baml.toml" in present:
        (tmp_path / "baml.toml").write_text("")
    if "baml_src" in present:
        (tmp_path / "baml_src").mkdir()
    inside = tmp_path / "a" / "b"
    inside.mkdir(parents=True)
    assert sdk_setup._project_root(inside) is None


def test_the_project_root_is_the_nearest_directory_with_both_marks(checkout: Path) -> None:
    inner = checkout / "inner"
    (inner / "baml_src").mkdir(parents=True)
    (inner / "baml.toml").write_text("")
    assert sdk_setup._project_root(inner / "x") == inner.resolve()
    assert sdk_setup._project_root(checkout / "src") == checkout.resolve()


def test_the_project_root_of_a_file_is_found_from_its_parents(checkout: Path) -> None:
    module = checkout / "src" / "ebaml" / "llm" / "x.py"
    module.write_text("")
    assert sdk_setup._project_root(module) == checkout.resolve()
