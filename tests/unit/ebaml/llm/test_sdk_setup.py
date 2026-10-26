"""Tests for the setup of the generated BAML SDK."""

import importlib
import importlib.util
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- the tests replace subprocess.run with a fake
import sys
import types
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path

import attrs
import pytest
from structlog.testing import capture_logs

from ebaml.core.exceptions import SDKUnavailableError
from ebaml.llm import sdk_setup

SPEC: ModuleSpec = ModuleSpec("baml_sdk", None)
REAL_IMPORT_MODULE = importlib.import_module
REAL_FORGET_SDK = sdk_setup._forget_sdk
INSTALL_COMMAND = "curl -fsSL https://pkg.boundaryml.com/install.sh | sh -s"
SKEW = (
    "BAML startup failed: version skew error.\n\n"
    "`baml_sdk` was generated using BAML toolchain 0.22.2, but baml-bridge is installed at 0.23.0 and expects "
    "baml_sdk to be generated using BAML toolchain 0.23.0.\n\nTo fix this, either:"
)
SKILL_ERROR = "error: the installed BAML agent skill does not match this toolchain; run `baml agent install`"


def finished(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["baml"], returncode, stdout, stderr)


class Runs:
    """A fake of ``subprocess.run`` that records the calls and plays the outcomes in order, the last one forever."""

    def __init__(
        self,
        setup: "Setup",
        outcomes: Sequence[subprocess.CompletedProcess[str] | Exception],
        *,
        creates_sdk: bool,
    ) -> None:
        """Set the outcomes.

        Parameters
        ----------
        setup : Setup
            The setup of the test, which holds the events and the SDK.
        outcomes : Sequence[subprocess.CompletedProcess[str] | Exception]
            The result to return, or the error to raise, for each call.
        creates_sdk : bool
            Whether the SDK exists after the first call.

        """
        self.setup: Setup = setup
        self.outcomes: Sequence[subprocess.CompletedProcess[str] | Exception] = outcomes
        self.creates_sdk: bool = creates_sdk
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        """Record the call, then return or raise its outcome.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The result.

        """
        self.calls.append((args, options))
        self.setup.events.append("run " + " ".join(args[1:]))
        if self.creates_sdk:
            self.setup.specs_present = True
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@attrs.define
class Setup:
    """The fakes of one test, and the directories: everything that ``ensure_sdk`` touches."""

    checkout: Path
    home: Path
    monkeypatch: pytest.MonkeyPatch
    events: list[str] = attrs.field(factory=list)
    specs_present: bool = False
    import_errors: list[Exception] = attrs.field(factory=list)
    forgotten: list[str] = attrs.field(factory=list)

    def runs(self, *outcomes: subprocess.CompletedProcess[str] | Exception, creates_sdk: bool = False) -> Runs:
        """Replace ``subprocess.run`` with a fake.

        Parameters
        ----------
        outcomes : subprocess.CompletedProcess[str] | Exception
            The result to return, or the error to raise, for each call.
        creates_sdk : bool
            Whether the SDK exists after the first call.

        Returns
        -------
        Runs
            The fake, which records the calls.

        """
        fake = Runs(self, outcomes, creates_sdk=creates_sdk)
        self.monkeypatch.setattr(subprocess, "run", fake)
        return fake

    def find_spec(self, name: str) -> ModuleSpec | None:
        """Answer like ``importlib.util.find_spec`` for the SDK.

        Returns
        -------
        ModuleSpec | None
            The spec, or ``None`` if the SDK does not exist.

        """
        self.events.append(f"find {name}")
        return SPEC if self.specs_present else None

    def import_module(self, name: str, package: str | None = None) -> types.ModuleType:
        """Import any other module for real, and play the queued errors for ``baml_sdk``.

        Returns
        -------
        types.ModuleType
            The module.

        """
        if name != "baml_sdk":
            return REAL_IMPORT_MODULE(name, package)
        self.events.append("import")
        if self.import_errors:
            raise self.import_errors.pop(0)
        return types.ModuleType(name)

    def forget(self, package: str) -> None:
        """Take the place of the removal from the module cache, so that the real SDK of the test run stays."""
        self.forgotten.append(package)
        self.events.append("forget")

    def no_cache_clear(self) -> None:
        """Replace ``importlib.invalidate_caches`` with a function that does nothing."""
        self.monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    """Build a source checkout, a home directory without the toolchain, and the fakes.

    Returns
    -------
    Setup
        The setup.

    """
    (tmp_path / "baml.toml").write_text("[package]\n")
    (tmp_path / "baml_src").mkdir()
    (tmp_path / "src" / "ebaml" / "llm").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    made = Setup(tmp_path.resolve(), home, monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", made.find_spec)
    monkeypatch.setattr(importlib, "import_module", made.import_module)
    monkeypatch.setattr(sdk_setup, "_forget_sdk", made.forget)
    return made


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
    """Take ``baml`` off ``PATH`` for the lookup."""
    monkeypatch.setattr(shutil, "which", fake_which_nothing)


def install_baml(home: Path) -> Path:
    binary = home / ".baml" / "bin" / "baml"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    return binary


# --- a missing SDK ---


def test_a_present_sdk_needs_no_command(setup: Setup) -> None:
    setup.specs_present = True
    runs = setup.runs(AssertionError("no command is expected"))
    sdk_setup.ensure_sdk(start=setup.checkout / "src")
    assert setup.events == ["find baml_sdk", "import"]
    assert not runs.calls


def test_a_missing_sdk_outside_a_checkout_names_the_reinstall(
    setup: Setup,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    elsewhere = tmp_path_factory.mktemp("site-packages")
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=elsewhere / "ebaml", home=setup.home)
    message = str(raised.value)
    assert message.startswith("baml_sdk is missing.\n")
    assert "Reinstall ebaml from a release file." in message
    assert "baml generate" in message
    assert INSTALL_COMMAND not in message


@pytest.mark.usefixtures("off_path")
def test_a_missing_baml_command_names_the_install_steps(setup: Setup) -> None:
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert message.startswith(
        "baml_sdk is missing, and ebaml cannot generate it because the `baml` command was not found.",
    )
    assert f"  1. Install the BAML toolchain: {INSTALL_COMMAND}\n" in message
    assert '  2. Open a new shell, or run `. "$HOME/.baml/env"`, so that `baml` is on PATH.\n' in message
    assert "  3. Check that `baml --version` prints 0.23 or later." in message
    assert f"  4. Run `baml generate` in {setup.checkout}." in message


@pytest.mark.usefixtures("on_path")
def test_generate_runs_the_command_of_the_path_in_the_checkout(setup: Setup) -> None:
    runs = setup.runs(finished(), creates_sdk=True)
    sdk_setup.ensure_sdk(start=setup.checkout / "src" / "ebaml", home=setup.home)
    assert runs.calls == [
        (
            ["/usr/bin/baml", "generate"],
            {"cwd": setup.checkout, "capture_output": True, "text": True, "timeout": 300, "check": False},
        ),
    ]


@pytest.mark.usefixtures("off_path")
def test_the_command_in_the_home_directory_runs_when_the_path_has_none(setup: Setup) -> None:
    binary = install_baml(setup.home)
    runs = setup.runs(finished(), creates_sdk=True)
    sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert [args for args, _ in runs.calls] == [[str(binary), "generate"]]


@pytest.mark.usefixtures("on_path")
def test_the_command_of_the_path_wins_over_the_home_directory(setup: Setup) -> None:
    install_baml(setup.home)
    runs = setup.runs(finished(), creates_sdk=True)
    sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert runs.calls[0][0] == ["/usr/bin/baml", "generate"]


@pytest.mark.usefixtures("on_path")
def test_a_generated_sdk_clears_the_import_caches_and_logs(setup: Setup) -> None:
    setup.runs(finished(), creates_sdk=True)
    setup.monkeypatch.setattr(importlib, "invalidate_caches", lambda: setup.events.append("invalidate"))
    with capture_logs() as logs:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert setup.events == ["find baml_sdk", "run generate", "invalidate", "find baml_sdk", "import"]
    assert [(log["event"], log["log_level"]) for log in logs] == [("sdk_command", "info"), ("sdk_generated", "info")]
    assert logs[0]["command"] == "/usr/bin/baml generate"
    assert logs[0]["cwd"] == str(setup.checkout)
    assert logs[1]["root"] == str(setup.checkout)


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_shows_the_end_of_its_error_output(setup: Setup) -> None:
    stderr = "\n".join(f"line {number}" for number in range(40))
    setup.runs(finished(2, stdout="ignored", stderr=stderr))
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert message.startswith("baml_sdk is missing, and `/usr/bin/baml generate` failed with exit code 2.")
    assert "Its output ends:\n" in message
    assert "  line 39\n" in message
    assert "  line 25\n" in message
    assert "line 24" not in message
    assert "ignored" not in message
    assert "  2. Run `baml generate` in" in message
    assert INSTALL_COMMAND not in message


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_with_only_standard_output_shows_that(setup: Setup) -> None:
    setup.runs(finished(1, stdout="unknown command: generate"))
    with pytest.raises(SDKUnavailableError, match=r"ends:\n  unknown command: generate\n"):
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)


@pytest.mark.usefixtures("on_path")
def test_a_failed_command_without_output_says_so(setup: Setup) -> None:
    setup.runs(finished(1))
    with pytest.raises(SDKUnavailableError, match=r"ends:\n  \(no output\)\n"):
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)


@pytest.mark.usefixtures("on_path")
@pytest.mark.parametrize(
    "error",
    [PermissionError("denied"), subprocess.TimeoutExpired(["baml", "generate"], 300)],
    ids=["cannot start", "times out"],
)
def test_a_command_that_does_not_finish_gives_the_cause(setup: Setup, error: Exception) -> None:
    setup.runs(error)
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert "`/usr/bin/baml generate` did not finish: " in str(raised.value)
    assert str(error) in str(raised.value)
    assert raised.value.__cause__ is error


@pytest.mark.usefixtures("on_path")
def test_an_sdk_that_stays_missing_after_the_command_names_the_output_directory(setup: Setup) -> None:
    setup.runs(finished())
    with pytest.raises(SDKUnavailableError) as raised, capture_logs() as logs:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert message.startswith("`/usr/bin/baml generate` finished, but baml_sdk is still not importable.")
    assert str(setup.checkout / "baml.toml") in message
    assert "`uv sync`" in message
    assert [log["event"] for log in logs] == ["sdk_command"]


# --- the agent skill check ---


@pytest.mark.usefixtures("on_path")
def test_a_command_that_fails_on_the_agent_skill_check_runs_again_with_the_check_off(setup: Setup) -> None:
    runs = setup.runs(finished(1, stderr=SKILL_ERROR), finished(), creates_sdk=True)
    with capture_logs() as logs:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert [args for args, _ in runs.calls] == [
        ["/usr/bin/baml", "generate"],
        ["/usr/bin/baml", "generate", "--agent-skill-check", "off"],
    ]
    assert [log["event"] for log in logs] == ["sdk_command", "sdk_skill_check_off", "sdk_command", "sdk_generated"]
    assert logs[1]["command"] == "/usr/bin/baml generate"


@pytest.mark.usefixtures("on_path")
def test_the_agent_skill_check_is_turned_off_once(setup: Setup) -> None:
    runs = setup.runs(finished(1, stderr=SKILL_ERROR))
    with pytest.raises(SDKUnavailableError, match="`/usr/bin/baml generate --agent-skill-check off` failed"):
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert len(runs.calls) == 2


@pytest.mark.usefixtures("on_path")
def test_a_failure_that_is_not_the_agent_skill_check_does_not_run_again(setup: Setup) -> None:
    runs = setup.runs(finished(1, stderr="syntax error"))
    with pytest.raises(SDKUnavailableError):
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert len(runs.calls) == 1


# --- a version skew ---


def test_an_error_that_is_not_a_skew_reaches_the_caller(setup: Setup) -> None:
    setup.specs_present = True
    runs = setup.runs(AssertionError("no command is expected"))
    setup.import_errors.append(RuntimeError("the runtime is broken"))
    with pytest.raises(RuntimeError, match="the runtime is broken") as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src")
    assert not isinstance(raised.value, SDKUnavailableError)
    assert not runs.calls


@pytest.mark.usefixtures("on_path")
def test_a_skew_selects_the_expected_toolchain_generates_and_imports_again(setup: Setup) -> None:
    setup.specs_present = True
    runs = setup.runs(finished())
    setup.import_errors.append(RuntimeError(SKEW))
    setup.no_cache_clear()
    with capture_logs() as logs:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    assert [args for args, _ in runs.calls] == [
        ["/usr/bin/baml", "toolchain", "use", "0.23.0"],
        ["/usr/bin/baml", "generate"],
    ]
    assert all(options["cwd"] == setup.checkout for _, options in runs.calls)
    assert setup.events == [
        "find baml_sdk",
        "import",
        "run toolchain use 0.23.0",
        "run generate",
        "forget",
        "import",
    ]
    assert setup.forgotten == ["baml_sdk"]
    assert logs[-1]["event"] == "sdk_repaired"
    assert logs[-1]["toolchain"] == "0.23.0"


@pytest.mark.parametrize(
    "text",
    [SKEW, "BAML startup failed: version skew error. The versions differ."],
    ids=["version named", "version not named"],
)
def test_a_skew_outside_a_checkout_names_the_reinstall(
    setup: Setup,
    tmp_path_factory: pytest.TempPathFactory,
    text: str,
) -> None:
    setup.specs_present = True
    runs = setup.runs(AssertionError("no command is expected"))
    error = RuntimeError(text)
    setup.import_errors.append(error)
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=tmp_path_factory.mktemp("site-packages") / "ebaml", home=setup.home)
    message = str(raised.value)
    assert message.startswith(text)
    assert "need the same BAML toolchain version." in message
    assert "reinstall ebaml and baml-bridge together from the same release." in message
    assert raised.value.__cause__ is error
    assert not runs.calls


def test_a_skew_without_the_expected_version_in_a_checkout_names_the_reinstall(setup: Setup) -> None:
    setup.specs_present = True
    setup.import_errors.append(RuntimeError("BAML startup failed: version skew error."))
    with pytest.raises(SDKUnavailableError, match="reinstall ebaml and baml-bridge together"):
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)


@pytest.mark.usefixtures("off_path")
def test_a_skew_without_the_baml_command_names_the_install_steps(setup: Setup) -> None:
    setup.specs_present = True
    error = RuntimeError(SKEW)
    setup.import_errors.append(error)
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert message.startswith(
        "baml_sdk was generated by another BAML toolchain than baml-bridge expects (0.23.0), "
        "and ebaml cannot repair it because the `baml` command was not found.",
    )
    assert f"  1. Install the BAML toolchain: {INSTALL_COMMAND}\n" in message
    assert f"  4. Run `baml toolchain use 0.23.0` in {setup.checkout}.\n" in message
    assert f"  5. Run `baml generate` in {setup.checkout}." in message
    assert raised.value.__cause__ is error


@pytest.mark.usefixtures("on_path")
def test_a_skew_repair_that_fails_to_select_the_toolchain_does_not_generate(setup: Setup) -> None:
    setup.specs_present = True
    runs = setup.runs(finished(3, stderr="no network"))
    setup.import_errors.append(RuntimeError(SKEW))
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert message.startswith(
        "baml_sdk was generated by another BAML toolchain than baml-bridge expects (0.23.0), "
        "and `/usr/bin/baml toolchain use 0.23.0` failed with exit code 3.",
    )
    assert "  no network\n" in message
    assert f"  2. Run `baml toolchain use 0.23.0` in {setup.checkout}.\n" in message
    assert f"  3. Run `baml generate` in {setup.checkout}." in message
    assert len(runs.calls) == 1


@pytest.mark.usefixtures("on_path")
def test_a_skew_that_stays_after_the_repair_names_the_pin(setup: Setup) -> None:
    setup.specs_present = True
    setup.runs(finished())
    setup.no_cache_clear()
    setup.import_errors.extend([RuntimeError(SKEW), RuntimeError(SKEW.replace("0.22.2", "0.21.0"))])
    with pytest.raises(SDKUnavailableError) as raised:
        sdk_setup.ensure_sdk(start=setup.checkout / "src", home=setup.home)
    message = str(raised.value)
    assert "generated using BAML toolchain 0.21.0," in message
    assert "ran `baml toolchain use 0.23.0` and `baml generate`, but the versions still differ." in message
    assert f"check that the `baml.toml` in {setup.checkout} does not pin another toolchain" in message
    assert "(`baml toolchain pin 0.23.0` pins this one)" in message


@pytest.mark.usefixtures("off_path")
def test_a_skew_repair_finds_the_checkout_from_a_deep_path_and_the_command_in_the_home_directory(
    setup: Setup,
) -> None:
    setup.specs_present = True
    binary = install_baml(setup.home)
    runs = setup.runs(finished())
    setup.no_cache_clear()
    setup.import_errors.append(RuntimeError(SKEW))
    sdk_setup.ensure_sdk(start=setup.checkout / "src" / "ebaml" / "llm", home=setup.home)
    assert runs.calls[0][0][0] == str(binary)
    assert runs.calls[0][1]["cwd"] == setup.checkout


# --- the module cache ---


def test_forgetting_a_package_removes_it_and_its_submodules_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("fakepkg", "fakepkg.sub", "fakepkg.sub.deep", "fakepkg_other", "other.fakepkg"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    cleared: list[bool] = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: cleared.append(True))
    REAL_FORGET_SDK("fakepkg")
    assert "fakepkg" not in sys.modules
    assert "fakepkg.sub" not in sys.modules
    assert "fakepkg.sub.deep" not in sys.modules
    assert "fakepkg_other" in sys.modules
    assert "other.fakepkg" in sys.modules
    assert cleared == [True]
