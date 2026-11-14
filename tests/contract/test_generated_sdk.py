"""The generated ``baml_sdk`` against the BAML sources and the toolchain that ebaml pins.

``baml_sdk`` is generated and ignored by git, so a checkout can hold an SDK that an older ``baml_src`` made. The unit
tests pass against such an SDK, because their fakes read its stub, and the first real call then fails with an
``InvalidArgument`` error. These tests find the stale SDK at once, and they name the fix.
"""

import importlib
import importlib.metadata
import tomllib
from pathlib import Path

import orjson

import baml_sdk

ROOT = Path(__file__).resolve().parents[2]
FIX = "Run `baml generate`: baml_sdk does not match baml_src."


def _sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT / "baml_src").as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "baml_src").rglob("*.baml"))
    }


def test_the_sdk_embeds_each_baml_source_file_unchanged() -> None:
    embedded: dict[str, str] = importlib.import_module("baml_sdk._baml_sources").FILES
    assert set(embedded) == set(_sources()), FIX
    stale = [name for name, text in _sources().items() if embedded[name] != text]
    assert not stale, f"{FIX} Changed files: {stale}"


def test_the_sdk_has_the_files_that_the_generator_recorded() -> None:
    directory = Path(baml_sdk.__file__).parent
    record = orjson.loads((directory / ".baml-generator-output.json").read_bytes())
    missing = [
        entry["relative_path"] for entry in record["files"] if not (directory / entry["relative_path"]).is_file()
    ]
    assert not missing, f"the SDK lacks {missing}"


def test_the_installed_bridge_the_pin_of_the_project_and_the_toolchain_name_one_version() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = [dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("baml-bridge")]
    assert pinned == [f"baml-bridge=={importlib.metadata.version('baml-bridge')}"]
    toolchain = tomllib.loads((ROOT / "baml.toml").read_text(encoding="utf-8"))["toolchain"]["version"]
    assert toolchain == importlib.metadata.version("baml-bridge")


def test_every_sdk_function_that_ebaml_names_is_plain_and_not_async() -> None:
    # ebaml calls each function as a plain function. A toolchain that writes `async def` makes a call return a
    # coroutine, which no fake notices.
    stub = (Path(baml_sdk.__file__).with_suffix(".pyi")).read_text(encoding="utf-8")
    for name in ("describe_graph_cot", "summarize_ebm", "explain_local", "check_model", "supported_providers"):
        assert f"def {name}(" in stub
        assert f"async def {name}(" not in stub
