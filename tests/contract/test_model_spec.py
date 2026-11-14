"""The model that Python sends to BAML: the fields, the providers, and the defaults that two files must keep equal."""

import re
from collections.abc import Callable
from pathlib import Path

import attrs
import pytest
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml import ModelOptions
from tests.unit.ebaml.conftest import sdk_fields

ROOT = Path(__file__).resolve().parents[2]
BAML_CLIENTS: str = (ROOT / "baml_src" / "talk_to_ebm.baml").read_text(encoding="utf-8")
# the default client of each call, with the function of model_spec.baml that holds its settings
DEFAULTS: dict[str, Callable[[], baml_sdk.CallDefaults]] = {
    "RouterBase": baml_sdk.short_defaults,
    "RouterLongBase": baml_sdk.long_defaults,
    "RouterBenchmarkBase": baml_sdk.benchmark_defaults,
}


def _setting(line: str, pattern: str) -> str:
    found = re.search(pattern, line)
    assert found, f"the client line has no {pattern}"
    return found.group(1)


def test_each_field_of_the_model_options_has_a_field_of_the_spec_and_the_other_way_round() -> None:
    options = {field.name for field in attrs.fields(ModelOptions)}
    assert options == set(sdk_fields("ModelSpec"))


@pytest.mark.parametrize("client", sorted(DEFAULTS))
def test_the_defaults_of_the_spec_equal_the_settings_of_the_default_client(client: str) -> None:
    line = next(line for line in BAML_CLIENTS.splitlines() if line.startswith(f"client {client} "))
    temperature = float(_setting(line, r"temperature = ([0-9.]+)"))
    max_tokens = int(_setting(line, r"max_tokens = (\d+)"))
    defaults = DEFAULTS[client]()
    assert (defaults.temperature, defaults.max_output_tokens) == (temperature, max_tokens)


def test_the_documented_defaults_are_the_defaults_of_the_code() -> None:
    doc = ModelOptions.__doc__ or ""
    assert "0.7" in doc
    assert str(baml_sdk.short_defaults().max_output_tokens) in doc
    assert str(baml_sdk.long_defaults().max_output_tokens) in doc
    assert "0.0" in doc


def test_every_provider_of_baml_builds_a_client_and_a_name_that_it_lacks_does_not() -> None:
    for provider in baml_sdk.supported_providers():
        baml_sdk.check_model(spec=baml_sdk.ModelSpec(provider=provider, model="model"))
    with pytest.raises(BamlError) as error:
        baml_sdk.check_model(spec=baml_sdk.ModelSpec(provider="no-such-provider", model="model"))
    assert error.value.class_name == "baml.errors.InvalidArgument"


def test_the_providers_that_the_docs_name_exist() -> None:
    documented = {"openai", "anthropic", "openrouter", "ollama", "openai-generic"}
    assert documented <= set(baml_sdk.supported_providers())
