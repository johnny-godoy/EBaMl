"""Tests for the options of the descriptors."""

import inspect
from collections.abc import Callable

import attrs
import pytest

from ebaml.core.exceptions import InvalidOptionError, OptionRangeError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions


def _defaults(function: Callable[..., object], names: set[str]) -> dict[str, object]:
    parameters = inspect.signature(function).parameters
    return {name: parameter.default for name, parameter in parameters.items() if name in names}


def test_graph_options_default_to_the_defaults_of_the_graph_methods() -> None:
    from_ebm = _defaults(EBMGraph.from_ebm, {"normalization", "use_feature_bounds"})
    to_text = _defaults(EBMGraph.to_text, set(attrs.fields_dict(GraphOptions)) - set(from_ebm))
    assert attrs.asdict(GraphOptions()) == {**from_ebm, **to_text}


def test_graph_options_cover_the_options_of_the_graph_methods() -> None:
    to_text = set(inspect.signature(EBMGraph.to_text).parameters) - {"self", "check_budget"}
    from_ebm = set(inspect.signature(EBMGraph.from_ebm).parameters) - {"cls", "ebm", "feature_index"}
    assert set(attrs.fields_dict(GraphOptions)) == to_text | from_ebm


def test_prompt_options_keep_the_baml_defaults() -> None:
    assert attrs.asdict(PromptOptions()) == {
        "expert_description": None,
        "graph_description": None,
        "dataset_description": None,
        "task_description": None,
    }


@pytest.mark.parametrize("options", [GraphOptions(), PromptOptions(), ModelOptions("openai", "m")])
def test_options_are_frozen(options: object) -> None:
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        options.max_tokens = 1  # type: ignore[attr-defined] -- the test sets a field of a frozen class on purpose


def test_model_options_keep_the_defaults_of_the_call() -> None:
    assert attrs.asdict(ModelOptions("openai", "gpt-5")) == {
        "provider": "openai",
        "model": "gpt-5",
        "api_key_env": None,
        "base_url": None,
        "temperature": None,
        "max_output_tokens": None,
        "headers": {},
    }


def test_model_options_do_not_share_the_default_headers() -> None:
    first, second = ModelOptions("openai", "m"), ModelOptions("openai", "m")
    assert first.headers is not second.headers


@pytest.mark.parametrize(
    ("text", "provider", "model"),
    [
        ("anthropic/claude-haiku-4-5", "anthropic", "claude-haiku-4-5"),
        ("openrouter/openrouter/free", "openrouter", "openrouter/free"),
        ("ollama/llama3.1:8b", "ollama", "llama3.1:8b"),
    ],
)
def test_parse_splits_at_the_first_slash(text: str, provider: str, model: str) -> None:
    assert ModelOptions.parse(text) == ModelOptions(provider, model)


def test_parse_passes_model_options_on() -> None:
    options = ModelOptions("openai", "m", temperature=0.2)
    assert ModelOptions.parse(options) is options


@pytest.mark.parametrize("text", ["gpt-5", "", "/gpt-5", "openai/", "/"])
def test_parse_rejects_a_string_without_provider_or_model(text: str) -> None:
    with pytest.raises(InvalidOptionError, match="provider/model"):
        ModelOptions.parse(text)


def test_model_options_reject_a_negative_temperature() -> None:
    with pytest.raises(OptionRangeError, match="temperature"):
        ModelOptions("openai", "m", temperature=-0.1)


def test_model_options_accept_a_temperature_of_zero() -> None:
    assert ModelOptions("openai", "m", temperature=0).temperature == 0


@pytest.mark.parametrize("limit", [0, -5])
def test_model_options_reject_an_output_limit_below_one(limit: int) -> None:
    with pytest.raises(OptionRangeError, match="max_output_tokens"):
        ModelOptions("openai", "m", max_output_tokens=limit)


def test_model_options_accept_an_output_limit_of_one() -> None:
    assert ModelOptions("openai", "m", max_output_tokens=1).max_output_tokens == 1
