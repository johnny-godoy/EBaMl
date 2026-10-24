"""Tests for the options of the descriptors."""

import inspect

import attrs
import pytest

from ebaml.graphs import EBMGraph
from ebaml.options import GraphOptions, PromptOptions


def _defaults(function: object, names: set[str]) -> dict[str, object]:
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


@pytest.mark.parametrize("options", [GraphOptions(), PromptOptions()])
def test_options_are_frozen(options: object) -> None:
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        options.max_tokens = 1  # type: ignore[attr-defined] -- the test sets a field of a frozen class on purpose
