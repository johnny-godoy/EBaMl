"""Options of the functions in ``ebaml.llm.descriptors``.

Each class holds the settings of one part of the work: ``GraphOptions`` for the graph and its text, ``PromptOptions``
for the prompts. ``ebaml.local.LocalOptions`` holds the settings of the local facts. A default of an option is the
default of the function that reads it, so a call with no options does what the function does alone.
"""

from typing import Literal

import attrs

from ebaml.core.custom_types import FeatureType, Normalization


@attrs.frozen
class GraphOptions:
    """Options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.

    Attributes
    ----------
    normalization: Normalization | str
        See ``EBMGraph.from_ebm``.
    use_feature_bounds: bool
        See ``EBMGraph.from_ebm``.
    include_description: bool
        See ``EBMGraph.to_text``.
    feature_format: FeatureType | str | None
        See ``EBMGraph.to_text``.
    x_axis_precision: int | None
        See ``EBMGraph.to_text``.
    y_axis_precision: int | Literal["auto"] | None
        See ``EBMGraph.to_text``.
    confidence_bounds: bool
        See ``EBMGraph.to_text``.
    confidence_level: float
        See ``EBMGraph.to_text``.
    max_tokens: int
        See ``EBMGraph.to_text``.
    overbudget_percent: int
        See ``EBMGraph.to_text``.

    """

    normalization: Normalization | str = Normalization.NONE
    use_feature_bounds: bool = True
    include_description: bool = True
    feature_format: FeatureType | str | None = None
    x_axis_precision: int | None = None
    y_axis_precision: int | Literal["auto"] | None = None
    confidence_bounds: bool = True
    confidence_level: float = 0.95
    max_tokens: int = 3000
    overbudget_percent: int = 50


@attrs.frozen
class PromptOptions:
    """The descriptions that the prompts use. A value of ``None`` keeps the BAML default.

    Attributes
    ----------
    expert_description: str | None
        Who the LLM plays.
    graph_description: str | None
        What a graph is and how to read it.
    dataset_description: str | None
        What the data is about. A good description helps the LLM.
    task_description: str | None
        What the model predicts.

    """

    expert_description: str | None = None
    graph_description: str | None = None
    dataset_description: str | None = None
    task_description: str | None = None
