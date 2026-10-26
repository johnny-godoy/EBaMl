"""Options of the functions in ``ebaml.llm.descriptors``.

Each class holds the settings of one part of the work: ``GraphOptions`` for the graph and its text, ``PromptOptions``
for the prompts, ``ModelOptions`` for the LLM. ``ebaml.local.LocalOptions`` holds the settings of the local facts. A
default of an option is the default of the function that reads it, so a call with no options does what the function
does alone.
"""

from collections.abc import Mapping
from typing import Literal

import attrs

from ebaml.core.custom_types import FeatureType, Normalization
from ebaml.core.exceptions import InvalidOptionError, OptionRangeError


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


def _check_temperature(_instance: object, _attribute: object, value: float | None) -> None:
    if value is not None and value < 0:
        msg = f"temperature must not be negative, got {value}"
        raise OptionRangeError(msg)


def _check_max_output_tokens(_instance: object, _attribute: object, value: int | None) -> None:
    if value is not None and value < 1:
        msg = f"max_output_tokens must be at least 1, got {value}"
        raise OptionRangeError(msg)


@attrs.frozen
class ModelOptions:
    """The LLM that a call uses. A field of ``None`` keeps the default of the call.

    The key of the model never sits in this class. ``api_key_env`` names the environment variable that holds it, and
    EBaMl reads the variable only when it sends a request. BAML lists the valid providers. ``describe_ebm`` and the
    other functions raise ``InvalidOptionError`` before the first paid call if BAML does not know the provider.

    Attributes
    ----------
    provider: str
        The BAML provider prefix, such as ``"openai"``, ``"anthropic"``, ``"openrouter"`` or ``"ollama"``.
    model: str
        The name of the model at the provider.
    api_key_env: str | None
        The name of the environment variable that holds the key. ``None`` selects the default variable of the provider.
    base_url: str | None
        The address of the server, for a gateway, a local server or Azure. ``None`` selects the address of the provider.
    temperature: float | None
        The sampling temperature. ``None`` selects 0.7, or 0.0 for the benchmark.
    max_output_tokens: int | None
        The most tokens of one reply. A reasoning model counts its hidden tokens in this limit. ``None`` selects 6000,
        or 8000 for the first turn of a conversation and for the benchmark.
    headers: Mapping[str, str]
        Headers to add to each request.

    """

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = attrs.field(default=None, validator=_check_temperature)
    max_output_tokens: int | None = attrs.field(default=None, validator=_check_max_output_tokens)
    headers: Mapping[str, str] = attrs.field(factory=dict)

    @classmethod
    def parse(cls, value: "ModelOptions | str") -> "ModelOptions":
        """Read the ``"provider/model"`` shorthand, or pass a ``ModelOptions`` on.

        The string splits at the first ``/`` only, so ``"openrouter/openrouter/free"`` is the provider ``openrouter``
        with the model ``openrouter/free``.

        Parameters
        ----------
        value: ModelOptions | str
            The options, or the shorthand.

        Returns
        -------
        ModelOptions
            The options.

        Raises
        ------
        InvalidOptionError
            If the string has no ``/``, or its provider or model is empty.

        """
        if isinstance(value, ModelOptions):
            return value
        provider, separator, model = value.partition("/")
        if not (separator and provider and model):
            raise InvalidOptionError(value, ["provider/model"])
        return cls(provider=provider, model=model)
