"""The requests of the functions in ``ebaml.llm.descriptors``.

Each function takes one request. A request holds the model, the settings of the call and the option sets, so that a
function has one parameter and a call names each value. A default of a field is the default of the function that reads
it. Every option field is keyword-only, because an option set has no natural place in a positional list.
"""

from collections.abc import Mapping

import attrs

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.local.audience import Audience
from ebaml.local.fact_types import LocalOptions


@attrs.frozen
class _EBMRequest:
    """The fields that every request has.

    Attributes
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    prompt: PromptOptions
        The descriptions that the prompts use.
    model: ModelOptions | str | None
        The LLM of the call, as ``ModelOptions`` or the ``"provider/model"`` shorthand. ``None`` selects OpenRouter's
        ``openrouter/free``. EBaMl sends every request of the call to this model.

    """

    ebm: ExplainableBoostingModel
    prompt: PromptOptions = attrs.field(factory=PromptOptions, kw_only=True)
    model: ModelOptions | str | None = attrs.field(default=None, kw_only=True)


@attrs.frozen
class _GraphRequest(_EBMRequest):
    """The fields of a request that describes graphs.

    Attributes
    ----------
    graph: GraphOptions
        The options of ``EBMGraph.from_ebm`` and ``EBMGraph.to_text``.

    """

    graph: GraphOptions = attrs.field(factory=GraphOptions, kw_only=True)


@attrs.frozen
class DescribeGraphRequest(_GraphRequest):
    """The input of ``describe_graph``.

    Attributes
    ----------
    feature_index: int
        The index of the feature to describe.
    num_sentences: int
        The desired number of sentences for the description.

    """

    feature_index: int
    num_sentences: int = 7


@attrs.frozen
class DescribeEBMRequest(_GraphRequest):
    """The input of ``describe_ebm``.

    Attributes
    ----------
    num_sentences: int
        The desired number of sentences for the description.
    max_concurrency: int
        The maximum number of features described at the same time.
    progress_bar: bool
        Whether to show a progress bar on the error stream: one step for each feature and one for the summary.
        Without a terminal, the bar writes one line for each step.

    """

    num_sentences: int = 30
    max_concurrency: int = 8
    progress_bar: bool = attrs.field(default=False, kw_only=True)


@attrs.frozen
class ExplainInstanceRequest(_EBMRequest):
    """The input of ``explain_instance``.

    The EBM is a classifier or a regressor, without interactions.

    Attributes
    ----------
    instance: Mapping[str, FeatureValue]
        The value of every feature, by feature name.
    global_explanation: str
        The explanation of the whole model, computed once. For example, the output of ``describe_ebm``.
    audience: Audience | str
        Who reads the explanation: an ``Audience`` or the name of a built-in one. A layperson gets no model numbers; a
        data scientist gets scores, probabilities and confidence intervals. A subclass of ``Audience`` sets the facts
        and the instruction for any other reader.
    num_sentences: int
        The most sentences of each explanation.
    language: str
        The language of the explanations, as a name that the LLM knows (``"French"``, ``"Deutsch"``). The facts and the
        numbers stay as they are, and only the words around them change.
    local: LocalOptions
        The options of ``compute_local_facts``. For a regressor, ``decision`` sets the question about the predicted
        value that the explanations keep. The local prompts read ``expert_description`` and ``dataset_description`` of
        ``prompt``.

    """

    instance: Mapping[str, FeatureValue]
    global_explanation: str
    audience: Audience | str = "layperson"
    num_sentences: int = 5
    language: str = attrs.field(default="English", kw_only=True)
    local: LocalOptions = attrs.field(factory=LocalOptions, kw_only=True)
