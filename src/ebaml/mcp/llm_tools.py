"""The tools that ask the LLM to describe the model. The server registers them only when the user gives ``--llm``.

Each tool awaits the async form of its LLM call, so the server serves other tool calls while the LLM answers.
"""

from collections.abc import Mapping
from typing import Self

import attrs

from ebaml.core.custom_types import FeatureValue
from ebaml.llm.async_descriptors import describe_ebm_async, describe_graph_async, propose_difficulties_async
from ebaml.llm.difficulties import FeatureDifficulty
from ebaml.llm.ebm_description import EBMDescription
from ebaml.llm.options import GraphOptions, ModelOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ProposeDifficultiesRequest
from ebaml.mcp.explain_tools import BASE_STRATEGIES, CostStrategyName, instance_values
from ebaml.mcp.model_store import ModelStore


@attrs.frozen
class FailedFeature:
    """A feature whose description failed, as the output of ``describe_model`` names it.

    Attributes
    ----------
    feature: str
        The name of the feature.
    error_type: str
        The class of the error, for example ``RateLimitedError``.
    message: str
        The message of the error, with the steps that fix it.

    """

    feature: str
    error_type: str
    message: str


@attrs.frozen
class ModelDescription:
    """The output of ``describe_model``: the description of the model and the parts that it does not cover.

    Attributes
    ----------
    text: str
        The description.
    skipped_terms: list[str]
        The interaction terms that the description leaves out.
    skipped_features: list[str]
        The features that have no graph, so the description leaves them out.
    failed_features: list[FailedFeature]
        The features whose description failed. Call the tool again to describe them.

    """

    text: str
    skipped_terms: list[str]
    skipped_features: list[str]
    failed_features: list[FailedFeature]

    @classmethod
    def of(cls, description: EBMDescription) -> Self:
        """Build the output from the result of ``describe_ebm``.

        Parameters
        ----------
        description: EBMDescription
            The result.

        Returns
        -------
        Self
            The output, with the class and the message of each error.

        """
        failed = [
            FailedFeature(failure.feature, type(failure.error).__name__, str(failure.error))
            for failure in description.failed_features
        ]
        return cls(description.text, description.skipped_terms, description.skipped_features, failed)


@attrs.frozen
class DifficultyAdvice:
    """The difficulties that the LLM proposed, and the arguments that pass them to ``explain_prediction``.

    Attributes
    ----------
    difficulties: list[FeatureDifficulty]
        The difficulty, the multiplier and the reason of each input that the reply rates.
    difficulty: dict[str, float]
        The multiplier of each rated input: the argument ``difficulty`` of ``explain_prediction``.
    cost_strategy: CostStrategyName
        The base under the multipliers: the argument ``cost_strategy`` of ``explain_prediction``.
    difficulty_from_model: bool
        Always true: pass it to ``explain_prediction``, so that its texts call the difficulties guesses.

    """

    difficulties: list[FeatureDifficulty]
    difficulty: dict[str, float]
    cost_strategy: CostStrategyName
    difficulty_from_model: bool = True


async def describe_feature(store: ModelStore, llm: ModelOptions, feature: str, graph: GraphOptions) -> str:
    """Ask the LLM to describe the graph of one feature.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions
        The LLM.
    feature: str
        The name of the feature.
    graph: GraphOptions
        The options of the graph, with the mode and the fallback.

    Returns
    -------
    str
        The output of ``describe_graph``.

    """
    request = DescribeGraphRequest(store.read().model, store.feature_index(feature), graph=graph, model=llm)
    return await describe_graph_async(request)


async def describe_model(store: ModelStore, llm: ModelOptions, graph: GraphOptions) -> ModelDescription:
    """Ask the LLM to describe the whole model at the head. The store keeps the result of each mode for the revision.

    The LLM call runs on a revision with no lock. If an edit moves the head during the call, the store drops the result,
    so the next call describes the new head. The store keeps no result with failed features, so the next call asks
    again.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions
        The LLM.
    graph: GraphOptions
        The options of the graphs, with the mode and the fallback.

    Returns
    -------
    ModelDescription
        The description, the parts that it leaves out, and the features that failed.

    """
    read = store.read()
    description = store.described(read.info.revision, graph.mode)
    if description is None:
        description = await describe_ebm_async(DescribeEBMRequest(read.model, graph=graph, model=llm))
        if not description.failed_features:
            store.keep_description(read.info.revision, description, graph.mode)
    return ModelDescription.of(description)


async def propose_feature_difficulties(
    store: ModelStore,
    llm: ModelOptions,
    instance: Mapping[str, FeatureValue] | None,
    row: int | None,
    cost_strategy: CostStrategyName,
) -> DifficultyAdvice:
    """Ask the LLM how hard the change of each input of an instance is, at the head of the session.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions
        The LLM.
    instance: Mapping[str, FeatureValue] | None
        The value of every feature, by name.
    row: int | None
        The position of a row of the data. Give exactly one of ``instance`` and ``row``.
    cost_strategy: CostStrategyName
        The base cost under the proposed multipliers.

    Returns
    -------
    DifficultyAdvice
        The proposal, and the arguments that pass it to ``explain_prediction``.

    """
    values = instance_values(store, instance, row)
    request = ProposeDifficultiesRequest(store.read().model, values, model=llm, base=BASE_STRATEGIES[cost_strategy])
    proposal = await propose_difficulties_async(request)
    return DifficultyAdvice(proposal.difficulties, dict(proposal.costs.multipliers), cost_strategy)
