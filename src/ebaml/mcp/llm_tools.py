"""The tools that ask the LLM to describe the model. The server registers them only when the user gives ``--llm``."""

from collections.abc import Mapping

import attrs

from ebaml.core.custom_types import FeatureValue
from ebaml.llm.descriptors import describe_ebm, describe_graph, propose_difficulties
from ebaml.llm.difficulties import FeatureDifficulty
from ebaml.llm.options import GraphOptions, ModelOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ProposeDifficultiesRequest
from ebaml.mcp.explain_tools import BASE_STRATEGIES, CostStrategyName, instance_values
from ebaml.mcp.model_store import ModelStore


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


def describe_feature(store: ModelStore, llm: ModelOptions, feature: str, graph: GraphOptions) -> str:
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
    return describe_graph(request)


def describe_model(store: ModelStore, llm: ModelOptions, graph: GraphOptions) -> str:
    """Ask the LLM to describe the whole model at the head. The store keeps the text of each mode for the revision.

    The LLM call runs on a revision with no lock. If an edit moves the head during the call, the store drops the text,
    so the next call describes the new head.

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
    str
        The output of ``describe_ebm``.

    """
    read = store.read()
    kept = store.described(read.info.revision, graph.mode)
    if kept is not None:
        return kept
    text = describe_ebm(DescribeEBMRequest(read.model, graph=graph, model=llm)).text
    store.keep_description(read.info.revision, text, graph.mode)
    return text


def propose_feature_difficulties(
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
    proposal = propose_difficulties(request)
    return DifficultyAdvice(proposal.difficulties, dict(proposal.costs.multipliers), cost_strategy)
