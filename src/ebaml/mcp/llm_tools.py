"""The tools that ask the LLM to describe the model. The server registers them only when the user gives ``--llm``."""

from ebaml.llm.descriptors import describe_ebm, describe_graph
from ebaml.llm.options import ModelOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest
from ebaml.mcp.model_store import ModelStore


def describe_feature(store: ModelStore, llm: ModelOptions, feature: str) -> str:
    """Ask the LLM to describe the graph of one feature.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions
        The LLM.
    feature: str
        The name of the feature.

    Returns
    -------
    str
        The output of ``describe_graph``.

    """
    return describe_graph(DescribeGraphRequest(store.ebm, store.feature_index(feature), model=llm))


def describe_model(store: ModelStore, llm: ModelOptions) -> str:
    """Ask the LLM to describe the whole model. The first call computes the text and the next calls reuse it.

    Parameters
    ----------
    store: ModelStore
        The session.
    llm: ModelOptions
        The LLM.

    Returns
    -------
    str
        The output of ``describe_ebm``.

    """
    if store.model_description is None:
        store.model_description = describe_ebm(DescribeEBMRequest(store.ebm, model=llm))
    return store.model_description
