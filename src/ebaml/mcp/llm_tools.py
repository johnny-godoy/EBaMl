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
    return describe_graph(DescribeGraphRequest(store.read().model, store.feature_index(feature), model=llm))


def describe_model(store: ModelStore, llm: ModelOptions) -> str:
    """Ask the LLM to describe the whole model at the head. The store keeps the text for the revision that it describes.

    The LLM call runs on a revision with no lock. If an edit moves the head during the call, the store drops the text,
    so the next call describes the new head.

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
    read = store.read()
    kept = store.described(read.info.revision)
    if kept is not None:
        return kept
    text = describe_ebm(DescribeEBMRequest(read.model, model=llm)).text
    store.keep_description(read.info.revision, text)
    return text
