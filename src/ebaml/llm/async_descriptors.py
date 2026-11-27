"""Describe EBM graphs and models using LLMs: the async form of each call.

Each function runs its pipeline with the async driver of ``ebaml.llm.drivers``, which awaits the async functions of
``baml_sdk``. An application that runs an event loop (a web server, an agent framework, a notebook) then serves other
work while the LLM answers. The sync functions of ``ebaml.llm.descriptors`` run the same pipelines, so both forms give
the same result for the same request.

The steps that need no LLM run in the event loop and block it while they run: the extraction and the text of the
graphs, the images of the vision mode, and the facts of a local explanation. These steps take less than a second for a
usual model.
"""

from ebaml.llm.difficulties import DifficultyProposal
from ebaml.llm.drivers import SDK_TYPES, run_async
from ebaml.llm.ebm_description import EBMDescription
from ebaml.llm.pipelines import (
    LocalExplanation,
    describe_ebm_pipeline,
    describe_graph_pipeline,
    explain_instance_pipeline,
    propose_difficulties_pipeline,
)
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)


async def describe_graph_async(request: DescribeGraphRequest) -> str:
    """Ask the LLM to describe a graph, and await the reply. The async form of ``describe_graph``.

    The function sends the same requests as ``describe_graph`` and returns the same description. The extraction and the
    text of the graph run in the event loop and block it while they run.

    Parameters
    ----------
    request: DescribeGraphRequest
        The request, as ``describe_graph`` takes it.

    Returns
    -------
    str
        The text that ``describe_graph`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import DescribeGraphRequest, describe_graph_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> print(asyncio.run(describe_graph_async(DescribeGraphRequest(ebm, feature_index=0))))
    A fake reply, which stands in for the text of the LLM.

    """
    return await run_async(describe_graph_pipeline(request, SDK_TYPES))


async def describe_ebm_async(request: DescribeEBMRequest) -> EBMDescription:
    """Ask the LLM to describe an EBM, and await the replies. The async form of ``describe_ebm``.

    The function describes the graphs as tasks of one task group, with at most ``max_concurrency`` requests at the same
    time, and then asks for the summary. It returns the same description as ``describe_ebm``, with the same partial
    results: a feature whose request fails on its own goes into ``failed_features``, and an error that affects every
    request stops the call at once. A cancelled call cancels the requests of the graphs that have not finished. The
    extraction and the text of the graphs run in the event loop and block it while they run.

    Parameters
    ----------
    request: DescribeEBMRequest
        The request, as ``describe_ebm`` takes it. ``max_concurrency`` limits the requests in flight, and
        ``progress_bar`` shows one step for each feature and one for the summary.

    Returns
    -------
    EBMDescription
        The description that ``describe_ebm`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import DescribeEBMRequest, describe_ebm_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> description = asyncio.run(describe_ebm_async(DescribeEBMRequest(ebm, max_concurrency=2)))
    >>> print(description.text)
    A fake reply, which stands in for the text of the LLM.
    >>> description.failed_features
    []

    """
    return await run_async(describe_ebm_pipeline(request, SDK_TYPES))


async def explain_instance_async(request: ExplainInstanceRequest) -> LocalExplanation:
    """Explain one prediction of an EBM, and await the reply. The async form of ``explain_instance``.

    The function computes the same facts and sends the same request as ``explain_instance``. The facts are computed in
    the event loop and block it while they run.

    Parameters
    ----------
    request: ExplainInstanceRequest
        The request, as ``explain_instance`` takes it.

    Returns
    -------
    LocalExplanation
        The explanation that ``explain_instance`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import ExplainInstanceRequest, explain_instance_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> request = ExplainInstanceRequest(ebm, X.iloc[0].to_dict(), "The model predicts survival.")
    >>> print(asyncio.run(explain_instance_async(request)).counterfactual)
    A fake reply, which stands in for the text of the LLM.

    """
    return await run_async(explain_instance_pipeline(request, SDK_TYPES))


async def propose_difficulties_async(request: ProposeDifficultiesRequest) -> DifficultyProposal:
    """Ask the LLM how hard each change of an input is, and await the reply. The async form of ``propose_difficulties``.

    The function sends the same request as ``propose_difficulties`` and returns the same proposal.

    Parameters
    ----------
    request: ProposeDifficultiesRequest
        The request, as ``propose_difficulties`` takes it.

    Returns
    -------
    DifficultyProposal
        The proposal that ``propose_difficulties`` returns for the request.

    Examples
    --------
    >>> import asyncio
    >>> from interpret.glassbox import ExplainableBoostingClassifier
    >>> from ebaml import ProposeDifficultiesRequest, propose_difficulties_async
    >>> ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=1).fit(X, y)
    >>> proposal = asyncio.run(propose_difficulties_async(ProposeDifficultiesRequest(ebm, X.iloc[0].to_dict())))
    >>> [(item.feature, item.difficulty.value) for item in proposal.difficulties]
    [('fare', 'Easy'), ('age', 'VeryHard')]

    """
    return await run_async(propose_difficulties_pipeline(request, SDK_TYPES))
