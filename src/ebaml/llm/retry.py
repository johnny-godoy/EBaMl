"""Retry an LLM call with a smaller graph when the reply is cut off.

This module holds the policy only. It imports no LLM client, so the caller passes the call, the truncation test, the
token estimate and the graph rebuild. The call is a generator pipeline (``ebaml.llm.pipelines``): it yields the steps
of the request, and a driver runs them. So one retry serves the sync and the async forms.
"""

from collections.abc import Callable, Generator
from typing import Any

import attrs
import structlog

from ebaml.core.exceptions import EBMGraphSimplificationError, EBMGraphTokenLimitError

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

# each retry after a truncated reply shrinks the graph text to this share of its previous estimated size
RETRY_SHRINK = 0.75
MAX_TRUNCATION_RETRIES = 4


@attrs.frozen
class ShrinkingGraphCall[S, R, E: Exception, G]:
    """The call and the services that ``call_with_shrinking_graph`` needs.

    ``G`` is the type of the graph that a call sends: its text, or the text with the graph that it renders. ``S`` is the
    type of the steps that the call yields to its driver.

    Attributes
    ----------
    call: Callable[[G], Generator[S, Any, R]]
        Returns the pipeline that sends the graph to the LLM. The pipeline returns the reply.
    graph: G
        The graph for the first call.
    error_type: type[E]
        The class of the error that ``call`` raises when the LLM call fails.
    is_truncated: Callable[[E], bool]
        Tells whether an error means that the reply hit the ``max_tokens`` limit.
    estimate_tokens: Callable[[G], int]
        Estimates the number of tokens of the text of a graph.
    rebuild: Callable[[int], G]
        Returns the graph again for a token budget. It raises ``EBMGraphSimplificationError`` or
        ``EBMGraphTokenLimitError`` when the graph cannot fit.
    on_truncated: Callable[[E], None]
        Called with the error of each attempt that was cut off, the last one included, before the retry or the raise.
        The caller counts the attempts with it, because the error carries no token usage.
    label: str
        The name of the feature, for the log.

    """

    # the driver sends each reply of a step into the pipeline, and the type of a reply depends on the step
    call: Callable[[G], Generator[S, Any, R]]
    graph: G
    error_type: type[E]
    is_truncated: Callable[[E], bool]
    estimate_tokens: Callable[[G], int]
    rebuild: Callable[[int], G]
    on_truncated: Callable[[E], None]
    label: str


def call_with_shrinking_graph[S, R, E: Exception, G](
    request: ShrinkingGraphCall[S, R, E, G],
) -> Generator[S, Any, R]:
    """Call the LLM with a graph text. If the reply is cut off, ask again with a smaller graph, up to 4 times.

    Each retry asks ``request.rebuild`` for a graph of 75% of the previous estimated size. The function raises the
    error of the LLM call if the error is not a truncation, if 4 retries failed, or if the graph cannot shrink.

    Parameters
    ----------
    request: ShrinkingGraphCall[S, R, E, G]
        The call, the first graph, and the services that the retry uses.

    Returns
    -------
    R
        The reply of the first call that succeeds.

    Yields
    ------
    S
        The steps of each call, for the driver of the pipeline.

    """
    graph = request.graph
    retries_left = MAX_TRUNCATION_RETRIES
    while True:
        try:
            return (yield from request.call(graph))
        except request.error_type as error:
            if not request.is_truncated(error):
                raise
            request.on_truncated(error)
            if not retries_left:
                raise
            retries_left -= 1
            # like TalkToEBM, answer a rejected prompt by simplifying the graph further and asking again
            budget = int(request.estimate_tokens(graph) * RETRY_SHRINK)
            logger.warning("truncated_reply", feature=request.label, retry=MAX_TRUNCATION_RETRIES - retries_left)
            try:
                graph = request.rebuild(budget)
            except (EBMGraphSimplificationError, EBMGraphTokenLimitError):
                raise error from None
