"""Retry an LLM call with a smaller graph when the reply is cut off.

This module holds the policy only. It imports no LLM client, so the caller passes the call, the truncation test, the
token estimate and the graph rebuild. ``ebaml.descriptors`` passes the real ``baml_sdk`` functions.
"""

from collections.abc import Callable

import structlog

from ebaml.exceptions import EBMGraphSimplificationError, EBMGraphTokenLimitError

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

# each retry after a truncated reply shrinks the graph text to this share of its previous estimated size
RETRY_SHRINK = 0.75
MAX_TRUNCATION_RETRIES = 4


def call_with_shrinking_graph[R, E: Exception](  # ruff: ignore[too-many-arguments] -- one argument per injected dependency
    call: Callable[[str], R],
    graph: str,
    *,
    error_type: type[E],
    is_truncated: Callable[[E], bool],
    estimate_tokens: Callable[[str], int],
    rebuild: Callable[[int], str],
    label: str,
) -> R:
    """Call the LLM with a graph text. If the reply is cut off, ask again with a smaller graph, up to 4 times.

    Each retry asks ``rebuild`` for a graph of 75% of the previous estimated size. The function raises the error of the
    LLM call if the error is not a truncation, if 4 retries failed, or if the graph cannot shrink.

    Parameters
    ----------
    call: Callable[[str], R]
        Sends the graph text to the LLM and returns the reply.
    graph: str
        The text of the graph for the first call.
    error_type: type[E]
        The class of the error that ``call`` raises when the LLM call fails.
    is_truncated: Callable[[E], bool]
        Tells whether an error means that the reply hit the ``max_tokens`` limit.
    estimate_tokens: Callable[[str], int]
        Estimates the number of tokens of a text.
    rebuild: Callable[[int], str]
        Returns the graph text again for a token budget. It raises ``EBMGraphSimplificationError`` or
        ``EBMGraphTokenLimitError`` when the graph cannot fit.
    label: str
        The name of the feature, for the log.

    Returns
    -------
    R
        The reply of the first call that succeeds.

    """
    retries_left = MAX_TRUNCATION_RETRIES
    while True:
        try:
            return call(graph)
        except error_type as error:
            if not is_truncated(error) or not retries_left:
                raise
            retries_left -= 1
            # like TalkToEBM, answer a rejected prompt by simplifying the graph further and asking again
            budget = int(estimate_tokens(graph) * RETRY_SHRINK)
            logger.warning("truncated_reply", feature=label, retry=MAX_TRUNCATION_RETRIES - retries_left)
            try:
                graph = rebuild(budget)
            except (EBMGraphSimplificationError, EBMGraphTokenLimitError):
                raise error from None
