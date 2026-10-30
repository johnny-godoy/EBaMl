"""The token budget of a text: an estimate of its tokens, compared with a limit.

No tokenizer runs before the LLM call, so the estimate is about 3 characters per token. The provider error stays the
real limit. ``estimate_tokens`` and ``check_token_budget`` in ``baml_src/talk_to_ebm.baml`` use the same rule, and a
test keeps the two sides equal. This module needs no ``baml_sdk``, so code that calls no LLM can check a budget too.
"""

from typing import Literal

import attrs

_CHARS_PER_TOKEN = 3


@attrs.frozen
class BudgetStatus:
    """The estimated tokens of a text, and how they compare with the budget.

    Attributes
    ----------
    estimated_tokens: int
        The estimated number of tokens of the text.
    status: Literal["ok", "warn", "over"]
        ``"ok"`` within ``max_tokens``, ``"warn"`` above it, and ``"over"`` beyond ``overbudget_percent`` above it.

    """

    estimated_tokens: int
    status: Literal["ok", "warn", "over"]


def estimate_tokens(text: str) -> int:
    """Estimate the tokens of a text at 3 characters per token, rounded up.

    Parameters
    ----------
    text: str
        The text.

    Returns
    -------
    int
        The estimated number of tokens.

    Examples
    --------
    >>> estimate_tokens("abcd")
    2

    """
    return -(-len(text) // _CHARS_PER_TOKEN)


def check_token_budget(*, text: str, max_tokens: int, overbudget_percent: int) -> BudgetStatus:
    """Compare the estimated tokens of a text with a budget and with the tolerance above it.

    The function is a ``BudgetCheck``, so ``EBMGraph.to_text`` accepts it as ``check_budget``.

    Parameters
    ----------
    text: str
        The text.
    max_tokens: int
        The budget.
    overbudget_percent: int
        How far above ``max_tokens``, in percent, the estimate may go before the status is ``"over"``.

    Returns
    -------
    BudgetStatus
        The estimate and the status.

    Examples
    --------
    >>> check_token_budget(text="a" * 300, max_tokens=80, overbudget_percent=50)
    BudgetStatus(estimated_tokens=100, status='warn')

    """
    estimated = estimate_tokens(text)
    if estimated > max_tokens * (100 + overbudget_percent) // 100:
        return BudgetStatus(estimated, "over")
    if estimated > max_tokens:
        return BudgetStatus(estimated, "warn")
    return BudgetStatus(estimated, "ok")
