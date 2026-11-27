"""The workloads of the LLM calls: each call in its sync form, the baseline, and in its async form.

The calls go to the local fake LLM server, so a round measures the work of EBaMl and of the BAML runtime around the
requests. The concurrency workloads delay each reply, so that the requests overlap as with a real provider.
"""

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence

import orjson
from tests.conftest import RecordedRequest, Reply

from benchmark.performance.workload_types import Context, Round, Workload
from ebaml import (
    DescribeEBMRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
    describe_ebm,
    describe_ebm_async,
    explain_instance,
    explain_instance_async,
    propose_difficulties,
    propose_difficulties_async,
)

# the delay of each reply of the fake LLM in the concurrency workloads
REPLY_DELAY_SECONDS = 0.01
DIFFICULTY_REPLY = [
    {"feature": "fare", "difficulty": "Easy", "reason": "A cheaper ticket is a choice."},
    {"feature": "age", "difficulty": "VeryHard", "reason": "Age cannot change."},
]
# the models of each call, with the rounds of the slow ones
DESCRIBE_MODELS: Mapping[str, int] = {
    "titanic": 5,
    "diabetes": 5,
    "multiclass": 5,
    "features-100-bins-1024": 3,
    "features-500-bins-32": 3,
}
EXPLAIN_MODELS: Sequence[str] = ("titanic", "diabetes", "multiclass")
CONCURRENCY_MODEL = "features-100-bins-32"
CONCURRENCY_LIMITS: Sequence[int] = (1, 4, 16)


def default_reply(_request: RecordedRequest) -> Reply:
    """Answer each request at once with the default reply of the fake LLM.

    Returns
    -------
    Reply
        The default reply.

    """
    return Reply()


def _delayed_reply(_request: RecordedRequest) -> Reply:
    time.sleep(REPLY_DELAY_SECONDS)
    return Reply()


def difficulty_reply(_request: RecordedRequest) -> Reply:
    """Answer each request with the difficulty of two features of the Titanic model.

    Returns
    -------
    Reply
        The JSON list that ``propose_difficulties`` reads.

    """
    return Reply(text=orjson.dumps(DIFFICULTY_REPLY).decode())


def _round[R](*, is_async: bool, sync: Callable[[], R], coroutine: Callable[[], object]) -> Round:
    """Make the round of one form of a call.

    Parameters
    ----------
    is_async: bool
        Whether the round runs the async form, in a new event loop.
    sync: Callable[[], R]
        Runs the sync form.
    coroutine: Callable[[], object]
        Makes the coroutine of the async form.

    Returns
    -------
    Round
        The round.

    """
    if is_async:
        return Round(lambda: asyncio.run(coroutine()))  # pyrefly: ignore[bad-argument-type] -- the factory returns a coroutine
    return Round(sync)


def _describe_ebm(
    model: str,
    *,
    is_async: bool,
    max_concurrency: int = 8,
    delayed: bool = False,
) -> Callable[[Context], Round]:
    """Measure ``describe_ebm`` or ``describe_ebm_async``.

    Parameters
    ----------
    model: str
        The name of the model.
    is_async: bool
        Whether to run the async form.
    max_concurrency: int
        The most graphs that the call describes at the same time.
    delayed: bool
        Whether the fake LLM waits before each reply.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        context.reply_with(_delayed_reply if delayed else default_reply)
        request = DescribeEBMRequest(context.model(model).ebm, model=context.llm, max_concurrency=max_concurrency)
        return _round(
            is_async=is_async,
            sync=lambda: describe_ebm(request),
            coroutine=lambda: describe_ebm_async(request),
        )

    return prepare


def _explain_instance(model: str, *, is_async: bool) -> Callable[[Context], Round]:
    """Measure ``explain_instance`` or ``explain_instance_async`` of the first row.

    Parameters
    ----------
    model: str
        The name of the model.
    is_async: bool
        Whether to run the async form.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        context.reply_with(default_reply)
        fitted = context.model(model)
        request = ExplainInstanceRequest(fitted.ebm, fitted.instance(), "A global explanation.", model=context.llm)
        return _round(
            is_async=is_async,
            sync=lambda: explain_instance(request),
            coroutine=lambda: explain_instance_async(request),
        )

    return prepare


def _propose_difficulties(model: str, *, is_async: bool) -> Callable[[Context], Round]:
    """Measure ``propose_difficulties`` or its async form for the first row.

    Parameters
    ----------
    model: str
        The name of the model. Its features must include ``fare`` and ``age``, which the fake reply rates.
    is_async: bool
        Whether to run the async form.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        context.reply_with(difficulty_reply)
        fitted = context.model(model)
        request = ProposeDifficultiesRequest(fitted.ebm, fitted.instance(), model=context.llm)
        return _round(
            is_async=is_async,
            sync=lambda: propose_difficulties(request),
            coroutine=lambda: propose_difficulties_async(request),
        )

    return prepare


def _forms(
    group: str,
    call: str,
    make: Callable[[bool], Callable[[Context], Round]],
    *,
    rounds: int = 5,
) -> list[Workload]:
    """Make the sync and the async workload of one call.

    Parameters
    ----------
    group: str
        The group of both workloads.
    call: str
        The call and its options, for example ``describe_ebm.titanic``.
    make: Callable[[bool], Callable[[Context], Round]]
        Builds the ``prepare`` function of a form: ``True`` for the async form.
    rounds: int
        The rounds of each workload.

    Returns
    -------
    list[Workload]
        The sync workload and then the async one.

    """
    return [
        Workload(f"{group}.{form}.{call}", group, make(form == "async"), rounds=rounds) for form in ("sync", "async")
    ]


def llm_workloads() -> list[Workload]:
    """Make the workloads of the LLM calls.

    Returns
    -------
    list[Workload]
        ``describe_ebm``, ``explain_instance`` and ``propose_difficulties`` in group ``llm``, and ``describe_ebm`` with
        several values of ``max_concurrency`` and a delay in each reply in group ``concurrency``.

    """
    workloads: list[Workload] = []
    for model, rounds in DESCRIBE_MODELS.items():
        workloads += _forms(
            "llm",
            f"describe_ebm.{model}",
            lambda is_async, model=model: _describe_ebm(model, is_async=is_async),
            rounds=rounds,
        )
    for model in EXPLAIN_MODELS:
        workloads += _forms(
            "llm",
            f"explain_instance.{model}",
            lambda is_async, model=model: _explain_instance(model, is_async=is_async),
        )
    workloads += _forms(
        "llm",
        "propose_difficulties.titanic",
        lambda is_async: _propose_difficulties("titanic", is_async=is_async),
    )
    for limit in CONCURRENCY_LIMITS:
        workloads += _forms(
            "concurrency",
            f"describe_ebm.max_concurrency-{limit}",
            lambda is_async, limit=limit: _describe_ebm(
                CONCURRENCY_MODEL,
                is_async=is_async,
                max_concurrency=limit,
                delayed=True,
            ),
            rounds=3,
        )
    return workloads
