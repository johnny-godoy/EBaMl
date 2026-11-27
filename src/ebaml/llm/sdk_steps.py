"""The steps that a pipeline yields to its driver, and the run that moves a pipeline from step to step.

A pipeline (``ebaml.llm.pipelines``) is a generator. It yields a step: an ``SDKCall`` (one call of a ``baml_sdk``
function) or a ``FanOut`` (several pipelines that run at the same time). A ``Driver`` runs the step, and a
``PipelineRun`` sends the reply into the generator, or throws the error of the step into it. The sync driver of
``ebaml.llm.drivers`` calls ``baml_sdk.<name>_sync``, and its async driver awaits ``baml_sdk.<name>``. So the logic of
a pipeline exists once for both forms.

The module imports no ``baml_sdk``. The classes of ``baml_sdk`` that a pipeline builds reach it in ``SDKTypes``, and
the protocols below name the fields of the replies that a pipeline reads.
"""

import functools
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import attrs

from ebaml.core.exceptions import LLMRequestError
from ebaml.llm.token_learning import RequestSample

# A value of a class of baml_sdk, such as GraphPromptOptions, or a reply of a step. The module imports no baml_sdk,
# so it cannot name the class. A pipeline passes such a value on, and reads only the fields that a protocol names.
type SDKValue = Any


@runtime_checkable
class TokenCounts(Protocol):
    """The tokens that the provider counted for a reply, as ``baml_sdk.TokenUsage`` holds them."""

    @property
    def input_tokens(self) -> int:
        """The tokens of the prompt."""
        ...

    @property
    def output_tokens(self) -> int:
        """The tokens of the reply."""
        ...


@runtime_checkable
class CountedUsage(TokenCounts, Protocol):
    """The tokens of a reply and the requests that the provider counted, as ``baml_sdk.TokenUsage`` holds them."""

    @property
    def samples(self) -> Sequence[RequestSample]:
        """One sample for each request that the provider counted."""
        ...


@runtime_checkable
class CountedText(Protocol):
    """A reply and its token usage, as ``baml_sdk.Counted`` holds them."""

    @property
    def text(self) -> str:
        """The text of the reply."""
        ...

    @property
    def usage(self) -> CountedUsage:
        """The tokens that the provider counted."""
        ...


@runtime_checkable
class ModelIdentity(Protocol):
    """The provider and the model of a request, as ``baml_sdk.ModelSpec`` holds them."""

    @property
    def provider(self) -> str:
        """The provider, for example ``"openrouter"``."""
        ...

    @property
    def model(self) -> str:
        """The model of the provider."""
        ...


@runtime_checkable
class Driver[T](Protocol):
    """Runs the steps of a pipeline: the sync driver returns each reply, and the async driver an awaitable of it."""

    def call(self, call: "SDKCall") -> T:
        """Call the function of ``baml_sdk`` that the step names."""
        ...

    def fan_out(self, fan_out: "FanOut") -> T:
        """Run the pipelines of the step at the same time."""
        ...


@attrs.frozen
class SDKCall:
    """One call of a function of ``baml_sdk``.

    Attributes
    ----------
    name: str
        The name of the BAML function, for example ``"describe_graph_cot"``. The sync driver calls
        ``baml_sdk.<name>_sync``, and the async driver awaits ``baml_sdk.<name>``.
    arguments: Mapping[str, SDKValue]
        The keyword arguments of the call.

    """

    name: str
    arguments: Mapping[str, SDKValue]

    def dispatch[T](self, driver: Driver[T]) -> T:
        """Let the driver run the call.

        Parameters
        ----------
        driver: Driver[T]
            The driver of the pipeline.

        Returns
        -------
        T
            The reply of the call, or for the async driver the awaitable of the reply.

        """
        return driver.call(self)


@attrs.frozen
class Described:
    """The description of one graph, with its token usage and the number of attempts that were cut off."""

    counted: CountedText
    truncated_attempts: int


@attrs.frozen
class Outcome:
    """The outcome of one step: its reply, or its error.

    Attributes
    ----------
    reply: SDKValue
        The reply of the step, or ``None`` if the step failed.
    error: Exception | None
        The error of the step, or ``None`` if the step succeeded.

    """

    reply: SDKValue = None
    error: Exception | None = None

    @classmethod
    def of(cls, step: Callable[[], SDKValue]) -> "Outcome":
        """Run a step and keep its reply or its error.

        Parameters
        ----------
        step: Callable[[], SDKValue]
            Runs the step and gives its reply.

        Returns
        -------
        Outcome
            The reply or the error.

        """
        try:
            return cls(reply=step())
        except Exception as error:  # ruff: ignore[blind-except] -- the pipeline or the FanOut decides which errors stop the call
            return cls(error=error)

    @classmethod
    async def of_awaitable(cls, step: Awaitable[SDKValue]) -> "Outcome":
        """Await a step and keep its reply or its error.

        Parameters
        ----------
        step: Awaitable[SDKValue]
            The step, which gives its reply.

        Returns
        -------
        Outcome
            The reply or the error.

        """
        try:
            return cls(reply=await step)
        except Exception as error:  # ruff: ignore[blind-except] -- the pipeline or the FanOut decides which errors stop the call
            return cls(error=error)


@attrs.frozen
class FeatureOutcome:
    """The outcome of one graph pipeline of a ``FanOut``: its description, or the error of its feature.

    Attributes
    ----------
    described: Described | None
        The description, or ``None`` if the pipeline failed.
    error: LLMRequestError | None
        The error that concerns this feature only, or ``None`` if the pipeline succeeded.

    """

    described: Described | None = None
    error: LLMRequestError | None = None


@attrs.frozen
class FanOut:
    """Several pipelines that describe one graph each, and that a driver runs at the same time.

    The driver replies with the ``FeatureOutcome`` of each pipeline, in the order of ``pipelines``. It passes the error
    of a pipeline to ``settle``. If ``settle`` raises, the error concerns every request: the driver cancels the
    pipelines that have not started and raises that error.

    Attributes
    ----------
    pipelines: Sequence[Pipeline[Described]]
        The pipelines, one for each graph.
    max_concurrency: int
        The maximum number of pipelines that run at the same time.
    finish_step: Callable[[], None]
        The driver calls it once for each pipeline that finishes or fails, in one thread.
    settle: Callable[[Exception], LLMRequestError]
        Turns the error of one pipeline into the error of its feature, or raises it again if it stops the whole call.

    """

    pipelines: Sequence[Generator["Step", SDKValue, Described]]
    max_concurrency: int
    finish_step: Callable[[], None]
    settle: Callable[[Exception], LLMRequestError]

    def feature_outcome(self, outcome: Outcome) -> FeatureOutcome:
        """Turn the outcome of one pipeline into the outcome of its feature.

        Parameters
        ----------
        outcome: Outcome
            The description that the pipeline returned, or its error.

        Returns
        -------
        FeatureOutcome
            The description, or the error of the feature. ``settle`` raises an error that stops the whole call.

        """
        if outcome.error is not None:
            return FeatureOutcome(error=self.settle(outcome.error))
        return FeatureOutcome(described=outcome.reply)

    def dispatch[T](self, driver: Driver[T]) -> T:
        """Let the driver run the pipelines.

        Parameters
        ----------
        driver: Driver[T]
            The driver of the pipeline.

        Returns
        -------
        T
            The results, or for the async driver the awaitable of the results.

        """
        return driver.fan_out(self)


# A pipeline is a Generator[Step, SDKValue, R]: it yields steps, receives the replies, and returns its result R. The
# driver sends each reply of a step into the pipeline, and the type of a reply depends on the step. The hints spell the
# generator out, because beartype reads no alias in the return hint of a generator function.
type Step = SDKCall | FanOut


@attrs.define
class PipelineRun[R]:
    """One run of a pipeline, which a driver moves on with the outcome of each step.

    A driver runs ``step`` while it is not ``None``, then passes its ``Outcome`` to ``advance``. An error that the
    pipeline does not handle leaves ``advance``. When ``step`` is ``None``, ``result`` gives
    the value that the pipeline returned. A driver closes the run when it stops (``contextlib.closing``), so that a
    run that stops early, for example when its task is cancelled, closes the pipeline.

    Attributes
    ----------
    step: Step | None
        The step that the driver runs next, or ``None`` when the pipeline returned.

    """

    _pipeline: Generator[Step, SDKValue, R]
    step: Step | None = attrs.field(init=False, default=None)
    _result: R = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        """Run the pipeline up to its first step."""
        self._resume(self._pipeline.__next__)

    def _resume(self, resume: Callable[[], Step]) -> None:
        try:
            self.step = resume()
        except StopIteration as stop:
            self.step = None
            self._result = stop.value

    def advance(self, outcome: Outcome) -> None:
        """Send the reply of the step into the pipeline, or throw its error into it, and read the next step.

        Parameters
        ----------
        outcome: Outcome
            The reply or the error of the step. If the pipeline does not handle the error, the method raises it.

        """
        if outcome.error is not None:
            self._resume(functools.partial(self._pipeline.throw, outcome.error))
        else:
            self._resume(functools.partial(self._pipeline.send, outcome.reply))

    def result(self) -> R:
        """Give the value that the pipeline returned.

        Returns
        -------
        R
            The value.

        """
        return self._result

    def close(self) -> None:
        """Close the pipeline. A pipeline that returned stays as it is."""
        self._pipeline.close()


@attrs.frozen
class SDKTypes:
    """The classes of ``baml_sdk`` that the pipelines build. The driver passes them, so the pipelines import no SDK.

    Attributes
    ----------
    model_spec: Callable[..., SDKValue]
        ``baml_sdk.ModelSpec``.
    local_fact_texts: Callable[..., SDKValue]
        ``baml_sdk.LocalFactTexts``.

    """

    model_spec: Callable[..., SDKValue]
    local_fact_texts: Callable[..., SDKValue]
