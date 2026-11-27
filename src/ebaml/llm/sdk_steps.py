"""The steps that a pipeline yields to its driver, and the run that moves a pipeline from step to step.

A pipeline (``ebaml.llm.pipelines``) is a generator. It yields a step: an ``SDKCall`` (one call of a ``baml_sdk``
function) or a ``FanOut`` (several pipelines that run at the same time). A ``Driver`` runs the step, and a
``PipelineRun`` sends the reply into the generator, or throws the error of the step into it. The sync driver in
``ebaml.llm.descriptors`` calls ``baml_sdk.<name>_sync``, and the async driver in ``ebaml.llm.async_descriptors``
awaits ``baml_sdk.<name>``. So the logic of a pipeline exists once for both forms.

The module imports no ``baml_sdk``. The classes of ``baml_sdk`` that a pipeline builds reach it in ``SDKTypes``, and
the protocols below name the fields of the replies that a pipeline reads.
"""

import functools
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import attrs

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
class FanOut:
    """Several pipelines that describe one graph each, and that a driver runs at the same time.

    The driver replies with the result of each pipeline, in the order of ``pipelines``. If a pipeline fails, the driver
    cancels the pipelines that have not started and raises the error of the first pipeline that failed.

    Attributes
    ----------
    pipelines: Sequence[Pipeline[Described]]
        The pipelines, one for each graph.
    max_concurrency: int
        The maximum number of pipelines that run at the same time.
    finish_step: Callable[[], None]
        The driver calls it once for each pipeline that finishes, in one thread.

    """

    pipelines: Sequence[Generator["Step", SDKValue, Described]]
    max_concurrency: int
    finish_step: Callable[[], None]

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

    A driver runs ``step`` while it is not ``None``, then passes the reply to ``reply``, or the error to ``fail``. An
    error that the pipeline does not handle leaves ``reply`` or ``fail``. When ``step`` is ``None``, ``result`` gives
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

    def reply(self, value: SDKValue) -> None:
        """Send the reply of the step into the pipeline, and read its next step.

        Parameters
        ----------
        value: SDKValue
            The reply.

        """
        self._resume(functools.partial(self._pipeline.send, value))

    def fail(self, error: Exception) -> None:
        """Throw the error of the step into the pipeline, and read its next step.

        Parameters
        ----------
        error: Exception
            The error of the step. If the pipeline does not handle it, the method raises it.

        """
        self._resume(functools.partial(self._pipeline.throw, error))

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
