"""The types of the workloads of the performance benchmark: the shared context, one round and one workload."""

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path

import attrs
from tests.conftest import FakeLLM, RecordedRequest, Reply

from benchmark.performance.models import FittedModel, load_model
from ebaml import ModelOptions


@attrs.define
class Context:
    """What the workloads share in one run: the saved models, the fake LLM and a directory for files.

    Attributes
    ----------
    model_dir: Path
        The directory of the saved models.
    fake_llm: FakeLLM
        The state of the local fake LLM server.
    llm: ModelOptions
        The model that points at the fake LLM server.
    work_dir: Path
        A directory where a workload may write files, such as the save of an edited model.

    """

    model_dir: Path
    fake_llm: FakeLLM
    llm: ModelOptions
    work_dir: Path
    _models: dict[str, FittedModel] = attrs.field(factory=dict, init=False)
    _counter: Iterator[int] = attrs.field(factory=itertools.count, init=False)

    def model(self, name: str) -> FittedModel:
        """Load a model once for the whole run.

        Parameters
        ----------
        name: str
            The name of the recipe.

        Returns
        -------
        FittedModel
            The EBM and its training data.

        """
        if name not in self._models:
            self._models[name] = load_model(name, self.model_dir)
        return self._models[name]

    def reply_with(self, handler: Callable[[RecordedRequest], Reply]) -> None:
        """Set the replies of the fake LLM for the next workload, and forget the requests of the last one.

        Parameters
        ----------
        handler: Callable[[RecordedRequest], Reply]
            Chooses the reply to each request.

        """
        self.fake_llm.handler = handler
        self.fake_llm.requests.clear()

    def new_directory(self) -> Path:
        """Make an empty directory for one round.

        Returns
        -------
        Path
            The directory.

        """
        directory = self.work_dir / f"round-{next(self._counter)}"
        directory.mkdir(parents=True)
        return directory


@attrs.frozen
class Round:
    """One round of a workload.

    Attributes
    ----------
    run: Callable[..., object]
        The work that the round measures. It takes the arguments that ``setup`` returns.
    setup: Callable[[], tuple[tuple[object, ...], dict[str, object]]] | None
        Work before each round that the round does not measure, such as a new edit session for a tool that changes
        it. It returns the positional and the keyword arguments of ``run``. ``None`` runs ``run`` with no argument.

    """

    run: Callable[..., object]
    setup: Callable[[], tuple[tuple[object, ...], dict[str, object]]] | None = None


@attrs.frozen
class Workload:
    """One measurement of the benchmark.

    Attributes
    ----------
    name: str
        The name, unique in the table, as ``group.call.options``.
    group: str
        The group of the workload in the results.
    prepare: Callable[[Context], Round]
        Does the work that the rounds share, and returns the round.
    rounds: int
        The number of measured rounds. A slow workload has fewer rounds, so the run stays short.

    """

    name: str
    group: str
    prepare: Callable[[Context], Round]
    rounds: int = 5
