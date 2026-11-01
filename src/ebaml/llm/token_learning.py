"""Token estimators that learn from the token counts of real LLM calls, one for each model.

The BAML runtime reports the input tokens of each request and the model that answered. ``TokenLearner`` keeps one
``TokenEstimator`` for each model. It starts each from the shipped table of weights, and moves it toward the counts of
the requests when learning is on. Learning is off by default, because learned weights change the estimate, and so the
text of a graph, from one call to the next.

When learning is on, the weights persist between sessions in a JSON file. By default the file is
``$XDG_CACHE_HOME/ebaml/token_weights.json``, or ``~/.cache/ebaml/token_weights.json``. The learner reads and writes the
file only when learning is on. A model that the table marks as final (``update_allowed`` false) never learns.
"""

import functools
import importlib.resources
import os
import threading
from collections.abc import Iterable
from itertools import starmap
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import attrs
import orjson
import structlog

from ebaml.core.exceptions import TokenFitError
from ebaml.graphs.request_text import message_text
from ebaml.graphs.token_budget import EstimatorState, TokenEstimator
from ebaml.graphs.token_weights import ModelKey, TextFeatures, TokenSample, TokenWeights, WeightsTable

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

_TABLE_NAME = "token_weights.json"
_COVARIANCE_ENTRIES = 9
_WEIGHT_ENTRIES = 3


@runtime_checkable
class RequestSample(Protocol):
    """One counted request, as the BAML runtime reports it. ``baml_sdk.TokenSample`` is such a sample."""

    @property
    def provider(self) -> str:
        """The provider that answered."""
        ...

    @property
    def model(self) -> str:
        """The model that answered."""
        ...

    @property
    def body(self) -> str:
        """The JSON body of the request. It never holds a header."""
        ...

    @property
    def input_tokens(self) -> int:
        """The input tokens that the provider counted, with the cached tokens included."""
        ...


@functools.cache
def default_table() -> WeightsTable:
    """Read the table of weights that ships with the package.

    Returns
    -------
    WeightsTable
        The default weights and the weights of the models that were fitted.

    """
    return WeightsTable.from_bytes(importlib.resources.files(__package__).joinpath(_TABLE_NAME).read_bytes())


def default_weights_path() -> Path:
    """Name the file where learned weights persist when the learner gets no path.

    Returns
    -------
    Path
        ``$XDG_CACHE_HOME/ebaml/token_weights.json``. Without the variable, ``~/.cache/ebaml/token_weights.json``.

    """
    cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache) if cache else Path.home() / ".cache"
    return root / "ebaml" / _TABLE_NAME


class _StateRecord(TypedDict):
    """The JSON object of the learned state of one model."""

    provider: str
    model: str
    weights: list[float]
    covariance: list[float]
    samples_seen: int


def _state_to_record(key: ModelKey, state: EstimatorState) -> _StateRecord:
    """Convert the learned state of one model to a JSON object.

    Parameters
    ----------
    key: ModelKey
        The model.
    state: EstimatorState
        What its estimator learned.

    Returns
    -------
    _StateRecord
        The JSON object.

    """
    return {
        "provider": key.provider,
        "model": key.model,
        "weights": list(state.weights.as_vector()),
        "covariance": list(state.covariance),
        "samples_seen": state.samples_seen,
    }


def _record_to_state(record: _StateRecord) -> EstimatorState:
    """Read the learned state of one model from a JSON object.

    Parameters
    ----------
    record: _StateRecord
        The JSON object that ``_state_to_record`` wrote.

    Returns
    -------
    EstimatorState
        The state.

    Raises
    ------
    TokenFitError
        If the covariance does not have nine entries or the weights do not have three, or a weight is not valid.

    """
    if len(record["covariance"]) != _COVARIANCE_ENTRIES or len(record["weights"]) != _WEIGHT_ENTRIES:
        msg = f"A learned state needs {_WEIGHT_ENTRIES} weights and {_COVARIANCE_ENTRIES} covariance entries."
        raise TokenFitError(msg)
    words, bytes_, digits = record["weights"]
    return EstimatorState(
        TokenWeights(words, bytes_, digits),
        tuple(record["covariance"]),
        record["samples_seen"],
    )


@attrs.define(eq=False)
class TokenLearner:
    """Keep a token estimator for each model, and learn its weights from the counts of real requests.

    Pass a learner as ``GraphOptions(token_learner=...)``. The budget check of ``describe_graph`` and ``describe_ebm``
    then uses the estimator of the model that you chose, and the calls feed their token counts back to it. The BAML
    runtime names the model that answered, which can differ from the one that you chose (``openrouter/free`` picks a
    model for each request). So a learner learns the weights of each model that answered. Its estimate before a call
    uses the model that you chose, and the default weights when the table does not list it.

    Attributes
    ----------
    learning: bool
        Whether the estimators change their weights. It is off by default.
    persist: bool
        Whether learned weights persist in ``path``. It has an effect only when learning is on.
    path: str | os.PathLike[str] | None
        The file of the learned weights. ``None`` selects ``default_weights_path()``.
    table: WeightsTable
        The weights that each estimator starts from. The default is the table that ships with the package.
    prior_samples: float
        See ``TokenEstimator.prior_samples``.
    margin: float
        See ``TokenEstimator.margin``.

    """

    learning: bool = False
    persist: bool = True
    path: str | os.PathLike[str] | None = None
    table: WeightsTable = attrs.field(factory=default_table)
    prior_samples: float = 10.0
    margin: float = 1.0
    _estimators: dict[ModelKey, TokenEstimator] = attrs.field(init=False, factory=dict)
    _stored: dict[ModelKey, EstimatorState] | None = attrs.field(init=False, default=None)
    _lock: threading.RLock = attrs.field(init=False, factory=threading.RLock)

    @property
    def _file(self) -> Path:
        """The file of the learned weights."""
        return default_weights_path() if self.path is None else Path(self.path)

    @property
    def _uses_file(self) -> bool:
        """Whether the learner reads and writes the file."""
        return self.learning and self.persist

    # nosemgrep: ebaml-return-attrs-not-container -- the keys are the models that the file holds
    def _load(self) -> dict[ModelKey, EstimatorState]:
        """Read the learned weights from the file, once. A file that is missing or unreadable gives no state.

        Returns
        -------
        dict[ModelKey, EstimatorState]
            The stored state of each model.

        """
        if self._stored is None:
            stored: dict[ModelKey, EstimatorState] = {}
            try:
                records = orjson.loads(self._file.read_bytes())["models"]
                stored = {ModelKey(record["provider"], record["model"]): _record_to_state(record) for record in records}
            except FileNotFoundError:
                pass
            except (
                OSError,
                orjson.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:  # TokenFitError is a ValueError
                logger.warning("token_weights_unreadable", path=str(self._file), error=repr(error))
            self._stored = stored
        return self._stored

    def estimator_for(self, key: ModelKey | None) -> TokenEstimator:
        """Get the estimator of a model, and create it from the table on the first request.

        Parameters
        ----------
        key: ModelKey | None
            The model, or ``None`` when it is not known. ``None`` gets the default entry of the table.

        Returns
        -------
        TokenEstimator
            The estimator of the model. The learner keeps it, so later calls get the same object.

        """
        with self._lock:
            if key is not None and key in self._estimators:
                return self._estimators[key]
            entry = self.table.entry_for(key)
            estimator = TokenEstimator(
                weights=entry.weights,
                relative_spread=entry.relative_spread,
                margin=self.margin,
                learning=self.learning and entry.update_allowed,
                prior_samples=self.prior_samples,
            )
            if key is not None:
                if self._uses_file and key in self._load():
                    estimator.restore(self._load()[key])
                self._estimators[key] = estimator
            return estimator

    def observe(self, samples: Iterable[RequestSample]) -> int:
        """Teach the estimators the token counts of requests, and save what they learned.

        A sample of a model whose entry is final changes nothing. The method does nothing when learning is off.

        Parameters
        ----------
        samples: Iterable[RequestSample]
            The counted requests, for example ``counted.usage.samples``.

        Returns
        -------
        int
            The number of samples that changed an estimator.

        """
        if not self.learning:
            return 0
        applied = 0
        for sample in samples:
            text = message_text(sample.body)
            if not text:
                continue
            estimator = self.estimator_for(ModelKey(sample.provider, sample.model))
            token_sample = TokenSample(TextFeatures.of(text), sample.input_tokens)
            applied += estimator.observe(token_sample)
        if applied:
            self.save()
        return applied

    def save(self) -> None:
        """Write the learned weights to the file. Nothing happens when learning or persistence is off.

        A file that cannot be written logs a warning and does not stop the call.
        """
        if not self._uses_file:
            return
        with self._lock:
            states = {**self._load()}
            states.update({key: estimator.state() for key, estimator in self._estimators.items() if estimator.learning})
            self._stored = states
            document = {"models": list(starmap(_state_to_record, states.items()))}
            target = self._file
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(orjson.dumps(document, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
                temporary.replace(target)
            except OSError as error:
                logger.warning("token_weights_not_saved", path=str(target), error=repr(error))
