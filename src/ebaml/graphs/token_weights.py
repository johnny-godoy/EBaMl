"""The word, byte and digit counts of a text, and the weights that turn them into a token estimate.

The estimate of the tokens of a text is ``words * w_words + bytes * w_bytes + digits * w_digits``. Graph text is mostly
digits and punctuation, so a digit count separates it from prose better than a character count. The three counts grow
together, so a plain least-squares fit can give a weight below zero. ``fit_weights`` keeps every weight at zero or
above. The module needs no tokenizer, no file and no ``baml_sdk``.
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Self, TypedDict

import attrs
import numpy as np
import numpy.typing as npt
import orjson

from ebaml.core.exceptions import TokenFitError

_FEATURE_COUNT = 3


@attrs.frozen
class TextFeatures:
    """The counts of a text that the token estimate reads.

    Attributes
    ----------
    words: int
        The number of parts of the text that whitespace separates.
    bytes: int
        The number of bytes of the UTF-8 encoding of the text.
    digits: int
        The number of characters of the text that are digits.

    """

    words: int
    bytes: int
    digits: int

    @classmethod
    def of(cls, text: str) -> Self:
        """Count the words, bytes and digits of a text.

        Parameters
        ----------
        text: str
            The text.

        Returns
        -------
        TextFeatures
            The counts.

        Examples
        --------
        >>> TextFeatures.of("x: 12 é")
        TextFeatures(words=3, bytes=8, digits=2)

        """
        return cls(words=len(text.split()), bytes=len(text.encode()), digits=sum(char.isdigit() for char in text))

    def as_vector(self) -> tuple[float, ...]:
        """Return the counts in the order of ``TokenWeights.as_vector``.

        Returns
        -------
        tuple[float, ...]
            The words, the bytes and the digits.

        """
        return (float(self.words), float(self.bytes), float(self.digits))


# A weight is zero or above and finite: ``ge`` rejects NaN and a negative number, ``lt`` rejects infinity.
_WEIGHT = attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.lt(math.inf))


@attrs.frozen
class TokenWeights:
    """The tokens that one word, one byte and one digit add to the estimate of a text.

    Attributes
    ----------
    words: float
        The tokens for each word.
    bytes: float
        The tokens for each byte.
    digits: float
        The tokens for each digit.

    """

    words: float = attrs.field(validator=_WEIGHT)
    bytes: float = attrs.field(validator=_WEIGHT)
    digits: float = attrs.field(validator=_WEIGHT)

    def as_vector(self) -> tuple[float, ...]:
        """Return the weights in the order of ``TextFeatures.as_vector``.

        Returns
        -------
        tuple[float, ...]
            The weights of the words, the bytes and the digits.

        """
        return (self.words, self.bytes, self.digits)

    def estimate(self, features: TextFeatures) -> float:
        """Estimate the tokens of a text from its counts.

        Parameters
        ----------
        features: TextFeatures
            The counts of the text.

        Returns
        -------
        float
            The estimated number of tokens, before any margin.

        Examples
        --------
        >>> TokenWeights(words=2.0, bytes=0.5, digits=1.0).estimate(TextFeatures(words=3, bytes=10, digits=2))
        13.0

        """
        return sum(weight * count for weight, count in zip(self.as_vector(), features.as_vector(), strict=True))


@attrs.frozen
class TokenSample:
    """One real request: the counts of its text and the input tokens that the provider reported for it.

    Attributes
    ----------
    features: TextFeatures
        The counts of the text of the messages of the request.
    tokens: int
        The input tokens that the provider counted, with the cached tokens included. It is at least 1.

    """

    features: TextFeatures
    tokens: int = attrs.field(validator=attrs.validators.ge(1))


def fit_weights(samples: Sequence[TokenSample]) -> TokenWeights:
    """Fit the weights by least squares, with every weight at zero or above.

    The fit tries each set of features that can have a weight above zero. It solves the least-squares problem for the
    set, and keeps the best set whose weights are all at zero or above. With three features this is exact.

    Parameters
    ----------
    samples: Sequence[TokenSample]
        The samples to fit.

    Returns
    -------
    TokenWeights
        The weights with the least squared error on the samples.

    Raises
    ------
    TokenFitError
        If ``samples`` is empty.

    Examples
    --------
    >>> samples = [
    ...     TokenSample(TextFeatures(words=w, bytes=0, digits=d), 2 * w + d) for w, d in [(1, 5), (4, 1), (3, 3)]
    ... ]
    >>> [round(weight, 6) for weight in fit_weights(samples).as_vector()]
    [2.0, 0.0, 1.0]

    """
    if not samples:
        msg = "Fitting the token weights needs at least one sample."
        raise TokenFitError(msg)
    design = np.array([sample.features.as_vector() for sample in samples])
    target = np.array([float(sample.tokens) for sample in samples])
    best_error = float(np.sum(target**2))
    best = np.zeros(_FEATURE_COUNT)
    for size in range(1, _FEATURE_COUNT + 1):
        for support in itertools.combinations(range(_FEATURE_COUNT), size):
            columns = list(support)
            solution: npt.NDArray[np.float64] = np.linalg.lstsq(design[:, columns], target, rcond=None)[0]
            if np.any(solution < 0):
                continue
            error = float(np.sum((design[:, columns] @ solution - target) ** 2))
            if error < best_error:
                best_error = error
                best = np.zeros(_FEATURE_COUNT)
                best[columns] = solution
    return TokenWeights(words=float(best[0]), bytes=float(best[1]), digits=float(best[2]))


def relative_spread(weights: TokenWeights, samples: Sequence[TokenSample]) -> float:
    """Measure how far the estimates are from the counts, as the root mean square of the relative error.

    Parameters
    ----------
    weights: TokenWeights
        The weights to judge.
    samples: Sequence[TokenSample]
        The samples to judge them on.

    Returns
    -------
    float
        The root mean square of ``(estimate - tokens) / tokens``. It is 0.0 for no samples.

    Examples
    --------
    >>> relative_spread(TokenWeights(words=1.0, bytes=0.0, digits=0.0), [TokenSample(TextFeatures(1, 0, 0), 2)])
    0.5

    """
    if not samples:
        return 0.0
    errors = [(weights.estimate(sample.features) - sample.tokens) / sample.tokens for sample in samples]
    return math.sqrt(sum(error**2 for error in errors) / len(errors))


@attrs.frozen
class ModelKey:
    """The provider and the name of the model that answered a request.

    Attributes
    ----------
    provider: str
        The provider that the BAML runtime resolved, for example ``"openrouter"``.
    model: str
        The model at the provider, for example ``"openai/gpt-6-luna"``.

    """

    provider: str
    model: str


@attrs.frozen
class WeightsEntry:
    """The weights of one model, and what they are worth.

    Attributes
    ----------
    weights: TokenWeights
        The weights that fit the model.
    relative_spread: float
        The root mean square of the relative error of ``weights`` on the samples that fitted them.
    samples: int
        The number of samples that fitted ``weights``.
    update_allowed: bool
        Whether a learning estimator may change ``weights``. A model whose entry is final sets it to ``False``.

    """

    weights: TokenWeights
    relative_spread: float = attrs.field(validator=attrs.validators.ge(0.0))
    samples: int = attrs.field(validator=attrs.validators.ge(0))
    update_allowed: bool = True


class _WeightsRecord(TypedDict):
    """The JSON object of the weights of an entry."""

    words: float
    bytes: float
    digits: float


class _EntryRecord(TypedDict):
    """The JSON object of an entry."""

    weights: _WeightsRecord
    relative_spread: float
    samples: int
    update_allowed: bool


class _ModelRecord(_EntryRecord):
    """The JSON object of the entry of one model."""

    provider: str
    model: str


def _read_entry(record: _EntryRecord) -> WeightsEntry:
    """Read one entry of the JSON table.

    Parameters
    ----------
    record: _EntryRecord
        The JSON object of the entry.

    Returns
    -------
    WeightsEntry
        The entry.

    """
    weights = record["weights"]
    return WeightsEntry(
        weights=TokenWeights(words=weights["words"], bytes=weights["bytes"], digits=weights["digits"]),
        relative_spread=record["relative_spread"],
        samples=record["samples"],
        update_allowed=record["update_allowed"],
    )


def _write_model(key: ModelKey, entry: WeightsEntry) -> _ModelRecord:
    """Convert the entry of one model to its JSON object.

    Parameters
    ----------
    key: ModelKey
        The model.
    entry: WeightsEntry
        Its entry.

    Returns
    -------
    _ModelRecord
        The JSON object.

    """
    return {
        "provider": key.provider,
        "model": key.model,
        "weights": {"words": entry.weights.words, "bytes": entry.weights.bytes, "digits": entry.weights.digits},
        "relative_spread": entry.relative_spread,
        "samples": entry.samples,
        "update_allowed": entry.update_allowed,
    }


def _write_entry(entry: WeightsEntry) -> _EntryRecord:
    """Convert an entry to the JSON object that ``_read_entry`` reads.

    Parameters
    ----------
    entry: WeightsEntry
        The entry.

    Returns
    -------
    _EntryRecord
        The JSON object.

    """
    return {
        "weights": {"words": entry.weights.words, "bytes": entry.weights.bytes, "digits": entry.weights.digits},
        "relative_spread": entry.relative_spread,
        "samples": entry.samples,
        "update_allowed": entry.update_allowed,
    }


@attrs.frozen
class WeightsTable:
    """The default weights and the weights of the models that were fitted.

    Attributes
    ----------
    default: WeightsEntry
        The entry for a model that the table does not list.
    models: Mapping[ModelKey, WeightsEntry]
        The entry of each model that was fitted.

    """

    default: WeightsEntry
    models: Mapping[ModelKey, WeightsEntry] = attrs.field(factory=dict)

    def entry_for(self, key: ModelKey | None) -> WeightsEntry:
        """Look up the entry of a model.

        Parameters
        ----------
        key: ModelKey | None
            The model, or ``None`` when it is not known.

        Returns
        -------
        WeightsEntry
            The entry of the model, or ``default`` when the table does not list it.

        """
        return self.models.get(key, self.default) if key is not None else self.default

    def with_entry(self, key: ModelKey, entry: WeightsEntry) -> Self:
        """Add the entry of a model, or replace the one that the table has.

        Parameters
        ----------
        key: ModelKey
            The model.
        entry: WeightsEntry
            Its entry.

        Returns
        -------
        WeightsTable
            A new table. The table itself does not change.

        """
        return attrs.evolve(self, models={**self.models, key: entry})

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Read a table from its JSON text.

        Parameters
        ----------
        data: bytes
            The JSON text that ``to_bytes`` writes.

        Returns
        -------
        WeightsTable
            The table.

        Raises
        ------
        TokenFitError
            If the text is not JSON, or lacks a field of the table.

        """
        try:
            document = orjson.loads(data)
            models = {
                ModelKey(record["provider"], record["model"]): _read_entry(record) for record in document["models"]
            }
            return cls(default=_read_entry(document["default"]), models=models)
        except (orjson.JSONDecodeError, KeyError, TypeError) as error:
            msg = f"The table of token weights cannot be read: {error!r}."
            raise TokenFitError(msg) from error

    def to_bytes(self) -> bytes:
        """Write the table as JSON text, with the models in order, so that the text is the same for the same table.

        Returns
        -------
        bytes
            The JSON text.

        """
        ordered = sorted(self.models.items(), key=lambda item: (item[0].provider, item[0].model))
        models = list(itertools.starmap(_write_model, ordered))
        document = {"default": _write_entry(self.default), "models": models}
        return orjson.dumps(document, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
