"""The kind of a fitted EBM: binary classifier, multiclass classifier or regressor.

Each kind is one class. A class tells the names of the outputs of the model, what a score means, and how the scores
become the output of the model. ``model_kind`` reads the kind from a fitted EBM, so other code asks the kind and does
not test ``link_`` or ``classes_`` itself.
"""

import abc
from collections.abc import Callable, Mapping
from typing import override

import attrs
import numpy as np
import numpy.typing as npt
import scipy

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import UnsupportedModelError

_BINARY_CLASSES = 2


class ModelKind(abc.ABC):
    """The kind of an EBM: the outputs of the model, what a score means, and how scores become the output."""

    @property
    @abc.abstractmethod
    def outputs(self) -> list[str]:
        """The names of the outputs: one for a binary classifier and a regressor, one per class for multiclass."""

    @property
    @abc.abstractmethod
    def score_meaning(self) -> str:
        """What a score of a graph means, as a sentence for the prompts."""

    @abc.abstractmethod
    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert the sum of the scores and the intercept to the output of the model.

        Parameters
        ----------
        scores: npt.ArrayLike
            The sum of the scores and the intercept, on the scale of the link. For a multiclass classifier, the last
            axis has one value per class.

        Returns
        -------
        npt.NDArray[np.float64]
            The probability of the positive class, the probability of each class, or the predicted value.

        """


@attrs.frozen
class BinaryClassifier(ModelKind):
    """A classifier with two classes. A score is a change of the log-odds of the second class.

    Examples
    --------
    >>> kind = BinaryClassifier(("no", "yes"))
    >>> kind.outputs
    ['yes']
    >>> kind.predict(0.0)
    array(0.5)

    """

    classes: tuple[object, object]

    @property
    @override
    def outputs(self) -> list[str]:
        return [str(self.classes[1])]

    @property
    @override
    def score_meaning(self) -> str:
        return f"A score is the change of the log-odds of the class {self.classes[1]!r}."

    @override
    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(scipy.special.expit(np.asarray(scores, dtype=float)), dtype=np.float64)


@attrs.frozen
class MulticlassClassifier(ModelKind):
    """A classifier with three or more classes. Each bin has one score per class, and softmax gives the probabilities.

    Softmax does not change when the same constant is added to the scores of all classes, so only the differences
    between the classes of a bin carry meaning.

    Examples
    --------
    >>> kind = MulticlassClassifier(("a", "b", "c"))
    >>> kind.outputs
    ['a', 'b', 'c']
    >>> kind.predict([0.0, 0.0, 0.0]).round(3)
    array([0.333, 0.333, 0.333])

    """

    classes: tuple[object, ...]

    @property
    @override
    def outputs(self) -> list[str]:
        return [str(label) for label in self.classes]

    @property
    @override
    def score_meaning(self) -> str:
        return (
            "Each bin has one score per class. A score is the change of the log-odds of its class, centred over the"
            " classes of the bin: only the differences between the classes change the predicted probabilities."
        )

    @override
    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(scipy.special.softmax(np.asarray(scores, dtype=float), axis=-1), dtype=np.float64)


@attrs.frozen
class Regressor(ModelKind):
    """A regressor. A score is a change of the predicted value, and the output is the sum of the scores.

    Examples
    --------
    >>> Regressor().outputs
    ['prediction']
    >>> Regressor().predict(2.5)
    array(2.5)

    """

    @property
    @override
    def outputs(self) -> list[str]:
        return ["prediction"]

    @property
    @override
    def score_meaning(self) -> str:
        return "A score is the change of the predicted value, in the units of the target."

    @override
    def predict(self, scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(scores, dtype=np.float64)


def _classes(ebm: ExplainableBoostingModel) -> tuple[object, ...]:
    """Read the class labels of a classifier as Python values.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.

    Returns
    -------
    tuple[object, ...]
        The labels, in the order of the scores.

    Raises
    ------
    UnsupportedModelError
        If the EBM has no classes.

    """
    if not hasattr(ebm, "classes_"):
        msg = "The EBM has a classification link but no classes_."
        raise UnsupportedModelError(msg)
    return tuple(np.asarray(ebm.classes_).tolist())


def _binary(ebm: ExplainableBoostingModel) -> ModelKind:
    classes = _classes(ebm)
    if len(classes) != _BINARY_CLASSES:
        msg = f"An EBM with the logit link needs two classes. Got {len(classes)}."
        raise UnsupportedModelError(msg)
    return BinaryClassifier((classes[0], classes[1]))


def _multiclass(ebm: ExplainableBoostingModel) -> ModelKind:
    classes = _classes(ebm)
    if len(classes) <= _BINARY_CLASSES:
        msg = f"An EBM with the mlogit link needs three or more classes. Got {len(classes)}."
        raise UnsupportedModelError(msg)
    return MulticlassClassifier(classes)


def _regressor(_ebm: ExplainableBoostingModel) -> ModelKind:
    return Regressor()


# the kind of each link of interpret; a new kind is a new class and a new row
_KINDS: Mapping[str, Callable[[ExplainableBoostingModel], ModelKind]] = {
    "logit": _binary,
    "mlogit": _multiclass,
    "identity": _regressor,
}


def model_kind(ebm: ExplainableBoostingModel) -> ModelKind:
    """Read the kind of a fitted EBM from its link function.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.

    Returns
    -------
    ModelKind
        ``BinaryClassifier``, ``MulticlassClassifier`` or ``Regressor``.

    Raises
    ------
    UnsupportedModelError
        If the EBM is not fitted, has a link that EBaMl cannot read, or its classes do not fit its link.

    """
    link = getattr(ebm, "link_", None)
    if link not in _KINDS:
        msg = f"EBaMl reads the links {sorted(_KINDS)} of a fitted EBM. Got {link!r}."
        raise UnsupportedModelError(msg)
    return _KINDS[link](ebm)
