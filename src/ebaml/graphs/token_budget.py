"""The token budget of a text: an estimate of its tokens, compared with a limit.

No tokenizer runs before the LLM call, so the estimate comes from the counts of the text. ``TokenEstimator`` turns the
words, bytes and digits of a text into a number of tokens, and adds a margin, so that an estimate that is too low is
rarer than one that is too high. A text that the estimate calls too large only gets a simpler graph, but a text that it
calls small enough can make the provider refuse the request. The provider error stays the real limit. The module needs
no ``baml_sdk``, so code that calls no LLM can check a budget too.

The estimator can learn. ``TokenEstimator.observe`` moves the weights toward the token counts of real requests, by
recursive least squares with a prior at the starting weights. Learning is off by default, because learned weights change
the estimate, and so the graph text, from one call to the next.
"""

import itertools
import math
import threading
from typing import Literal, Self

import attrs
import numpy as np
import numpy.typing as npt

from ebaml.graphs.token_weights import TextFeatures, TokenSample, TokenWeights

# The weights fitted on 100 graph texts of one current model (graph text only, tokens counted by the provider, with the
# tokens of the prompt template taken away). ``scripts/fit_token_weights.py`` refits them.
DEFAULT_WEIGHTS: TokenWeights = TokenWeights(words=4.1195141, bytes=0.0, digits=0.2923085)
# The root mean square of the relative error of the default weights on the samples that fitted them.
DEFAULT_RELATIVE_SPREAD: float = 0.1416
# The scale of one typical graph (words, bytes, digits). The learning divides the counts by it, so that a prior of the
# same strength fits all three weights.
_FEATURE_SCALE: npt.NDArray[np.float64] = np.array([500.0, 5000.0, 2500.0])
_FEATURE_COUNT: int = 3


def _project_nonnegative(
    coefficients: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Find the nearest point with no negative entry, where distance follows the covariance of the estimate.

    A plain clip at zero would leave the other entries as they are, although a weight that is wrong at zero shifts the
    best value of the weights that correlate with it. The projection sets the entries of an active set to zero and moves
    the others by the covariance between them. With three entries it tries every active set, so the result is exact.

    Parameters
    ----------
    coefficients: npt.NDArray[np.float64]
        The unconstrained coefficients.
    covariance: npt.NDArray[np.float64]
        The covariance matrix of the least squares. It must be positive definite.

    Returns
    -------
    npt.NDArray[np.float64]
        The coefficients with no negative entry that are nearest to ``coefficients``.

    """
    if np.all(coefficients >= 0):
        return coefficients
    best = np.zeros(_FEATURE_COUNT)
    best_cost = math.inf
    for size in range(1, _FEATURE_COUNT + 1):
        for active_set in itertools.combinations(range(_FEATURE_COUNT), size):
            active = list(active_set)
            free = [index for index in range(_FEATURE_COUNT) if index not in active]
            inverse_active = np.linalg.inv(covariance[np.ix_(active, active)])
            candidate = np.zeros(_FEATURE_COUNT)
            candidate[free] = (
                coefficients[free] - covariance[np.ix_(free, active)] @ inverse_active @ coefficients[active]
            )
            if np.any(candidate < 0):
                continue
            cost = float(coefficients[active] @ inverse_active @ coefficients[active])
            if cost < best_cost:
                best, best_cost = candidate, cost
    return best


@attrs.frozen
class BudgetStatus:
    """The estimated tokens of a text, and how they compare with the budget.

    Attributes
    ----------
    estimated_tokens: int
        The estimated number of tokens of the text, with the margin.
    status: Literal["ok", "warn", "over"]
        ``"ok"`` within ``max_tokens``, ``"warn"`` above it, and ``"over"`` beyond ``overbudget_percent`` above it.

    """

    estimated_tokens: int
    status: Literal["ok", "warn", "over"]


@attrs.frozen
class EstimatorState:
    """What a learning estimator knows after its samples. ``TokenEstimator.restore`` takes it back.

    Attributes
    ----------
    weights: TokenWeights
        The learned weights.
    covariance: tuple[float, ...]
        The nine entries of the 3 by 3 matrix of the recursive least squares, row by row. It holds how sure the
        estimator is about each weight.
    samples_seen: int
        The number of samples that the estimator learned from.

    """

    weights: TokenWeights
    covariance: tuple[float, ...]
    samples_seen: int


@attrs.define(eq=False)
class TokenEstimator:
    """Estimate the tokens of a text, and compare the estimate with a budget.

    An instance is a ``BudgetCheck``, so ``EBMGraph.to_text(check_budget=estimator)`` works. The estimate is an upper
    bound: the fit plus ``margin`` times ``relative_spread`` of the fit.

    Attributes
    ----------
    weights: TokenWeights
        The starting weights, and the center of the prior of the learning.
    relative_spread: float
        The root mean square of the relative error of ``weights``. It sets the size of the margin.
    margin: float
        How many spreads the estimate lies above the fit.
    learning: bool
        Whether ``observe`` changes the weights. It is off by default.
    prior_samples: float
        How many samples of a typical graph the prior of ``weights`` is worth. A larger value makes a sample move the
        weights less.

    Examples
    --------
    >>> estimator = TokenEstimator()
    >>> estimator.check_budget(text="1.5 2.5 " * 10, max_tokens=100, overbudget_percent=50).status
    'warn'

    """

    weights: TokenWeights = DEFAULT_WEIGHTS
    relative_spread: float = attrs.field(default=DEFAULT_RELATIVE_SPREAD, validator=attrs.validators.ge(0.0))
    margin: float = attrs.field(default=1.0, validator=attrs.validators.ge(0.0))
    learning: bool = False
    prior_samples: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))
    _current: TokenWeights = attrs.field(init=False)
    _covariance: npt.NDArray[np.float64] = attrs.field(init=False)
    _samples_seen: int = attrs.field(init=False, default=0)
    _lock: threading.Lock = attrs.field(init=False, factory=threading.Lock)

    def __attrs_post_init__(self) -> None:
        """Start the learning at the prior."""
        self._current = self.weights
        self._covariance = np.eye(_FEATURE_COUNT) / self.prior_samples

    @property
    def current_weights(self) -> TokenWeights:
        """The weights that the estimate uses now. They equal ``weights`` until a sample changes them."""
        return self._current

    @property
    def samples_seen(self) -> int:
        """The number of samples that changed the weights."""
        return self._samples_seen

    def estimate(self, text: str) -> int:
        """Estimate the tokens of a text, with the margin, rounded up.

        Parameters
        ----------
        text: str
            The text.

        Returns
        -------
        int
            The estimated number of tokens. It is 0 for a text of no words.

        Examples
        --------
        >>> TokenEstimator(weights=TokenWeights(words=2.0, bytes=0.0, digits=0.0), relative_spread=0.0).estimate(
        ...     "a b c"
        ... )
        6

        """
        point = self._current.estimate(TextFeatures.of(text))
        return math.ceil(point * (1 + self.margin * self.relative_spread))

    def check_budget(self, *, text: str, max_tokens: int, overbudget_percent: int) -> BudgetStatus:
        """Compare the estimated tokens of a text with a budget and with the tolerance above it.

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

        """
        estimated = self.estimate(text)
        if estimated > max_tokens * (100 + overbudget_percent) // 100:
            return BudgetStatus(estimated, "over")
        if estimated > max_tokens:
            return BudgetStatus(estimated, "warn")
        return BudgetStatus(estimated, "ok")

    def __call__(self, *, text: str, max_tokens: int, overbudget_percent: int) -> BudgetStatus:
        """Do the same as ``check_budget``, so that an estimator is a ``BudgetCheck``.

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

        """
        return self.check_budget(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent)

    def observe(self, sample: TokenSample) -> bool:
        """Move the weights toward the token count of one real request, if learning is on.

        The step is one of recursive least squares with the prior at ``weights``. A projection then keeps every
        weight at zero or above. The method is safe to call from several threads.

        Parameters
        ----------
        sample: TokenSample
            The counts of the text of a request and the input tokens that the provider reported for it.

        Returns
        -------
        bool
            Whether the sample changed the weights. It is ``False`` when learning is off.

        Examples
        --------
        >>> TokenEstimator().observe(TokenSample(TextFeatures(words=10, bytes=50, digits=5), 60))
        False
        >>> learner = TokenEstimator(learning=True)
        >>> learner.observe(TokenSample(TextFeatures(words=10, bytes=50, digits=5), 60))
        True

        """
        if not self.learning:
            return False
        scaled = np.array(sample.features.as_vector()) / _FEATURE_SCALE
        with self._lock:
            coefficients = np.array(self._current.as_vector()) * _FEATURE_SCALE
            gain_numerator = self._covariance @ scaled
            gain = gain_numerator / (1.0 + scaled @ gain_numerator)
            coefficients += gain * (sample.tokens - coefficients @ scaled)
            covariance = self._covariance - np.outer(gain, gain_numerator)
            self._covariance = (covariance + covariance.T) / 2
            coefficients = _project_nonnegative(coefficients, self._covariance)
            self._current = TokenWeights(*(coefficients / _FEATURE_SCALE).tolist())
            self._samples_seen += 1
        return True

    def state(self) -> EstimatorState:
        """Take a copy of what the estimator learned.

        Returns
        -------
        EstimatorState
            The learned weights, the matrix of the recursive least squares and the number of samples.

        """
        with self._lock:
            return EstimatorState(
                weights=self._current,
                covariance=tuple(self._covariance.ravel().tolist()),
                samples_seen=self._samples_seen,
            )

    def restore(self, state: EstimatorState) -> Self:
        """Take back what ``state`` made, for example after a restart.

        Parameters
        ----------
        state: EstimatorState
            A value that ``state`` returned.

        Returns
        -------
        TokenEstimator
            The estimator itself.

        """
        with self._lock:
            self._current = state.weights
            self._covariance = np.array(state.covariance).reshape(_FEATURE_COUNT, _FEATURE_COUNT)
            self._samples_seen = state.samples_seen
        return self


# The estimator behind ``estimate_tokens`` and ``check_token_budget``. It never learns, so these functions give the same
# number for the same text in every call.
_DEFAULT_ESTIMATOR: TokenEstimator = TokenEstimator()


def estimate_tokens(text: str) -> int:
    """Estimate the tokens of a text with the default weights, with the margin, rounded up.

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
    >>> estimate_tokens("0.1 0.2 0.3")
    17

    """
    return _DEFAULT_ESTIMATOR.estimate(text)


def check_token_budget(*, text: str, max_tokens: int, overbudget_percent: int) -> BudgetStatus:
    """Compare the estimated tokens of a text with a budget and with the tolerance above it.

    The function uses the default weights and never learns. It is a ``BudgetCheck``, so ``EBMGraph.to_text`` accepts it
    as ``check_budget``. Pass a ``TokenEstimator`` instead to choose the weights or to learn them.

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
    >>> check_token_budget(text="1.5 2.5 " * 10, max_tokens=100, overbudget_percent=50)
    BudgetStatus(estimated_tokens=108, status='warn')

    """
    return _DEFAULT_ESTIMATOR.check_budget(text=text, max_tokens=max_tokens, overbudget_percent=overbudget_percent)
