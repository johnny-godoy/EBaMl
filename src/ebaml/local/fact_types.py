"""The result classes and the options of the local facts of one EBM prediction."""

import attrs

from ebaml.core.custom_types import ClassLabel
from ebaml.local.covering_solvers import CoveringSolver, MipCover
from ebaml.local.decisions import Decision
from ebaml.local.local_kinds import OutputKind
from ebaml.local.minimality import MinimalityStatus


@attrs.frozen
class LocalOptions:
    """Options of the local facts.

    Attributes
    ----------
    confidence_level: float
        Confidence level of the claims. ``0`` ignores the uncertainty of the scores.
    max_steps: int
        The most features to reveal one by one in the count-completions steps.
    n_samples: int
        Samples used to estimate the share of completions.
    seed: int
        Seed of the sampling, so that the facts are reproducible.
    min_support: float
        Share of the training data a value needs to be offered as an alternative in the counterfactual and as a
        possible value of an unknown feature in the sufficient reason: a category needs this share, and the lowest and
        highest ``min_support`` of a continuous feature are trimmed. It keeps rare values, with extreme scores, out of
        the explanations.
    decision: Decision | None
        The question about the predicted value of a regressor that the explanations keep, such as ``Threshold(10)`` or
        ``Interval(10, 20)``. ``None`` asks whether the value is above the intercept, the average prediction. A
        classifier has its predicted class as the decision, so it accepts ``None`` only.
    search: CoveringSolver
        How to solve the covering problems: the sufficient reason of a decision with several margins, as "inside an
        interval" has, and the counterfactual of ``target_class``. ``MipCover()`` (the default) searches all sets in a
        MIP solver of OR-Tools (HiGHS), and proves the smallest set unless its time limit ends first. ``GreedyCover``
        and ``LocalSearchCover`` need no MIP solver and prove less. A decision with one margin ignores this option,
        because sorting finds its smallest set exactly.
    target_class: ClassLabel | None
        For a multiclass classifier, the class that the counterfactual must reach. ``None`` (the default) asks for the
        fewest changes after which the class is no longer the predicted class. A class asks for the fewest changes
        after which every margin of that class holds at once, over every choice of bins. ``search`` finds the changes.
        Another kind of model accepts ``None`` only.

    """

    confidence_level: float = 0.95
    max_steps: int = 5
    n_samples: int = 10_000
    seed: int = 0
    min_support: float = 0.01
    decision: Decision | None = None
    search: CoveringSolver = attrs.field(factory=MipCover)
    target_class: ClassLabel | None = None


@attrs.frozen
class Prediction:
    """The prediction to explain, with the confidence interval of its output.

    The fields fit every model kind. ``docs/local-explanations-proof.md`` defines them for each kind.

    Attributes
    ----------
    decision: ClassLabel
        What the explanations keep or change: the predicted class of a classifier, or the answer of a regressor to its
        ``Decision``, such as ``"above 10"``.
    output: float
        The output of the model for the decision: the probability of the predicted class, or the predicted value.
    lower: float
        The lower end of the confidence interval of ``output``.
    upper: float
        The upper end of the confidence interval of ``output``.
    score: float
        The sum of the intercept and the scores, on the scale of the link: the log-odds of the second class, or the
        predicted value. For a multiclass classifier it is the score of the predicted class, with the scores of each
        bin centred over the classes as in the graphs (``EBMGraph.from_ebm``). It is then lower than
        ``decision_function`` by one number for all the classes, so the margins and the probabilities do not change.
    score_std: float
        The standard deviation of ``score``.
    margin: float
        The margin of the decision, positive while the decision holds: the log-odds of the predicted class, or the
        distance of the predicted value from the threshold, on the side of the decision. The margin means the same for
        every kind, and ``score`` keeps the scale of the link.
    threshold: float
        The score at which the decision stops to hold: ``0`` for the log-odds of a binary classifier, and the highest
        score of the other classes for a multiclass classifier. The score is above it when the decision says that the
        score is high.
    output_kind: OutputKind
        What ``output`` is.

    """

    decision: ClassLabel
    output: float
    lower: float
    upper: float
    score: float
    score_std: float
    margin: float
    threshold: float = 0.0
    output_kind: OutputKind = OutputKind.PROBABILITY


@attrs.frozen
class Contribution:
    """The effect of one input on the prediction."""

    feature: str
    value: str
    score: float
    std: float


@attrs.frozen
class Change:
    """One input to change in a counterfactual."""

    feature: str
    from_value: str
    to_value: str
    score_change: float


@attrs.frozen
class Counterfactual:
    """A set of changes after which the decision does not hold, at the requested confidence.

    The set is the smallest one only if ``minimality`` says so (``MinimalityStatus.is_smallest``). The default
    counterfactual always has that proof. The counterfactual of ``LocalOptions.target_class`` has it only if the search
    proves it: ``GreedyCover``, ``LocalSearchCover`` and a ``MipCover`` that ends at its time limit can return a larger
    set.

    Attributes
    ----------
    changes: list[Change]
        The changes.
    new_decision: ClassLabel
        The decision after the changes: the other class of a binary classifier, the class that a multiclass classifier
        then predicts, or the opposite answer of a regressor.
    new_output: float
        The output of the model for the new decision: the probability of the new class, or the predicted value.
    minimality: MinimalityStatus
        Why the set is, or is not, known to be the smallest. Sorting proves it for the default counterfactual
        (``SORTING_PROOF``). For a chosen target class, the search proves it (``SOLVER_PROOF``), or the size equals
        ``lower_bound`` (``AT_LOWER_BOUND``), or the search proves nothing (``UNPROVED``).
    lower_bound: int
        No set of fewer changes gives the new decision. It equals the number of changes when ``minimality`` is a
        proof of sorting or of the solver.

    """

    changes: list[Change]
    new_decision: ClassLabel
    new_output: float
    minimality: MinimalityStatus
    lower_bound: int


@attrs.frozen
class SufficientReason:
    """A set of inputs that fixes the decision whatever the other inputs are.

    Attributes
    ----------
    features: list[str]
        The inputs.
    guaranteed_margin: float
        The worst case of the margin of the decision when these inputs keep their values.
    minimality: MinimalityStatus
        Why the set is, or is not, known to be the smallest. Sorting proves it for a decision with one margin
        (``SORTING_PROOF``). For several margins, the search proves it (``SOLVER_PROOF``), or the size equals
        ``lower_bound`` (``AT_LOWER_BOUND``), or the search proves nothing (``UNPROVED``).
    lower_bound: int
        No set of fewer inputs fixes the decision. It is the LP bound of the search for several margins, and the size of
        the set for one margin.

    """

    features: list[str]
    guaranteed_margin: float
    minimality: MinimalityStatus
    lower_bound: int


@attrs.frozen
class CompletionStep:
    """The share of completions with the same prediction when only ``fixed`` features are known."""

    fixed: list[str]
    share: float


@attrs.frozen
class LocalFacts:
    """Everything computed about one prediction, with the options that say under which conditions the facts hold.

    Attributes
    ----------
    prediction: Prediction
        The prediction to explain.
    contributions: list[Contribution]
        The effect of each input, largest first.
    counterfactual: Counterfactual | None
        The changes that flip the decision, or ``None`` if the search found none.
    sufficient_reason: SufficientReason | None
        The inputs that fix the decision, or ``None`` if the search found none.
    completions: list[CompletionStep]
        The estimated share of completions with the same prediction, for each step.
    confidence_level: float
        ``LocalOptions.confidence_level`` of the facts. At ``0`` no score moves, so no claim covers the uncertainty of
        the model.
    min_support: float
        ``LocalOptions.min_support`` of the facts. At ``0`` the counterfactual and the sufficient reason leave out no
        value.
    n_samples: int
        ``LocalOptions.n_samples`` of the facts: the size of the sample behind each share of ``completions``.

    """

    prediction: Prediction
    contributions: list[Contribution]
    counterfactual: Counterfactual | None
    sufficient_reason: SufficientReason | None
    completions: list[CompletionStep]
    confidence_level: float
    min_support: float
    n_samples: int
