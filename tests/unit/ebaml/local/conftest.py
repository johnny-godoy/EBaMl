"""Helpers for the tests of the local explanations.

The first part serves the covering problems: a brute-force oracle and the strategies of Hypothesis. The worst cases
that the strategies draw are half-integers and the gains are integers, so no margin of a drawn problem is zero or tiny,
and the tolerances of the solvers cannot change an answer.

The second part serves the facts: random small EBMs, and an oracle of the local explanations that follows the proof
paper. The oracle computes the explanations from the definitions of ``proofs/papers/local-explanations/paper.typ`` by
enumeration. It reads the inputs of the paper only: the signed scores, the deviations, the plausible bins, the offset
of each margin and the number of deviations. It imports nothing from ``ebaml.local.facts``, so a test that compares
the two fails when the code stops computing what the paper proves (contract 1 of
``.claude/rules/semantic-contracts.md``). If the oracle and the code differ, report the difference. Do not change the
oracle to fit the code.

Every number of a small model is a multiple of a power of two, and the intercepts and the thresholds have offsets that
no sum of scores reaches. A margin of a drawn model is then never zero, and no sum has a rounding error.
"""

import copy
import itertools
import statistics
from collections.abc import Callable, Mapping, Sequence

import attrs
import numpy as np
import pytest
from hypothesis import strategies as st

from ebaml.core.custom_types import FeatureValue
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.local.covering_backend import CoveringModel, ModelResult, ModelStatus, SolveSettings
from ebaml.local.covering_problem import Choice, Cover, CoveringProblem
from ebaml.local.ortools_backend import OrToolsBackend
from tests.unit.ebaml.conftest import PandasFrame, frame_of_rows

# the parts of a problem: the worst case of each margin, and groups[group][alternative][margin]
type ProblemParts = tuple[list[float], list[list[list[float]]]]

WORST: st.SearchStrategy[float] = st.integers(min_value=-12, max_value=2).map(lambda whole: whole + 0.5)
GAIN: st.SearchStrategy[float] = st.integers(min_value=0, max_value=8).map(float)
SIGNED_GAIN: st.SearchStrategy[float] = st.integers(min_value=-4, max_value=8).map(float)
ELEMENT_COUNT = 5


@st.composite
def problem_parts(
    draw: st.DrawFn,
    *,
    max_groups: int = 6,
    max_alternatives: int = 3,
    signed: bool = True,
) -> ProblemParts:
    """Draw the parts of a covering problem.

    Hypothesis draws lists, not the attrs class, because beartype samples with ``random`` when it checks a class.

    Returns
    -------
    ProblemParts
        The worst cases and the groups.

    """
    margin_count = draw(st.integers(min_value=1, max_value=3))
    group_count = draw(st.integers(min_value=1, max_value=max_groups))
    gain = SIGNED_GAIN if signed else GAIN
    worst = [draw(WORST) for _ in range(margin_count)]
    groups = [
        [
            [draw(gain) for _ in range(margin_count)]
            for _ in range(draw(st.integers(min_value=1, max_value=max_alternatives)))
        ]
        for _ in range(group_count)
    ]
    return worst, groups


def oracle_size(problem: CoveringProblem) -> int | None:
    """Try every choice of at most one alternative from each group, by size.

    Returns
    -------
    int | None
        The fewest picks of a sufficient set, or None if no set is sufficient.

    """
    group_count = len(problem.groups)
    for size in range(group_count + 1):
        for chosen_groups in itertools.combinations(range(group_count), size):
            alternative_counts = [len(problem.groups[group]) for group in chosen_groups]
            ranges = [range(alternative_count) for alternative_count in alternative_counts]
            for alternatives in itertools.product(*ranges):
                choices = list(itertools.starmap(Choice, zip(chosen_groups, alternatives, strict=True)))
                if problem.sufficient(choices):
                    return size
    return None


def check_result(problem: CoveringProblem, cover: Cover) -> None:
    """Check the invariants of every result: one pick per group at most, sufficient, and a true guaranteed margin."""
    groups = [choice.group for choice in cover.choices]
    assert len(groups) == len(set(groups))
    assert problem.sufficient(cover.choices)
    assert cover.guaranteed_margin == min(problem.margins(cover.choices))
    assert cover.guaranteed_margin > 0
    assert cover.lower_bound <= len(cover.choices)


@st.composite
def set_covers(draw: st.DrawFn) -> list[frozenset[int]]:
    """Draw a family of sets over the elements ``0`` to ``ELEMENT_COUNT - 1``.

    Returns
    -------
    list[frozenset[int]]
        The elements of each set.

    """
    set_count = draw(st.integers(min_value=1, max_value=7))
    return [draw(st.frozensets(st.integers(min_value=0, max_value=ELEMENT_COUNT - 1))) for _ in range(set_count)]


def set_cover_problem(sets: Sequence[frozenset[int]]) -> CoveringProblem:
    """Build the problem of the reduction of Theorem 8 of ``proofs/EBaMlProofs/LocalExplanations/Hardness.lean``.

    A margin for each element with the worst case -1/2, and a group for each set with the gain 1 for its elements. A
    sufficient set is then a cover.

    Returns
    -------
    CoveringProblem
        The problem.

    """
    return CoveringProblem(
        [-0.5] * ELEMENT_COUNT,
        [[[1.0 if element in members else 0.0 for element in range(ELEMENT_COUNT)]] for members in sets],
    )


def smallest_cover(sets: Sequence[frozenset[int]]) -> int | None:
    """Return the fewest sets that cover every element, by brute force.

    Returns
    -------
    int | None
        The size, or None if no cover exists.

    """
    covers = [
        size
        for size in range(len(sets) + 1)
        for chosen in itertools.combinations(sets, size)
        if set().union(*chosen) >= set(range(ELEMENT_COUNT))
    ]
    return min(covers) if covers else None


def random_problem(seed: int, feature_count: int, margin_count: int) -> CoveringProblem:
    """Draw a monotone problem with real gains, too large for the oracle.

    Returns
    -------
    CoveringProblem
        One group for each feature, and worst cases that need a part of the gains.

    """
    random_generator = np.random.default_rng(seed)
    gains = random_generator.uniform(0, 1, size=(feature_count, margin_count)).round(3)
    worst = -gains.sum(axis=0) * random_generator.uniform(0.2, 0.4, size=margin_count)
    return CoveringProblem(worst.tolist(), [[row.tolist()] for row in gains])


@attrs.define
class ScriptedBackend:
    """A ``MipBackend`` that answers the LP with the real backend, and the integer models from a script.

    ``answer`` gets the model of each integer call and returns the result. It obeys the contract: a status, and picks
    only with a solution.
    """

    answer: Callable[[CoveringModel], ModelResult] | None = None
    lp: ModelResult | None = None
    calls: list[tuple[CoveringModel, SolveSettings]] = attrs.field(factory=list)

    def solve(self, model: CoveringModel, settings: SolveSettings) -> ModelResult:
        """Answer one model.

        Returns
        -------
        ModelResult
            The scripted answer, or the answer of ``OrToolsBackend`` when there is no script for the model.

        """
        if not model.integer:
            return self.lp if self.lp is not None else OrToolsBackend().solve(model, settings)
        self.calls.append((model, settings))
        if self.answer is not None:
            return self.answer(model)
        return OrToolsBackend().solve(model, settings)


# An LP that ends with no solution, so that the lower bound of a solver is the trivial one.
NO_LP: ModelResult = ModelResult(ModelStatus.NO_SOLUTION)

# The greedy picks {2}, then {0} and {1}, and its removal pass keeps all three. The LP bound is 2, and {1, 3} is the
# only set of size 2, so a solver must do better than the greedy.
GREEDY_IS_NOT_OPTIMAL: CoveringProblem = CoveringProblem(
    [-1.5, -2.5, -3.5],
    [[[2.0, 2.0, 0.0]], [[2.0, 0.0, 2.0]], [[1.0, 2.0, 3.0]], [[0.0, 3.0, 3.0]]],
)
# The greedy finds no set, because the first pick breaks the second margin. {0, 1} is the only set.
GREEDY_FINDS_NOTHING: CoveringProblem = CoveringProblem([-1.5, 0.5], [[[3.0, -5.0]], [[0.0, 5.0]]])


type FittedModel = ExplainableBoostingClassifier | ExplainableBoostingRegressor
type Templates = Mapping[tuple[str, tuple[int, ...]], FittedModel]

MODEL_KINDS: tuple[str, ...] = ("binary", "multiclass", "regressor")
# the number of bins of each feature: 2 to 4 features with 2 to 4 bins
MODEL_SHAPES: tuple[tuple[int, ...], ...] = ((2, 3), (3, 2, 4), (2, 3, 4, 3))
CLASS_COUNT = 3
CONFIDENCE_LEVELS: tuple[float, ...] = (0.0, 0.5, 0.95)
# no share of a drawn weight equals one of these, so no bin is on the edge of the support rule
MIN_SUPPORTS: tuple[float, ...] = (0.0, 0.07, 0.23)

_SCORE: st.SearchStrategy[float] = st.integers(min_value=-8, max_value=8).map(lambda quarters: quarters / 4)
_DEVIATION: st.SearchStrategy[float] = st.integers(min_value=0, max_value=4).map(lambda eighths: eighths / 8)
_WEIGHT: st.SearchStrategy[float] = st.integers(min_value=1, max_value=20).map(float)


def feature_name(feature_index: int) -> str:
    """Name a feature of a small model: the even features are nominal, the odd ones continuous.

    Returns
    -------
    str
        The name.

    """
    return f"n{feature_index}" if feature_index % 2 == 0 else f"x{feature_index}"


def bin_value(feature_index: int, bin_index: int) -> FeatureValue:
    """Return a value of a feature that falls in a bin.

    Returns
    -------
    FeatureValue
        The category of a nominal feature, or a number inside the bin of a continuous feature.

    """
    return f"c{bin_index}" if feature_index % 2 == 0 else float(bin_index)


def model_row(bins: Sequence[int]) -> dict[str, FeatureValue]:
    """Return the input of a small model whose features fall in the given bins.

    Returns
    -------
    dict[str, FeatureValue]
        The value of each feature, by name.

    """
    return {feature_name(index): bin_value(index, bin_index) for index, bin_index in enumerate(bins)}


@attrs.frozen
class SmallModel:
    """The numbers of one small EBM and of one instance.

    Attributes
    ----------
    kind: str
        One of ``MODEL_KINDS``.
    scores: list[list[list[float]]]
        ``scores[feature][bin][output]``: the score of each bin for each output of the model.
    deviations: list[list[list[float]]]
        The standard deviation of each score, in the same layout.
    weights: list[list[float]]
        The training weight of each bin of each feature.
    intercepts: list[float]
        The intercept of each output.
    instance_bins: list[int]
        The bin of the instance for each feature.

    """

    kind: str
    scores: list[list[list[float]]]
    deviations: list[list[list[float]]]
    weights: list[list[float]]
    intercepts: list[float]
    instance_bins: list[int]

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of bins of each feature."""
        return tuple(len(feature_weights) for feature_weights in self.weights)

    @property
    def instance(self) -> dict[str, FeatureValue]:
        """The instance, as ``compute_local_facts`` takes it."""
        return model_row(self.instance_bins)

    def output_scores(self, bins: Sequence[int]) -> list[float]:
        """Add up the score of each output for an input, from the drawn numbers.

        Returns
        -------
        list[float]
            The intercept plus the scores of the bins, for each output.

        """
        return [
            intercept + sum(self.scores[feature][bin_index][output] for feature, bin_index in enumerate(bins))
            for output, intercept in enumerate(self.intercepts)
        ]


@st.composite
def small_models(draw: st.DrawFn, kind: str) -> SmallModel:
    """Draw a small model of one kind, with an instance.

    Returns
    -------
    SmallModel
        The model.

    """
    shape = draw(st.sampled_from(MODEL_SHAPES))
    output_count = CLASS_COUNT if kind == "multiclass" else 1

    def table(value: st.SearchStrategy[float]) -> list[list[list[float]]]:
        return [[[draw(value) for _ in range(output_count)] for _ in range(bin_count)] for bin_count in shape]

    return SmallModel(
        kind=kind,
        scores=table(_SCORE),
        deviations=table(_DEVIATION),
        weights=[[draw(_WEIGHT) for _ in range(bin_count)] for bin_count in shape],
        # the offsets differ by 1/16 or 2/16 between two outputs, and a sum of scores is a multiple of 1/4
        intercepts=[draw(_SCORE) + (output + 1) / 16 for output in range(output_count)],
        instance_bins=[draw(st.integers(min_value=0, max_value=bin_count - 1)) for bin_count in shape],
    )


# The limits of the question of a regressor: none for "above the intercept", one for a threshold, two for an interval.
# The offsets 1/32 and 3/32 keep each limit apart from every predicted value, which is a multiple of 1/4 plus 1/16.
LIMITS: st.SearchStrategy[tuple[float, ...]] = st.one_of(
    st.just(()),
    st.integers(min_value=-12, max_value=12).map(lambda quarters: (quarters / 4 + 1 / 32,)),
    st.tuples(st.integers(min_value=-12, max_value=8), st.integers(min_value=0, max_value=8)).map(
        lambda low_and_width: (low_and_width[0] / 4 + 1 / 32, sum(low_and_width) / 4 + 3 / 32),
    ),
)


@st.composite
def model_cases(draw: st.DrawFn) -> tuple[SmallModel, tuple[float, ...]]:
    """Draw a small model of any kind, with the limits of its question.

    Returns
    -------
    tuple[SmallModel, tuple[float, ...]]
        The model, and the limits of the question of a regressor. Another kind has no limits.

    """
    kind = draw(st.sampled_from(MODEL_KINDS))
    return draw(small_models(kind)), draw(LIMITS) if kind == "regressor" else ()


def template_data(shape: Sequence[int]) -> PandasFrame:
    """Build the training data of a template: every combination of the bins, three times.

    Returns
    -------
    PandasFrame
        One column for each feature.

    """
    rows = list(itertools.product(*(range(bin_count) for bin_count in shape))) * 3
    return frame_of_rows([model_row(row) for row in rows])


def fit_template(fit_ebm: Callable[..., FittedModel], kind: str, shape: Sequence[int]) -> FittedModel:
    """Fit the EBM of one kind and shape. ``build_ebm`` then replaces its numbers.

    Returns
    -------
    FittedModel
        The fitted model.

    """
    data = template_data(shape)
    random_generator = np.random.default_rng(len(shape))
    targets = {
        "binary": random_generator.integers(0, 2, len(data)),
        "multiclass": random_generator.integers(0, CLASS_COUNT, len(data)),
        "regressor": random_generator.normal(size=len(data)),
    }
    return fit_ebm(data, targets[kind], interactions=0, random_state=0, regression=kind == "regressor")


@pytest.fixture(scope="session")
def small_templates(fit_ebm: Callable[..., FittedModel]) -> dict[tuple[str, tuple[int, ...]], FittedModel]:
    """Fit one EBM for each kind and shape of the strategy of small models.

    Returns
    -------
    dict[tuple[str, tuple[int, ...]], FittedModel]
        The fitted model of each kind and shape.

    """
    return {(kind, shape): fit_template(fit_ebm, kind, shape) for kind in MODEL_KINDS for shape in MODEL_SHAPES}


def build_ebm(templates: Templates, model: SmallModel) -> FittedModel:
    """Build the EBM of a small model: a copy of the template with the drawn numbers.

    Returns
    -------
    FittedModel
        The EBM. Its ``decision_function`` or ``predict`` gives ``model.output_scores``.

    """
    ebm = copy.deepcopy(templates[model.kind, model.shape])
    single_output = model.kind != "multiclass"
    for term, (scores, deviations, weights) in enumerate(
        zip(model.scores, model.deviations, model.weights, strict=True),
    ):
        # the first and the last entry of each array are the bins of a missing and of an unseen value
        ebm.term_scores_[term][1:-1] = np.array(scores)[:, 0] if single_output else np.array(scores)
        ebm.standard_deviations_[term][1:-1] = np.array(deviations)[:, 0] if single_output else np.array(deviations)
        ebm.bin_weights_[term][1:-1] = weights
    # interpret types the intercept of a regressor as a number and of a classifier as an array, so set it by name
    vars(ebm).update(intercept_=model.intercepts[0] if model.kind == "regressor" else np.array(model.intercepts))
    return ebm


def model_scores(ebm: FittedModel, rows: Sequence[Mapping[str, FeatureValue]]) -> np.ndarray:
    """Ask the EBM for the score of each output, for some inputs.

    Returns
    -------
    np.ndarray
        One row for each input and one column for each output.

    """
    frame = frame_of_rows(rows)
    scores = ebm.predict(frame) if isinstance(ebm, ExplainableBoostingRegressor) else ebm.decision_function(frame)
    return np.asarray(scores, dtype=float).reshape(len(rows), -1)


def deviation_count(confidence_level: float) -> float:
    """Return the number of deviations of a confidence level, as the paper defines it.

    Returns
    -------
    float
        The two-sided quantile of the normal distribution.

    """
    return statistics.NormalDist().inv_cdf((1 + confidence_level) / 2)


def plausible_bins(weights: Sequence[float], *, continuous: bool, min_support: float, instance_bin: int) -> list[int]:
    """List the plausible bins of a feature, by the rule of ``LocalOptions.min_support``.

    A category needs the share ``min_support`` of the training weight. A continuous feature loses the lowest and the
    highest ``min_support`` of the weight. The bin of the instance is always plausible.

    Returns
    -------
    list[int]
        The bins.

    """
    total = sum(weights)
    plausible = {instance_bin}
    weight_below = 0.0
    for bin_index, weight in enumerate(weights):
        weight_up_to = weight_below + weight
        if continuous:
            enough = weight_up_to > min_support * total and weight_below < (1 - min_support) * total
        else:
            enough = weight >= min_support * total
        if enough:
            plausible.add(bin_index)
        weight_below = weight_up_to
    return sorted(plausible)


@attrs.frozen
class OracleMargin:
    """One margin of a decision, in the notation of the paper. The decision holds while every margin is positive.

    Attributes
    ----------
    offset: float
        The part of the margin that no feature changes: the signed intercept, minus the signed threshold.
    signed_scores: list[list[float]]
        ``signed_scores[feature][bin]``: the term of each bin in the margin.
    deviations: list[list[float]]
        The standard deviation of each term.

    """

    offset: float
    signed_scores: list[list[float]]
    deviations: list[list[float]]


@attrs.frozen
class OracleProblem:
    """The inputs of the paper for one instance.

    Attributes
    ----------
    margins: list[OracleMargin]
        The margins of the decision.
    plausible: list[list[int]]
        The plausible bins of each feature.
    instance_bins: list[int]
        The bin of the instance for each feature.
    deviation_count: float
        The number of deviations that a score moves against a claim.

    """

    margins: list[OracleMargin]
    plausible: list[list[int]]
    instance_bins: list[int]
    deviation_count: float

    @property
    def feature_count(self) -> int:
        """The number of features."""
        return len(self.instance_bins)

    def _term(self, margin: OracleMargin, feature: int, bin_index: int, direction: int) -> float:
        moved = direction * self.deviation_count * margin.deviations[feature][bin_index]
        return margin.signed_scores[feature][bin_index] + moved

    def robust_margin(self, margin: OracleMargin, change: Mapping[int, int]) -> float:
        """Return the robust margin of a change: each changed feature has the pessimistic score of its new bin.

        Returns
        -------
        float
            The margin. The change flips the decision when it is below zero.

        """
        unchanged = sum(
            margin.signed_scores[feature][bin_index]
            for feature, bin_index in enumerate(self.instance_bins)
            if feature not in change
        )
        return margin.offset + unchanged + sum(self._term(margin, feature, new, 1) for feature, new in change.items())

    def flips(self, features: Sequence[int]) -> bool:
        """Tell whether some change of exactly these features makes one margin fail.

        Returns
        -------
        bool
            True if a choice of plausible bins gives a robust margin below zero.

        """
        choices = itertools.product(*(self.plausible[feature] for feature in features))
        return any(
            self.robust_margin(margin, dict(zip(features, bins, strict=True))) < 0
            for bins in choices
            for margin in self.margins
        )

    def worst_case_margin(self, fixed: Sequence[int]) -> float:
        """Return the lowest guaranteed margin over every completion of the free features and every margin.

        Returns
        -------
        float
            The margin. The fixed features are sufficient when it is above zero.

        """
        free = [feature for feature in range(self.feature_count) if feature not in fixed]
        lowest = []
        for margin in self.margins:
            known = sum(self._term(margin, feature, self.instance_bins[feature], -1) for feature in fixed)
            completions = itertools.product(*(self.plausible[feature] for feature in free))
            lowest.append(
                margin.offset
                + known
                + min(
                    sum(
                        self._term(margin, feature, bin_index, -1)
                        for feature, bin_index in zip(free, bins, strict=True)
                    )
                    for bins in completions
                ),
            )
        return min(lowest)

    def _smallest(self, accepts: Callable[[Sequence[int]], bool], first_size: int) -> int | None:
        for size in range(first_size, self.feature_count + 1):
            if any(accepts(features) for features in itertools.combinations(range(self.feature_count), size)):
                return size
        return None

    def smallest_flip(self) -> int | None:
        """Return the fewest features of a change that flips the decision.

        Returns
        -------
        int | None
            The size, or None if no change flips the decision.

        """
        return self._smallest(self.flips, 1)

    def smallest_sufficient(self) -> int | None:
        """Return the fewest fixed features that keep the decision for every completion.

        Returns
        -------
        int | None
            The size, or None if not even all the features are sufficient.

        """
        return self._smallest(lambda features: self.worst_case_margin(features) > 0, 0)


def _signed(model: SmallModel, sign: int, threshold: float) -> OracleMargin:
    return OracleMargin(
        offset=sign * (model.intercepts[0] - threshold),
        signed_scores=[[sign * outputs[0] for outputs in feature] for feature in model.scores],
        deviations=[[outputs[0] for outputs in feature] for feature in model.deviations],
    )


def oracle_margins(model: SmallModel, limits: Sequence[float] = ()) -> list[OracleMargin]:
    """Write the margins of the decision of a small model, by the definitions of the paper.

    Parameters
    ----------
    model: SmallModel
        The model and the instance.
    limits: Sequence[float]
        For a regressor, the limits of the question: none for "above the intercept", one for a threshold, two for an
        interval.

    Returns
    -------
    list[OracleMargin]
        One margin for a binary classifier and for a side of a threshold, two for "inside an interval", and one for
        each rival class of a multiclass classifier.

    """
    totals = model.output_scores(model.instance_bins)
    if model.kind == "multiclass":
        winner = max(range(CLASS_COUNT), key=lambda output: totals[output])
        return [
            OracleMargin(
                offset=model.intercepts[winner] - model.intercepts[rival],
                signed_scores=[[outputs[winner] - outputs[rival] for outputs in feature] for feature in model.scores],
                deviations=[[outputs[winner] + outputs[rival] for outputs in feature] for feature in model.deviations],
            )
            for rival in range(CLASS_COUNT)
            if rival != winner
        ]
    if model.kind == "binary":
        return [_signed(model, 1 if totals[0] > 0 else -1, 0.0)]
    low, high = (limits[0], limits[-1]) if limits else (model.intercepts[0], model.intercepts[0])
    if totals[0] <= low:
        return [_signed(model, -1, low)]
    if totals[0] > high:
        return [_signed(model, 1, high)]
    return [_signed(model, 1, low), _signed(model, -1, high)]


def oracle_problem(
    model: SmallModel,
    *,
    confidence_level: float,
    min_support: float,
    limits: Sequence[float] = (),
) -> OracleProblem:
    """Build the inputs of the paper for a small model and the options of the facts.

    Returns
    -------
    OracleProblem
        The margins, the plausible bins, the instance and the number of deviations.

    """
    return OracleProblem(
        margins=oracle_margins(model, limits),
        plausible=[
            plausible_bins(weights, continuous=index % 2 == 1, min_support=min_support, instance_bin=instance_bin)
            for index, (weights, instance_bin) in enumerate(zip(model.weights, model.instance_bins, strict=True))
        ],
        instance_bins=model.instance_bins,
        deviation_count=deviation_count(confidence_level),
    )
