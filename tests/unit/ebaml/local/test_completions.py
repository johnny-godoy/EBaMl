"""Tests for the completions of a partially known instance."""

from collections.abc import Sequence
from unittest.mock import patch

import attrs
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ebaml.local import completions as completions_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local.covering_problem import Cover, CoveringProblem
from ebaml.local.covering_solvers import GreedyCover
from ebaml.local.decisions import Interval, Threshold
from ebaml.local.fact_features import Feature
from ebaml.local.fact_types import CompletionStep
from ebaml.local.sufficient_reasons import Bound
from tests.unit.ebaml.conftest import (
    FittedClassifier,
    FittedRegressor,
    instance_of,
)
from tests.unit.ebaml.local.conftest import SmallModel, Templates, build_ebm, model_cases, multiclass_model


def test_completions_are_reproducible_shares(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    row = instance_of(data, 5)
    first = facts_module.compute_local_facts(ebm, row, fact_types_module.LocalOptions(seed=7, max_steps=2))
    again = facts_module.compute_local_facts(ebm, row, fact_types_module.LocalOptions(seed=7, max_steps=2))
    assert first.completions == again.completions
    assert [len(s.fixed) for s in first.completions] == [0, 1, 2]
    assert all(0 <= s.share <= 1 for s in first.completions)


@attrs.frozen
class _SeedSpy:
    """A ``CoveringSolver`` that runs the greedy and records the seed of each call."""

    seeds: list[int] = attrs.field(factory=list)

    def solve(self, problem: CoveringProblem, *, seed: int) -> Cover | None:
        """Record the seed, and solve with the greedy.

        Returns
        -------
        Cover | None
            The answer of ``GreedyCover``.

        """
        self.seeds.append(seed)
        return GreedyCover().solve(problem, seed=seed)


def test_the_searches_get_the_seed_of_the_options(
    categorical_regressor: FittedRegressor,
) -> None:
    ebm, _ = categorical_regressor
    spy = _SeedSpy()
    options = fact_types_module.LocalOptions(
        confidence_level=0,
        min_support=0,
        decision=Interval(-1.5, 1.5),
        seed=13,
        search=spy,
    )
    facts_module.compute_local_facts(ebm, {"f0": "q", "f1": "q", "f2": "q"}, options)
    assert spy.seeds == [13]


def test_multiclass_completions_fall_to_the_share_of_the_class() -> None:
    model, frame = multiclass_model()
    options = fact_types_module.LocalOptions(confidence_level=0.0, max_steps=3, n_samples=2000)
    facts = facts_module.compute_local_facts(model, instance_of(frame, 0), options)
    assert facts.completions[-1].share >= facts.completions[0].share
    assert 0.0 <= facts.completions[0].share <= 1.0


def test_the_reveal_order_of_several_margins_ranks_by_the_smallest_gain() -> None:
    assert completions_module._by_smallest_gain([[1.0, 4.0, 2.0], [3.0, 4.0, 0.5]]) == [1, 0, 2]


def _reference_completions(
    features: Sequence[Feature],
    order: Sequence[Feature],
    bounds: Sequence[Bound],
    options: fact_types_module.LocalOptions,
) -> list[CompletionStep]:
    """Run ``sample_completions`` of version 0.15.0, the oracle of the differential test.

    Returns
    -------
    list[CompletionStep]
        The share of each step.

    """
    random_generator = np.random.default_rng(options.seed)
    sampled_bins = {
        f.name: random_generator.choice(len(f.weights), size=options.n_samples, p=f.weights / f.weights.sum())
        for f in features
    }
    steps = []
    for step in range(min(options.max_steps, len(order)) + 1):
        known = {f.name for f in order[:step]}
        holds = np.ones(options.n_samples, dtype=bool)
        for bound in bounds:
            views = [f.against(bound.columns) for f in features]
            total = sum(v.current if v.name in known else v.scores[sampled_bins[v.name]] for v in views)
            holds &= (bound.intercept + total > 0) == (bound.sign == 1)
        steps.append(CompletionStep([f.name for f in order[:step]], float(np.mean(holds))))
    return steps


@settings(max_examples=40, deadline=None, derandomize=True)
@given(case=model_cases(), seed=st.integers(0, 3))
def test_each_share_equals_the_share_of_the_sampling_loop_of_version_0_15(
    small_templates: Templates,
    case: tuple[SmallModel, tuple[float, ...]],
    seed: int,
) -> None:
    model, limits = case
    decision = None if not limits else Threshold(limits[0]) if len(limits) == 1 else Interval(limits[0], limits[1])
    options = fact_types_module.LocalOptions(n_samples=500, seed=seed, decision=decision, max_steps=4)
    ebm = build_ebm(small_templates, model)
    facts = facts_module.compute_local_facts(ebm, model.instance, options)
    with patch.object(facts_module, "sample_completions", _reference_completions):
        reference = facts_module.compute_local_facts(ebm, model.instance, options)
    assert facts.completions == reference.completions
