"""Tests for the completions of a partially known instance."""

from collections.abc import Sequence

import attrs
import numpy as np
import pytest

from ebaml.local import completions as completions_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local.covering_problem import Cover, CoveringProblem
from ebaml.local.covering_solvers import GreedyCover
from ebaml.local.decisions import Interval
from ebaml.local.fact_features import Feature
from ebaml.local.fact_types import CompletionStep
from ebaml.local.sufficient_reasons import Bound
from tests.unit.ebaml.conftest import (
    FittedClassifier,
    FittedRegressor,
    instance_of,
)
from tests.unit.ebaml.local.conftest import multiclass_model


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


def test_each_feature_reads_its_scores_once_for_each_margin(
    continuous_ebm: FittedClassifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sample_completions of 0.15.0 read the scores of every feature again at each step of the completions
    ebm, data = continuous_ebm
    reads: list[str] = []
    in_completions = [False]
    read_scores = Feature.scores.fget
    assert read_scores is not None

    def counted_scores(feature: Feature) -> np.ndarray:
        if in_completions[0]:
            reads.append(feature.name)
        scores: np.ndarray = read_scores(feature)
        return scores

    def flagged(
        features: Sequence[Feature],
        order: Sequence[Feature],
        bounds: Sequence[Bound],
        options: fact_types_module.LocalOptions,
    ) -> list[CompletionStep]:
        in_completions[0] = True
        try:
            return completions_module.sample_completions(features, order, bounds, options)
        finally:
            in_completions[0] = False

    monkeypatch.setattr(Feature, "scores", property(counted_scores))
    monkeypatch.setattr(facts_module, "sample_completions", flagged)
    facts_module.compute_local_facts(ebm, instance_of(data, 5), fact_types_module.LocalOptions(max_steps=4))
    assert sorted(reads) == sorted(ebm.feature_names_in_)
