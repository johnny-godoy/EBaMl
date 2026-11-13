"""Tests for the completions of a partially known instance."""

import attrs

from ebaml.local import completions as completions_module
from ebaml.local import fact_types as fact_types_module
from ebaml.local import facts as facts_module
from ebaml.local.covering_problem import Cover, CoveringProblem
from ebaml.local.covering_solvers import GreedyCover
from ebaml.local.decisions import Interval
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
