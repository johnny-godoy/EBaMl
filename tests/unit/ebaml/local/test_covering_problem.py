"""Tests for the covering problem: the check of a set, its properties, the result and the reduction."""

from hypothesis import given, settings

from ebaml.local.covering_problem import Choice, Cover, CoveringProblem, make_cover, reduce_problem
from tests.unit.ebaml.local.conftest import ProblemParts, oracle_size, problem_parts


def test_a_margin_of_exactly_zero_is_not_sufficient() -> None:
    # a logit of zero is not the positive class, so the check uses a strict inequality
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 1.0]], [[0.0, 2.0]]])
    assert problem.margins([Choice(0, 0)]) == [1.0, 0.0]
    assert not problem.sufficient([Choice(0, 0)])
    assert problem.sufficient([Choice(0, 0), Choice(1, 0)])


def test_the_monotone_flag_needs_one_alternative_and_no_negative_gain() -> None:
    assert CoveringProblem([-1.0], [[[1.0]], [[0.0]]]).monotone
    assert not CoveringProblem([-1.0], [[[1.0], [2.0]]]).monotone
    assert not CoveringProblem([-1.0], [[[-1.0]]]).monotone


def test_every_group_skips_a_group_with_no_alternative() -> None:
    assert CoveringProblem([-0.5], [[], [[1.0]]]).every_group == [Choice(1, 0)]


def test_the_scale_of_a_margin_is_its_largest_absolute_number() -> None:
    problem = CoveringProblem([-3.0, 0.0], [[[1.0, 0.0]], [[-5.0, 0.0]]])
    assert problem.scales == [5.0, 1.0]


def test_a_cover_reaching_its_bound_is_the_smallest_set() -> None:
    problem = CoveringProblem([-1.0], [[[2.0]], [[3.0]]])
    assert make_cover(problem, [Choice(1, 0)], 1, proved=False) == Cover([Choice(1, 0)], 2.0, 1, minimum=True)
    two = make_cover(problem, [Choice(1, 0), Choice(0, 0)], 1, proved=False)
    assert two == Cover([Choice(0, 0), Choice(1, 0)], 4.0, 1, minimum=False)
    assert make_cover(problem, [Choice(1, 0), Choice(0, 0)], 1, proved=True).minimum


def _reduced(problem: CoveringProblem) -> CoveringProblem:
    kept = reduce_problem(problem)
    return CoveringProblem(
        problem.worst,
        [[problem.groups[group][index] for index in indices] for group, indices in enumerate(kept)],
    )


@settings(max_examples=100, deadline=None)
@given(parts=problem_parts(max_groups=5))
def test_the_reduction_keeps_the_smallest_size(parts: ProblemParts) -> None:
    problem = CoveringProblem(*parts)
    assert oracle_size(_reduced(problem)) == oracle_size(problem)


def test_the_reduction_keeps_the_only_alternative_that_meets_a_margin() -> None:
    problem = CoveringProblem([-1.5, -1.5], [[[2.0, 0.0], [1.0, 0.0]], [[0.0, 2.0]]])
    assert reduce_problem(problem) == [[0], [0]]


def test_the_reduction_keeps_the_first_of_two_equal_alternatives_of_a_group() -> None:
    assert reduce_problem(CoveringProblem([-0.5], [[[1.0], [1.0], [0.5]]])) == [[0]]


def test_the_reduction_keeps_two_groups_with_equal_gains() -> None:
    # Both features are needed: a removal of one, as a dominance across groups would do, leaves no sufficient set.
    problem = CoveringProblem([-1.5], [[[1.0]], [[1.0]]])
    assert reduce_problem(problem) == [[0], [0]]
    assert oracle_size(problem) == 2


def test_the_reduction_removes_the_alternatives_with_no_positive_gain() -> None:
    assert reduce_problem(CoveringProblem([-0.5, -0.5], [[[0.0, 0.0], [-1.0, 0.0]], [[2.0, -1.0]]])) == [[], [0]]
