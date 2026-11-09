"""Tests for the covering solvers.

The solvers rest on third-party MIP solvers that EBaMl does not prove, so these tests are the main guard of the claim
"smallest set". A brute-force oracle is the reference on small problems. A scripted backend, which obeys the contract of
``MipBackend``, drives each way a solve can end.
"""

from collections.abc import Callable, Sequence

import attrs
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from ebaml.core.exceptions import SearchLimitError
from ebaml.local import covering_solvers as solvers_module
from ebaml.local.covering_backend import CoveringModel, MipBackend, ModelResult, ModelStatus, SolveSettings, Stop
from ebaml.local.covering_greedy import greedy_cover
from ebaml.local.covering_problem import Choice, Cover, CoveringProblem
from ebaml.local.covering_solvers import CoveringSolver, GreedyCover, LocalSearchCover, MipCover
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver
from tests.unit.ebaml.local.conftest import (
    GREEDY_FINDS_NOTHING,
    GREEDY_IS_NOT_OPTIMAL,
    ProblemParts,
    check_result,
    oracle_size,
    problem_parts,
    random_problem,
    set_cover_problem,
    set_covers,
    smallest_cover,
)

SOLVERS: list[CoveringSolver] = [
    MipCover(),
    MipCover(OrToolsBackend(OrToolsSolver.SCIP)),
    MipCover(OrToolsBackend(OrToolsSolver.CP_SAT)),
    MipCover(stop=Stop.FIRST),
    MipCover(OrToolsBackend(OrToolsSolver.CP_SAT), stop=Stop.FIRST),
    LocalSearchCover(),
    LocalSearchCover(bound_backend=None),
    GreedyCover(),
    GreedyCover(bound_backend=None),
]
PROVING_SOLVERS: list[MipCover] = [MipCover(OrToolsBackend(solver)) for solver in OrToolsSolver]


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


def _solve_or_limit(solver: CoveringSolver, problem: CoveringProblem) -> Cover | SearchLimitError | None:
    try:
        return solver.solve(problem, seed=0)
    except SearchLimitError as error:
        return error


def test_every_solver_and_backend_satisfies_its_protocol() -> None:
    assert all(isinstance(solver, CoveringSolver) for solver in SOLVERS)
    assert isinstance(ScriptedBackend(), MipBackend)


@settings(max_examples=60, deadline=None)
@given(parts=problem_parts())
@example(parts=([-0.5], [[[1.0]]]))  # one margin, one group
@example(parts=([1.5, 0.5], [[[-1.0, 0.0]]]))  # the empty set is sufficient
@example(parts=([-5.5, -5.5], [[[1.0, 1.0]], [[1.0, 1.0]]]))  # no set is sufficient
@example(parts=([-1.5, -1.5], [[[2.0, 0.0]], [[0.0, 2.0]], [[2.0, 2.0]]]))  # ties between sets of size 1
@example(parts=([-1.5, 0.5], [[[3.0, -5.0]], [[0.0, 5.0]]]))  # a negative gain: the greedy finds no set
@pytest.mark.parametrize("solver", SOLVERS, ids=repr)
def test_every_solver_returns_a_sufficient_set_no_smaller_than_the_oracle(
    solver: CoveringSolver,
    parts: ProblemParts,
) -> None:
    problem = CoveringProblem(*parts)
    smallest = oracle_size(problem)
    result = _solve_or_limit(solver, problem)
    if isinstance(result, SearchLimitError):
        # only an algorithm that starts from the greedy gives up, and only on a problem with negative gains
        assert isinstance(solver, GreedyCover | LocalSearchCover)
        assert not problem.monotone
        return
    if smallest is None:
        assert result is None
        return
    assert result is not None
    check_result(problem, result)
    assert len(result.choices) >= smallest
    assert result.lower_bound <= smallest
    if result.minimum:
        assert len(result.choices) == smallest
    if isinstance(solver, MipCover) and solver.stop is Stop.OPTIMAL:
        assert len(result.choices) == smallest
        assert result.minimum


@pytest.mark.parametrize(("seed", "feature_count", "margin_count"), [(0, 60, 3), (1, 100, 6), (2, 150, 9)])
def test_the_solvers_of_ortools_agree_on_the_size_of_large_problems(
    seed: int,
    feature_count: int,
    margin_count: int,
) -> None:
    problem = random_problem(seed, feature_count, margin_count)
    sizes = set()
    for solver in PROVING_SOLVERS:
        result = solver.solve(problem, seed=0)
        assert result is not None
        check_result(problem, result)
        assert result.minimum
        sizes.add(len(result.choices))
    assert len(sizes) == 1


def _exact_size(problem: CoveringProblem) -> int | None:
    result = MipCover().solve(problem, seed=0)
    return None if result is None else len(result.choices)


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts(), power=st.integers(min_value=-6, max_value=6))
def test_the_size_survives_a_scaling_of_all_numbers_by_a_power_of_ten(parts: ProblemParts, power: int) -> None:
    worst, groups = parts
    factor = 10.0**power
    scaled = CoveringProblem(
        [value * factor for value in worst],
        [[[gain * factor for gain in gains] for gains in alternatives] for alternatives in groups],
    )
    assert _exact_size(scaled) == _exact_size(CoveringProblem(worst, groups))


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts())
def test_the_size_survives_a_permutation_of_the_groups_and_the_margins(parts: ProblemParts) -> None:
    worst, groups = parts
    reversed_groups = CoveringProblem(worst, list(reversed(groups)))
    reversed_margins = CoveringProblem(
        list(reversed(worst)),
        [[list(reversed(gains)) for gains in alternatives] for alternatives in groups],
    )
    size = _exact_size(CoveringProblem(worst, groups))
    assert _exact_size(reversed_groups) == size
    assert _exact_size(reversed_margins) == size


@settings(max_examples=30, deadline=None)
@given(parts=problem_parts())
def test_the_size_survives_a_useless_group_a_dominated_alternative_and_a_copied_margin(parts: ProblemParts) -> None:
    worst, groups = parts
    size = _exact_size(CoveringProblem(worst, groups))
    zero_group = CoveringProblem(worst, [*groups, [[0.0] * len(worst)]])
    dominated = CoveringProblem(worst, [[*groups[0], [gain - 1.0 for gain in groups[0][0]]], *groups[1:]])
    copied_margin = CoveringProblem(
        [*worst, worst[0]],
        [[[*gains, gains[0]] for gains in alternatives] for alternatives in groups],
    )
    assert _exact_size(zero_group) == size
    assert _exact_size(dominated) == size
    assert _exact_size(copied_margin) == size


def test_a_margin_of_exactly_zero_is_not_sufficient_for_any_solver() -> None:
    # The one set of size 1 leaves the second margin at exactly zero, so the set of size 2 is the smallest.
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 1.0]], [[0.0, 2.0]]])
    for solver in SOLVERS:
        result = solver.solve(problem, seed=0)
        assert result is not None
        assert len(result.choices) == 2
        assert result.guaranteed_margin > 0


def test_a_tiny_positive_margin_is_sufficient_but_below_epsilon() -> None:
    # The set {0} keeps a margin of 1e-12, which the check in Python accepts and the MIP, with ε relative to the
    # numbers, does not. The greedy finds the set, and the MIP, which looks for a smaller one, proves that none exists.
    problem = CoveringProblem([-1.0], [[[1.0 + 1e-12]], [[0.5]], [[0.5]]])
    for solver in SOLVERS:
        result = solver.solve(problem, seed=0)
        assert result is not None
        check_result(problem, result)
        assert len(result.choices) == 1


def _picks(*groups: int) -> Callable[[CoveringModel], ModelResult]:
    def answer(model: CoveringModel) -> ModelResult:
        picks = [choice for choice in model.choices if choice.group in groups]
        return ModelResult(ModelStatus.OPTIMAL, picks, float(len(picks)))

    return answer


def _ends(status: ModelStatus) -> Callable[[CoveringModel], ModelResult]:
    def answer(_model: CoveringModel) -> ModelResult:
        return ModelResult(status)

    return answer


NO_LP: ModelResult = ModelResult(ModelStatus.NO_SOLUTION)


def test_the_mip_asks_for_fewer_picks_than_the_greedy_with_its_settings() -> None:
    backend = ScriptedBackend()
    result = MipCover(backend, stop=Stop.FIRST, time_limit=3.0).solve(GREEDY_IS_NOT_OPTIMAL, seed=7)
    [(model, solve_settings)] = backend.calls
    assert model.max_picks == len(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or []) - 1
    assert solve_settings == SolveSettings(seed=7, time_limit=3.0, stop=Stop.FIRST)
    assert result is not None
    assert [choice.group for choice in result.choices] == [1, 3]


def test_a_set_of_the_solver_that_fails_the_check_is_solved_again_with_a_larger_epsilon() -> None:
    answers = [_picks(2), _picks(0, 1)]

    def wrong_then_right(model: CoveringModel) -> ModelResult:
        return answers.pop(0)(model)

    backend = ScriptedBackend(answer=wrong_then_right, lp=NO_LP)
    # {2} leaves both margins at exactly zero, as a solver with a loose tolerance may return
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]], [[1.0, 1.0]]])
    result = MipCover(backend).solve(problem, seed=0)
    # each need is ε plus the worst case -1 over the scale 2 of the margin
    first, second = (model.margin_needs[0] - 0.5 for model, _ in backend.calls)
    assert second == pytest.approx(100 * first)
    assert result is not None
    assert [choice.group for choice in result.choices] == [0, 1]


def test_every_epsilon_failing_the_check_leaves_the_set_of_the_greedy() -> None:
    backend = ScriptedBackend(answer=_picks(2), lp=NO_LP)
    problem = CoveringProblem([-1.0, -1.0], [[[2.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]], [[1.0, 1.0]]])
    result = MipCover(backend).solve(problem, seed=0)
    assert len(backend.calls) == 3
    assert result is not None
    assert result.choices == sorted(greedy_cover(problem) or [])
    assert not result.minimum


def test_a_feasible_set_at_the_time_limit_is_returned_without_the_claim() -> None:
    def feasible(model: CoveringModel) -> ModelResult:
        picks = [choice for choice in model.choices if choice.group in {1, 3}]
        return ModelResult(ModelStatus.FEASIBLE, picks, 2.0)

    result = MipCover(ScriptedBackend(answer=feasible, lp=NO_LP)).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [1, 3]
    assert result.lower_bound == 1
    assert not result.minimum


def test_no_solution_at_the_time_limit_keeps_the_set_of_the_greedy() -> None:
    result = MipCover(ScriptedBackend(answer=_ends(ModelStatus.NO_SOLUTION))).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert result.choices == sorted(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or [])
    assert not result.minimum


def test_an_infeasible_model_with_the_size_limit_proves_the_set_of_the_greedy() -> None:
    result = MipCover(ScriptedBackend(answer=_ends(ModelStatus.INFEASIBLE))).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert len(result.choices) == len(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or [])
    assert result.minimum


def test_an_infeasible_model_without_a_greedy_set_proves_that_no_set_exists() -> None:
    assert MipCover(ScriptedBackend(answer=_ends(ModelStatus.INFEASIBLE))).solve(GREEDY_FINDS_NOTHING, seed=0) is None


def test_no_solution_at_the_time_limit_and_no_greedy_set_raises_with_the_fix() -> None:
    solver = MipCover(ScriptedBackend(answer=_ends(ModelStatus.NO_SOLUTION)), time_limit=2.5)
    with pytest.raises(SearchLimitError, match=r"time limit of 2\.5 s\. Give MipCover a larger time_limit"):
        solver.solve(GREEDY_FINDS_NOTHING, seed=0)


def test_the_mip_finds_the_set_that_the_greedy_misses() -> None:
    result = MipCover().solve(GREEDY_FINDS_NOTHING, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [0, 1]
    assert result.minimum


@pytest.mark.parametrize("solver", [GreedyCover(bound_backend=None), LocalSearchCover(bound_backend=None)], ids=repr)
def test_an_algorithm_that_starts_from_the_greedy_gives_up_with_the_fix(solver: CoveringSolver) -> None:
    with pytest.raises(SearchLimitError, match="Use MipCover"):
        solver.solve(GREEDY_FINDS_NOTHING, seed=0)


@pytest.mark.parametrize("solver_class", [GreedyCover, LocalSearchCover])
def test_an_infeasible_lp_proves_that_no_set_exists(solver_class: type[GreedyCover | LocalSearchCover]) -> None:
    backend = ScriptedBackend(lp=ModelResult(ModelStatus.INFEASIBLE))
    assert solver_class(bound_backend=backend).solve(GREEDY_FINDS_NOTHING, seed=0) is None
    assert MipCover(backend).solve(GREEDY_FINDS_NOTHING, seed=0) is None


def test_the_lower_bound_of_a_fractional_lp_is_rounded_up() -> None:
    # each margin needs a gain of 1 from two of the three groups, so the LP puts 1/2 on each group: a value of 1.5
    problem = CoveringProblem([-1.0] * 3, [[[1.0, 1.0, 0.0]], [[0.0, 1.0, 1.0]], [[1.0, 0.0, 1.0]]])
    assert solvers_module._lower_bound(problem, [[0], [0], [0]], OrToolsBackend(), 0) == 2


def test_an_lp_value_of_an_integer_is_not_rounded_past_it() -> None:
    lp = ModelResult(ModelStatus.OPTIMAL, [], 2.0000001)
    assert solvers_module._lower_bound(GREEDY_IS_NOT_OPTIMAL, [[0]] * 4, ScriptedBackend(lp=lp), 0) == 2


def test_a_greedy_without_the_lp_reports_the_trivial_bound() -> None:
    result = GreedyCover(bound_backend=None).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert result.lower_bound == 1
    assert not result.minimum


def test_a_monotone_problem_with_no_set_found_falls_back_to_every_group() -> None:
    # The greedy always finds a set of a monotone problem whose full set is sufficient, so only a direct call reaches
    # the fallback. It returns every group, which the caller has checked, without the claim.
    problem = CoveringProblem([-1.5], [[[1.0]], [[1.0]], [[0.0]]])
    cover = solvers_module._no_set_found(problem, 1, "unused")
    assert cover == Cover([Choice(0, 0), Choice(1, 0), Choice(2, 0)], 0.5, 1, minimum=False)


def test_a_monotone_problem_whose_full_set_fails_has_no_set_and_no_call() -> None:
    backend = ScriptedBackend()
    assert MipCover(backend).solve(CoveringProblem([-5.5], [[[1.0]], [[2.0]]]), seed=0) is None
    assert backend.calls == []


@pytest.mark.parametrize("solver", SOLVERS, ids=repr)
def test_each_solver_gives_the_same_set_twice(solver: CoveringSolver) -> None:
    problem = random_problem(4, 40, 4)
    assert solver.solve(problem, seed=5) == solver.solve(problem, seed=5)


def test_the_local_search_finds_the_smallest_set_that_the_greedy_misses() -> None:
    result = LocalSearchCover(bound_backend=None).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [1, 3]
    assert not result.minimum


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_local_search_is_never_worse_than_the_greedy(seed: int) -> None:
    problem = random_problem(seed, 60, 5)
    greedy = GreedyCover().solve(problem, seed=0)
    local = LocalSearchCover(iterations=30).solve(problem, seed=seed)
    assert greedy is not None
    assert local is not None
    check_result(problem, local)
    assert len(local.choices) <= len(greedy.choices)


def test_the_local_search_with_no_steps_returns_the_greedy_set() -> None:
    result = LocalSearchCover(iterations=0, bound_backend=None).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert result.choices == sorted(greedy_cover(GREEDY_IS_NOT_OPTIMAL) or [])


def test_the_local_search_uses_its_own_seed_over_the_seed_of_the_call() -> None:
    problem = random_problem(5, 80, 6)
    own = LocalSearchCover(iterations=10, seed=3)
    assert own.solve(problem, seed=0) == own.solve(problem, seed=99)
    assert own.solve(problem, seed=0) == LocalSearchCover(iterations=10).solve(problem, seed=3)


def test_the_local_search_removes_at_least_one_pick_per_step() -> None:
    # a share of 1% of three picks rounds to zero, so each step removes one pick and can still improve the set
    result = LocalSearchCover(destroy_share=0.01, bound_backend=None).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert len(result.choices) == 2


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: MipCover(time_limit=0.0), "time_limit"),
        (lambda: LocalSearchCover(iterations=-1), "iterations"),
        (lambda: LocalSearchCover(destroy_share=0.0), "destroy_share"),
        (lambda: LocalSearchCover(destroy_share=1.5), "destroy_share"),
    ],
)
def test_the_options_of_the_solvers_are_checked(make: Callable[[], CoveringSolver], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make()


@settings(max_examples=40, deadline=None)
@given(sets=set_covers())
@pytest.mark.parametrize("solver", PROVING_SOLVERS, ids=repr)
def test_the_mip_solves_set_cover_through_the_reduction_of_the_proof(
    solver: CoveringSolver,
    sets: Sequence[frozenset[int]],
) -> None:
    # Theorem 8 of proofs/EBaMlProofs/LocalExplanations/Hardness.lean links the code to Set Cover
    result = solver.solve(set_cover_problem(sets), seed=0)
    smallest = smallest_cover(sets)
    if smallest is None:
        assert result is None
    else:
        assert result is not None
        assert len(result.choices) == smallest
        assert result.minimum


def test_the_local_search_stops_and_proves_the_set_that_reaches_the_lp_bound() -> None:
    # the LP bound is 2, so the search stops at {1, 3} and marks it as the smallest set
    result = LocalSearchCover(iterations=1000).solve(GREEDY_IS_NOT_OPTIMAL, seed=0)
    assert result is not None
    assert [choice.group for choice in result.choices] == [1, 3]
    assert result.lower_bound == 2
    assert result.minimum
