"""Tests of the difficulty proposal: the text of the inputs, and the reading of a reply."""

import attrs
import pytest
import structlog.testing

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.llm import difficulties
from ebaml.llm.difficulties import Rating, features_text, read_proposal
from ebaml.local.costs import DataCost, Difficulty, UnitCost
from tests.unit.ebaml.conftest import FittedClassifier, fake_model, instance_of


@attrs.frozen
class _Rating:
    feature: str
    difficulty: str
    reason: str


def test_a_reply_rating_keeps_the_shape_of_the_protocol() -> None:
    assert isinstance(_Rating("a", "Easy", "r"), Rating)


def test_the_text_names_each_input_with_its_type_and_its_current_value(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    instance = instance_of(data, 0)
    lines = features_text(ebm, instance).splitlines()
    assert len(lines) == len(ebm.feature_names_in_)
    for line, name in zip(lines, ebm.feature_names_in_, strict=True):
        assert line.startswith(f'- "{name}" (')
        assert line.endswith(f"current value {instance[name]}")
    assert {line.split("(")[1].split(")")[0] for line in lines} >= {"continuous", "nominal"}


def continuous_type(_ebm: ExplainableBoostingClassifier, _feature_index: int) -> str:
    return "continuous"


def test_the_text_quotes_a_name_as_json_so_that_its_end_is_clear() -> None:
    ebm = fake_model(feature_names_in_=['savings "net"'], term_features_=[(0,)])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(difficulties, "_feature_type", continuous_type)
        assert features_text(ebm, {'savings "net"': 1.0}) == '- "savings \\"net\\"" (continuous), current value 1.0'


@pytest.mark.parametrize(
    "echoed",
    ["monthly savings (continuous)", '"monthly savings"', '"monthly savings" (continuous)'],
)
def test_a_rating_that_echoes_the_quotes_or_the_type_of_its_input_keeps_its_cost(echoed: str) -> None:
    # a live model returned "monthly savings (continuous)", and the proposal dropped every rating
    proposal = read_proposal([_Rating(echoed, "Easy", "cheap")], ["monthly savings", "age"], DataCost())
    assert [item.feature for item in proposal.difficulties] == ["monthly savings"]


def test_a_name_with_a_bracket_of_its_own_is_matched_as_it_is() -> None:
    proposal = read_proposal([_Rating("size (m2)", "Hard", "r")], ["size (m2)"], DataCost())
    assert [item.feature for item in proposal.difficulties] == ["size (m2)"]


def test_each_rating_of_a_known_input_becomes_its_multiplier() -> None:
    proposal = read_proposal(
        [_Rating("a", "Easy", "cheap"), _Rating("b", "VeryHard", "the past")],
        ["a", "b", "c"],
        DataCost(),
    )
    assert [(item.feature, item.difficulty, item.multiplier, item.reason) for item in proposal.difficulties] == [
        ("a", Difficulty.EASY, 0.5, "cheap"),
        ("b", Difficulty.VERY_HARD, 4.0, "the past"),
    ]
    assert proposal.costs.multipliers == {"a": 0.5, "b": 4.0}
    assert proposal.costs.base == DataCost()
    # contract 3: the facts say that a language model proposed these difficulties
    assert proposal.costs.proposed_by_model


@pytest.mark.parametrize(
    ("ratings", "known"),
    [
        pytest.param([_Rating("z", "Easy", "?")], False, id="unknown-input"),
        pytest.param([_Rating("a", "Hard", "first"), _Rating("a", "Easy", "second")], True, id="second-rating"),
    ],
)
def test_a_rating_that_cannot_give_a_cost_is_dropped_and_logged(ratings: list[_Rating], *, known: bool) -> None:
    with structlog.testing.capture_logs() as logs:
        proposal = read_proposal(ratings, ["a"], UnitCost())
    assert [item.reason for item in proposal.difficulties] == (["first"] if known else [])
    assert logs == [
        {"event": "difficulty_dropped", "feature": ratings[-1].feature, "known": known, "log_level": "warning"},
    ]


def test_a_level_outside_the_scale_is_an_error() -> None:
    with pytest.raises(ValueError, match="Impossible"):
        read_proposal([_Rating("a", "Impossible", "?")], ["a"], UnitCost())
