"""The difficulty proposal from an EBM to the cost strategy, with the real ``baml_sdk`` and a local LLM server."""

import orjson

from ebaml import Difficulty, ModelOptions, ProposeDifficultiesRequest, propose_difficulties
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, Reply
from tests.unit.ebaml.conftest import PandasFrame

INSTANCE = {"age": 20.0, "fare": 50.0, "siblings": 1.0}
RATINGS = [
    {"feature": "fare", "difficulty": "Easy", "reason": "A cheaper ticket is a choice."},
    {"feature": "age", "difficulty": "VeryHard", "reason": "Age cannot change, so mark it as immutable."},
]


def test_one_request_carries_every_input_with_its_current_value(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(text=orjson.dumps(RATINGS).decode())
    propose_difficulties(ProposeDifficultiesRequest(ebm, INSTANCE, model=llm_model))
    (prompt,) = fake_llm.prompts
    assert all(f'- "{name}" (continuous), current value {value}' in prompt for name, value in INSTANCE.items())


def test_the_reply_becomes_a_scaled_cost_that_a_language_model_proposed(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(text=orjson.dumps(RATINGS).decode())
    proposal = propose_difficulties(ProposeDifficultiesRequest(ebm, INSTANCE, model=llm_model))
    assert [(item.feature, item.difficulty) for item in proposal.difficulties] == [
        ("fare", Difficulty.EASY),
        ("age", Difficulty.VERY_HARD),
    ]
    assert proposal.costs.multipliers == {"fare": 0.5, "age": 4.0}
    assert proposal.costs.proposed_by_model
