"""The path of ``explain_instance``: exact facts from the EBM, three prompts, three replies."""

import orjson
import pytest
import structlog.testing

from ebaml import (
    Audience,
    DataScientist,
    DescribeEBMRequest,
    ExplainInstanceRequest,
    Layperson,
    ModelOptions,
    UnsupportedModelError,
    describe_ebm,
    explain_instance,
    facts_to_texts,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.docs.conftest import training_data
from tests.unit.ebaml.conftest import PandasFrame

INSTANCE = {"age": 20.0, "fare": 50.0, "siblings": 1.0}
GLOBAL_TEXT = "GLOBAL-EXPLANATION-OF-THE-TESTS"
# the opening words of the question of each explanation
QUESTION_KEYS = {
    "counterfactual": "what would have to change for the prediction to be different",
    "sufficient_reason": "which inputs are enough to settle the prediction",
    "completions": "how certain the outcome is as the inputs become known",
}


def _which(request: RecordedRequest) -> str:
    (name,) = (name for name, words in QUESTION_KEYS.items() if words in request.prompt)
    return name


def _request(
    ebm: ExplainableBoostingClassifier,
    model: ModelOptions,
    *,
    audience: Audience | str = "layperson",
    language: str = "English",
    num_sentences: int = 5,
) -> ExplainInstanceRequest:
    return ExplainInstanceRequest(
        ebm,
        INSTANCE,
        GLOBAL_TEXT,
        model=model,
        audience=audience,
        language=language,
        num_sentences=num_sentences,
    )


def test_three_independent_requests_go_out_and_each_one_carries_the_global_explanation(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    explain_instance(_request(ebm, llm_model))
    assert sorted(_which(request) for request in fake_llm.requests) == sorted(QUESTION_KEYS)
    assert all(GLOBAL_TEXT in prompt for prompt in fake_llm.prompts)
    assert {request.turns for request in fake_llm.requests} == {1}


def test_each_reply_lands_in_the_field_of_its_explanation(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda request: Reply(text=f"text of the {_which(request)}")
    explanation = explain_instance(_request(ebm, llm_model))
    assert explanation.counterfactual == "text of the counterfactual"
    assert explanation.sufficient_reason == "text of the sufficient_reason"
    assert explanation.completions == "text of the completions"


def test_the_facts_in_the_prompts_are_the_facts_that_python_computed(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    explanation = explain_instance(_request(ebm, llm_model))
    texts = facts_to_texts(explanation.facts, Layperson())
    by_name = {_which(request): request.prompt for request in fake_llm.requests}
    for name in QUESTION_KEYS:
        fact = orjson.loads(getattr(texts, name))
        for key in fact:
            assert f'"{key}"' in by_name[name], f"the key {key!r} of the {name} facts is missing from its prompt"
    for prompt in by_name.values():
        assert f'"predicted_class": {explanation.facts.prediction.decision}' in prompt


@pytest.mark.parametrize("audience", [Layperson(), DataScientist()], ids=["layperson", "data_scientist"])
def test_the_prompt_holds_the_guidance_of_the_audience(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    audience: Layperson | DataScientist,
) -> None:
    _, ebm = binary_data_and_model
    explain_instance(_request(ebm, llm_model, audience=audience))
    assert all(audience.guidance() in prompt for prompt in fake_llm.prompts)


def test_the_language_and_the_length_reach_the_prompts(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    explain_instance(_request(ebm, llm_model, language="Spanish", num_sentences=2))
    assert all("Spanish" in prompt for prompt in fake_llm.prompts)
    assert all("at most 2 sentences" in prompt for prompt in fake_llm.prompts)


def test_a_model_with_an_interaction_is_rejected_before_any_request(llm_model: ModelOptions, fake_llm: FakeLLM) -> None:
    features, target = training_data()
    ebm = ExplainableBoostingClassifier(interactions=2, random_state=0).fit(features, target)
    assert any(" & " in name for name in ebm.term_names_), "the fit must hold an interaction term"
    with pytest.raises(UnsupportedModelError):
        explain_instance(_request(ebm, llm_model))
    assert fake_llm.requests == []


def test_a_description_of_a_model_with_an_interaction_names_the_skipped_term(
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    features, target = training_data()
    ebm = ExplainableBoostingClassifier(interactions=2, random_state=0).fit(features, target)
    description = describe_ebm(DescribeEBMRequest(ebm, model=llm_model, num_sentences=3))
    assert description.skipped_terms, "the description must say which terms it leaves out"
    assert all(" & " in name for name in description.skipped_terms)
    assert fake_llm.requests


def test_the_usage_of_the_replies_is_logged(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(usage=(7, 5))
    with structlog.testing.capture_logs() as events:
        explain_instance(_request(ebm, llm_model))
    (usage,) = (event for event in events if event["event"] == "usage")
    assert usage["what"] == "explain_instance"
    assert usage["input_tokens"] == 3 * 7
    assert usage["output_tokens"] == 3 * 5
