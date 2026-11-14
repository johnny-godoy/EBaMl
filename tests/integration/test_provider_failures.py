"""What a user sees when the provider fails, through the real BAML runtime and a local server that misbehaves.

The retry rules, as these tests pin them: a rejected key (401, 403) ends the call at once, because a second request
cannot succeed. A limit of the rate (429) and a failure of the server (5xx) get two more tries.
"""

import pytest
import structlog.testing
from baml_bridge.errors import BamlError

from ebaml import (
    APIKeyRejectedError,
    DescribeGraphRequest,
    InvalidOptionError,
    LLMRequestError,
    MissingAPIKeyError,
    ModelOptions,
    ProviderUnreachableError,
    RateLimitedError,
    describe_graph,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, Reply
from tests.unit.ebaml.conftest import PandasFrame

TRIES_WITH_RETRY = 3


def _describe(ebm: ExplainableBoostingClassifier, model: ModelOptions | str) -> str:
    return describe_graph(DescribeGraphRequest(ebm, feature_index=0, model=model, num_sentences=2))


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_raises_at_once_and_is_not_retried(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    status: int,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(status=status)
    with pytest.raises(APIKeyRejectedError, match=str(status)):
        _describe(ebm, llm_model)
    assert len(fake_llm.requests) == 1


def test_a_limit_of_the_rate_is_retried_and_then_raised_with_the_steps_to_fix_it(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(status=429)
    with pytest.raises(RateLimitedError, match="To fix it"):
        _describe(ebm, llm_model)
    assert len(fake_llm.requests) == TRIES_WITH_RETRY


@pytest.mark.parametrize("status", [500, 503])
def test_a_failure_of_the_server_is_retried_and_then_raised_as_unreachable(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    status: int,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(status=status)
    with pytest.raises(ProviderUnreachableError):
        _describe(ebm, llm_model)
    assert len(fake_llm.requests) == TRIES_WITH_RETRY


def test_a_server_that_recovers_gives_the_text_of_the_good_reply(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(status=503) if len(fake_llm.requests) == 1 else Reply(text="The graph.")
    assert _describe(ebm, llm_model) == "The graph."


def test_a_refused_connection_is_unreachable_and_names_the_provider(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
) -> None:
    _, ebm = binary_data_and_model
    nobody_listens = ModelOptions(
        provider=llm_model.provider,
        model=llm_model.model,
        api_key_env=llm_model.api_key_env,
        base_url="http://127.0.0.1:1/v1",
    )
    with pytest.raises(ProviderUnreachableError, match="openai-generic"):
        _describe(ebm, nobody_listens)


def test_every_user_fixable_failure_is_an_llm_request_error() -> None:
    for error in (APIKeyRejectedError, MissingAPIKeyError, ProviderUnreachableError, RateLimitedError):
        assert issubclass(error, LLMRequestError)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_a_provider_that_needs_a_key_reports_an_unset_variable_before_a_request_goes_out(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    provider: str,
) -> None:
    _, ebm = binary_data_and_model
    keyless = ModelOptions(
        provider=provider,
        model="model",
        api_key_env="UNSET_VARIABLE_OF_THE_TESTS",
        base_url=llm_model.base_url,
    )
    with pytest.raises(MissingAPIKeyError, match="To fix it"):
        _describe(ebm, keyless)
    assert fake_llm.requests == []


@pytest.mark.parametrize("provider", ["openai-generic", "ollama"])
def test_a_local_server_needs_no_key_and_receives_no_authorization_header(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
    provider: str,
) -> None:
    _, ebm = binary_data_and_model
    keyless = ModelOptions(
        provider=provider,
        model="model",
        api_key_env="UNSET_VARIABLE_OF_THE_TESTS",
        base_url=llm_model.base_url,
    )
    assert _describe(ebm, keyless) == "A fake reply."
    assert all("authorization" not in request.headers for request in fake_llm.requests)


@pytest.mark.usefixtures("llm_model")
def test_an_unknown_provider_lists_the_valid_ones_and_sends_nothing(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    with pytest.raises(InvalidOptionError, match="openrouter"):
        _describe(ebm, "nosuchprovider/model")
    assert fake_llm.requests == []


def test_a_reply_cut_by_the_token_limit_is_retried_with_a_shorter_graph(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(finish_reason="length")
    with structlog.testing.capture_logs() as events, pytest.raises(BamlError, match="FinishReasonError"):
        _describe(ebm, llm_model)
    retries = [event["retry"] for event in events if event["event"] == "truncated_reply"]
    assert retries == [1, 2]
    first_turn_prompts = [request.prompt for request in fake_llm.requests if request.turns == 1]
    assert len(first_turn_prompts) >= 2
    assert len(first_turn_prompts[-1]) < len(first_turn_prompts[0]), "a retry must send a shorter graph"


def test_a_reply_without_usage_still_gives_the_text(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = lambda _request: Reply(text="No usage.", usage=None)
    assert _describe(ebm, llm_model) == "No usage."
