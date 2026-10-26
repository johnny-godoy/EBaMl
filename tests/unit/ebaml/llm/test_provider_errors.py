"""Tests for the explanation of the errors of an LLM provider."""

import pytest
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.core.exceptions import (
    APIKeyRejectedError,
    LLMRequestError,
    MissingAPIKeyError,
    ProviderUnreachableError,
    RateLimitedError,
)
from ebaml.llm.provider_errors import provider_errors, translate_provider_error

OPENROUTER_KEY = "openrouter needs an API key: pass api_key, or set OPENROUTER_API_KEY"
ANTHROPIC_KEY = (
    "anthropic needs an API key: pass api_key (a literal or env.NAME), or set one of the variables it consulted — "
    "ANTHROPIC_API_KEY"
)


def invalid_request(provider: str, detail: str, status_code: int | None = None) -> BamlError:
    value = baml_sdk.ai.errors.InvalidRequest(provider=provider, detail=detail, status_code=status_code)
    return BamlError(value, class_name="ai.errors.InvalidRequest")


def provider_error(status_code: int | None, message: str = "no access", provider: str = "openai") -> BamlError:
    value = baml_sdk.ai.errors.ProviderError(provider=provider, message=message, status_code=status_code)
    return BamlError(value, class_name="ai.errors.ProviderError")


def test_a_missing_openrouter_key_names_the_variable_and_the_key_page() -> None:
    explained = translate_provider_error(invalid_request("openrouter", OPENROUTER_KEY))
    assert isinstance(explained, MissingAPIKeyError)
    assert str(explained) == (
        "The openrouter provider needs an API key, and the environment has none.\n"
        "To fix it:\n"
        "  1. Create a key at https://openrouter.ai/settings/keys.\n"
        "  2. Export it in the shell that runs your code: export OPENROUTER_API_KEY=...\n"
        '  3. To read the key from another variable, pass model=ModelOptions(provider="openrouter", model="...", '
        'api_key_env="NAME").'
    )


def test_a_missing_key_with_several_variables_lists_them_once_each() -> None:
    detail = "x needs an API key: set one of GEMINI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY"
    explained = translate_provider_error(invalid_request("gemini", detail))
    message = str(explained)
    assert "  1. Create a key in the account that you have at gemini.\n" in message
    assert "  2. Export it in the shell that runs your code: export GEMINI_API_KEY=...\n" in message
    assert "  3. The provider reads any of these variables: GEMINI_API_KEY, GOOGLE_API_KEY.\n" in message


def test_a_missing_key_without_a_variable_in_the_detail_says_so() -> None:
    explained = translate_provider_error(invalid_request("acme", "acme needs an API key"))
    assert "export <the variable of the provider>=..." in str(explained)
    assert 'provider="acme"' in str(explained)


def test_the_anthropic_detail_gives_its_variable_and_page() -> None:
    explained = translate_provider_error(invalid_request("anthropic", ANTHROPIC_KEY))
    assert isinstance(explained, MissingAPIKeyError)
    assert "export ANTHROPIC_API_KEY=..." in str(explained)
    assert "https://console.anthropic.com/settings/keys" in str(explained)
    assert "NAME" in str(explained)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_quotes_the_provider(status: int) -> None:
    explained = translate_provider_error(provider_error(status, "Incorrect API key provided"))
    assert isinstance(explained, APIKeyRejectedError)
    message = str(explained)
    assert message.startswith(f"The openai provider rejected the API key (HTTP {status}).\n")
    assert "The provider said: Incorrect API key provided\n" in message
    assert "  1. Check that the variable holds the key of this provider" in message
    assert "  3. Create a new key if the old key has expired" in message


def test_an_invalid_request_with_a_rejected_status_is_a_rejected_key() -> None:
    explained = translate_provider_error(invalid_request("openai", "bad key", status_code=401))
    assert isinstance(explained, APIKeyRejectedError)
    assert "The provider said: bad key\n" in str(explained)


def test_a_rate_limit_status_is_a_rate_limit() -> None:
    explained = translate_provider_error(provider_error(429, "slow down"))
    assert isinstance(explained, RateLimitedError)
    message = str(explained)
    assert message.startswith("The openai provider limited the rate of the requests.\n")
    assert "The provider said: slow down\n" in message
    assert "describe_ebm(ebm, max_concurrency=2)" in message


@pytest.mark.parametrize(
    ("retry_after_ms", "wait"),
    [(1500, " Wait 1.5 seconds before the next request."), (None, "")],
    ids=["asked to wait", "no wait asked"],
)
def test_a_rate_limited_record_names_the_wait(retry_after_ms: int | None, wait: str) -> None:
    value = baml_sdk.ai.errors.RateLimited(provider="openrouter", retry_after_ms=retry_after_ms, status_code=429)
    explained = translate_provider_error(BamlError(value, class_name="ai.errors.RateLimited"))
    assert isinstance(explained, RateLimitedError)
    assert str(explained).startswith(f"The openrouter provider limited the rate of the requests.{wait}\n")
    assert "The provider said" not in str(explained)


def test_a_network_failure_lists_the_checks() -> None:
    value = baml_sdk.ai.errors.NetworkFailure(provider="ollama", detail="connection refused")
    explained = translate_provider_error(BamlError(value, class_name="ai.errors.NetworkFailure"))
    assert isinstance(explained, ProviderUnreachableError)
    message = str(explained)
    assert message.startswith("A request did not reach the ollama provider.\nThe provider said: connection refused\n")
    assert "HTTPS_PROXY" in message
    assert "base_url" in message
    assert "  4. Check the status page of ollama, then run the call again." in message


def test_a_long_reply_is_cut_to_three_hundred_characters() -> None:
    explained = translate_provider_error(provider_error(401, "x" * 1000))
    assert f"The provider said: {'x' * 300}\n" in str(explained)
    assert "x" * 301 not in str(explained)


def test_a_record_without_text_quotes_nothing() -> None:
    explained = translate_provider_error(BamlError(object(), class_name="ai.errors.RateLimited"))
    assert isinstance(explained, RateLimitedError)
    assert "The openai" not in str(explained)
    assert str(explained).startswith("The LLM provider limited the rate of the requests.\n")
    assert "The provider said" not in str(explained)


@pytest.mark.parametrize(
    "error",
    [
        provider_error(500),
        provider_error(None),
        invalid_request("openai", "the model does not exist"),
        BamlError(
            baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason="MaxTokens", raw_output=""),
            class_name="ai.errors.FinishReasonError",
        ),
        BamlError("anything", class_name=None),
    ],
    ids=["server error", "no status", "other invalid request", "cut off reply", "no class name"],
)
def test_other_errors_have_no_explanation(error: BamlError) -> None:
    assert translate_provider_error(error) is None


def test_the_context_raises_the_explanation_from_the_baml_error() -> None:
    cause = invalid_request("openrouter", OPENROUTER_KEY)
    with pytest.raises(LLMRequestError) as raised, provider_errors():
        raise cause
    assert isinstance(raised.value, MissingAPIKeyError)
    assert raised.value.__cause__ is cause


def test_the_context_passes_other_baml_errors_unchanged() -> None:
    cause = provider_error(500)
    with pytest.raises(BamlError) as raised, provider_errors():
        raise cause
    assert raised.value is cause


def test_the_context_passes_other_exceptions_unchanged() -> None:
    error = KeyError("x")
    with pytest.raises(KeyError) as raised, provider_errors():
        raise error
    assert raised.value is error


def test_the_context_returns_normally_without_an_error() -> None:
    with provider_errors():
        pass
