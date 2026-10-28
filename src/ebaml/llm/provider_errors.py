"""Turn the errors of an LLM provider into ``EBaMlError`` subclasses whose messages list the steps that fix them.

The BAML runtime raises a ``BamlError`` that wraps a record of the failure, for example
``ai.errors.InvalidRequest(provider='openrouter', detail='openrouter needs an API key: ...')``. ``provider_errors``
maps the failures that the user can fix to the classes of ``ebaml.core.exceptions``. It leaves all other errors
unchanged.
"""

import contextlib
import re
from collections.abc import Callable, Generator, Mapping

from baml_bridge.errors import BamlError

from ebaml.core.exceptions import (
    APIKeyRejectedError,
    LLMRequestError,
    MissingAPIKeyError,
    ProviderUnreachableError,
    RateLimitedError,
)

_UNAUTHORIZED = frozenset({401, 403})
_RATE_LIMITED = 429
_QUOTE_CHARACTERS = 300
# the pages where a user creates a key; a provider that is not in the table gets a generic step
_KEY_PAGES: Mapping[str, str] = {
    "openrouter": "https://openrouter.ai/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
}
_VARIABLE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")


# nosemgrep: ebaml-no-object-type -- the BAML runtime record has no static type; each caller narrows the value
def _field(error: BamlError, name: str) -> object:
    """Read a field of the failure record that the error wraps.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.
    name : str
        The name of the field.

    Returns
    -------
    object
        The value, or ``None`` if the record has no such field.

    """
    return getattr(error.value, name, None)


def _provider(error: BamlError) -> str:
    """Name the provider of the failure.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    str
        The provider, or ``"LLM"`` if the record names none.

    """
    provider = _field(error, "provider")
    return provider if isinstance(provider, str) and provider else "LLM"


def _said(error: BamlError) -> str:
    """Quote what the provider said, as far as the record holds it.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    str
        A line that starts with ``"The provider said: "``, or an empty string if the record holds no text.

    """
    for name in ("detail", "message", "raw_body"):
        text = _field(error, name)
        if isinstance(text, str) and text.strip():
            return f"The provider said: {text.strip()[:_QUOTE_CHARACTERS]}\n"
    return ""


def _missing_key(error: BamlError) -> LLMRequestError:
    """Explain a missing API key.

    Parameters
    ----------
    error : BamlError
        An ``InvalidRequest`` whose detail says that the provider needs an API key.

    Returns
    -------
    LLMRequestError
        A ``MissingAPIKeyError`` that names the variable to export, as far as the detail names it.

    """
    provider = _provider(error)
    variables = list(dict.fromkeys(_VARIABLE.findall(str(_field(error, "detail")))))
    export = f"export {variables[0]}=..." if variables else "export <the variable of the provider>=..."
    page = _KEY_PAGES.get(provider)
    steps = [
        f"Create a key at {page}." if page else f"Create a key in the account that you have at {provider}.",
        f"Export it in the shell that runs your code: {export}",
        (
            f'To read the key from another variable, pass model=ModelOptions(provider="{provider}", model="...", '
            'api_key_env="NAME").'
        ),
    ]
    if len(variables) > 1:
        steps.insert(2, f"The provider reads any of these variables: {', '.join(variables)}.")
    listed = "\n".join(f"  {number}. {step}" for number, step in enumerate(steps, start=1))
    return MissingAPIKeyError(
        f"The {provider} provider needs an API key, and the environment has none.\nTo fix it:\n{listed}",
    )


def _from_status(error: BamlError) -> LLMRequestError | None:
    """Explain a failure by the HTTP status of the answer.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    LLMRequestError | None
        An error for status 401, 403 or 429, and ``None`` for any other status.

    """
    status = _field(error, "status_code")
    if status in _UNAUTHORIZED:
        return _rejected(error)
    if status == _RATE_LIMITED:
        return _rate_limited(error)
    return None


def _rejected(error: BamlError) -> LLMRequestError:
    """Explain a key that the provider rejects.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    LLMRequestError
        An ``APIKeyRejectedError``.

    """
    provider = _provider(error)
    return APIKeyRejectedError(
        f"The {provider} provider rejected the API key (HTTP {_field(error, 'status_code')}).\n"
        f"{_said(error)}"
        "To fix it:\n"
        "  1. Check that the variable holds the key of this provider, with no quote marks or spaces around it.\n"
        "  2. Check that the key is active, and that its account has credit and may use the model.\n"
        "  3. Create a new key if the old key has expired, and export it again in the shell that runs your code.",
    )


def _rate_limited(error: BamlError) -> LLMRequestError:
    """Explain a rate limit.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    LLMRequestError
        A ``RateLimitedError``. It names the wait that the provider asks for, as far as the record holds it.

    """
    provider = _provider(error)
    wait_ms = _field(error, "retry_after_ms")
    wait = f" Wait {wait_ms / 1000:g} seconds before the next request." if isinstance(wait_ms, int) else ""
    return RateLimitedError(
        f"The {provider} provider limited the rate of the requests.{wait}\n"
        f"{_said(error)}"
        "To fix it:\n"
        "  1. Lower the number of parallel requests, for example describe_ebm(ebm, max_concurrency=2).\n"
        "  2. Wait, and run the call again. A free model has a low limit.\n"
        '  3. Choose a paid model, for example model="provider/model", or another provider.',
    )


def _unreachable(error: BamlError) -> LLMRequestError:
    """Explain a request that did not reach the provider.

    Parameters
    ----------
    error : BamlError
        A ``NetworkFailure``.

    Returns
    -------
    LLMRequestError
        A ``ProviderUnreachableError``.

    """
    provider = _provider(error)
    return ProviderUnreachableError(
        f"A request did not reach the {provider} provider.\n"
        f"{_said(error)}"
        "To fix it:\n"
        "  1. Check the network connection of this machine.\n"
        "  2. If a proxy is needed, set HTTPS_PROXY in the shell that runs your code.\n"
        '  3. If you pass base_url in ModelOptions, check that it is correct, with the scheme, for example "https://".\n'
        f"  4. Check the status page of {provider}, then run the call again.",
    )


def _invalid_request(error: BamlError) -> LLMRequestError | None:
    """Explain an ``InvalidRequest``: a rejected key, or a missing key.

    Parameters
    ----------
    error : BamlError
        An ``InvalidRequest``.

    Returns
    -------
    LLMRequestError | None
        The explanation, or ``None`` if the request failed for another reason.

    """
    explained = _from_status(error)
    if explained is not None:
        return explained
    return _missing_key(error) if "needs an API key" in str(_field(error, "detail")) else None


# One row for each BAML error class that the user can fix. A new row adds a case without an edit of the code below.
_TRANSLATORS: Mapping[str, Callable[[BamlError], LLMRequestError | None]] = {
    "ai.errors.InvalidRequest": _invalid_request,
    "ai.errors.ProviderError": _from_status,
    "ai.errors.InvalidResponse": _from_status,
    "ai.errors.RateLimited": _rate_limited,
    "ai.errors.NetworkFailure": _unreachable,
}


def translate_provider_error(error: BamlError) -> LLMRequestError | None:
    """Explain an error of the BAML runtime, if the user can fix its cause.

    Parameters
    ----------
    error : BamlError
        The error of the BAML runtime.

    Returns
    -------
    LLMRequestError | None
        The explanation, or ``None`` if the error is of another kind (a reply cut off by ``max_tokens``, a refusal, a
        bug).

    """
    translator = _TRANSLATORS.get(error.class_name or "")
    return translator(error) if translator is not None else None


@contextlib.contextmanager
def provider_errors() -> Generator[None]:
    """Raise ``LLMRequestError`` instead of the ``BamlError`` for a failure that the user can fix.

    Any other ``BamlError`` passes unchanged. The ``BamlError`` is the ``__cause__`` of the new error.

    Yields
    ------
    None
        Control, for the body of the ``with`` statement.

    A request that fails for a reason that the user can fix raises an ``LLMRequestError``.

    Raises
    ------
    BamlError
        If a request of the body fails for another reason.

    """
    try:
        yield
    except BamlError as error:
        explained = translate_provider_error(error)
        if explained is None:
            raise
        raise explained from error
