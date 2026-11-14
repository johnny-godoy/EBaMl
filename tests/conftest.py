"""Fixtures and hooks for every test suite, and the local LLM server of the integration and contract tests.

The suites are ``unit``, ``integration``, ``api``, ``contract`` and ``docs`` (``.claude/rules/testing.md``). The hook
below marks each test with the name of its suite, so that ``pytest -m integration`` runs one suite and
``--strict-markers`` rejects a misspelled name.

The server speaks the chat completions protocol of OpenAI on the loopback interface. A test points a ``ModelOptions`` at
it, so the real BAML runtime sends its requests there, and the test reads them from ``FakeLLM.requests``.
"""

import http.server
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from typing import cast, override

import attrs
import orjson
import pytest

from ebaml import DescribeEBMRequest, ModelOptions
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.docs.conftest import training_data
from tests.unit.ebaml.conftest import PandasFrame


@attrs.frozen
class RecordedRequest:
    """One request that the server received.

    Attributes
    ----------
    path: str
        The path of the URL, for example ``/v1/chat/completions``.
    headers: Mapping[str, str]
        The headers, with lower-case names.
    body: Mapping[str, object]
        The parsed JSON body.

    """

    path: str
    headers: Mapping[str, str]
    body: Mapping[str, object]

    @property
    def messages(self) -> list[Mapping[str, object]]:
        """Read the messages of the conversation.

        Returns
        -------
        list[Mapping[str, object]]
            The messages, each with a ``role`` and a ``content``.

        """
        messages = self.body["messages"]
        assert isinstance(messages, list)
        return cast("list[Mapping[str, object]]", messages)

    @property
    def turns(self) -> int:
        """Count the turns of the conversation.

        Returns
        -------
        int
            The number of messages of the user.

        """
        return sum(message["role"] == "user" for message in self.messages)

    @property
    def prompt(self) -> str:
        """Join the text of every message of the request, in order.

        Returns
        -------
        str
            The text, one message after the other.

        """
        return "\n".join(_message_text(message) for message in self.messages)


def _message_text(message: Mapping[str, object]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    assert isinstance(content, list)
    parts = cast("list[Mapping[str, object]]", content)
    return "".join(str(part["text"]) for part in parts if "text" in part)


@attrs.frozen
class Reply:
    """What the server answers to one request.

    Attributes
    ----------
    status: int
        The HTTP status code.
    text: str
        The text of the model, for a status of 200.
    finish_reason: str
        Why the model stopped. ``"length"`` marks a reply that the token limit cut.
    usage: tuple[int, int] | None
        The input and the output tokens that the reply reports. ``None`` leaves the field out.
    error_message: str
        The message of the error, for another status.
    extra_fields: Mapping[str, object]
        Fields to add to the top level of the body. A real provider adds fields that the protocol does not list.

    """

    status: int = 200
    text: str = "A fake reply."
    finish_reason: str = "stop"
    usage: tuple[int, int] | None = (3, 2)
    error_message: str = "fake error"
    extra_fields: Mapping[str, object] = attrs.field(factory=dict)

    def payload(self, model: object) -> bytes:
        """Serialize the reply as the JSON that a provider sends.

        Returns
        -------
        bytes
            The JSON body.

        """
        if self.status != 200:
            return orjson.dumps({"error": {"message": self.error_message, "type": "fake_error", "code": self.status}})
        body: dict[str, object] = {
            "id": "fake-completion",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.text},
                    "finish_reason": self.finish_reason,
                },
            ],
        }
        if self.usage is not None:
            input_tokens, output_tokens = self.usage
            body["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        return orjson.dumps({**body, **self.extra_fields})


Handler = Callable[[RecordedRequest], Reply]


def _default_handler(_request: RecordedRequest) -> Reply:
    return Reply()


@attrs.define
class FakeLLM:
    """The state of a running server: the requests that it received and the function that answers them.

    Attributes
    ----------
    base_url: str
        The address to give to a model as ``base_url``.
    requests: list[RecordedRequest]
        Every request, in the order of arrival.
    handler: Handler
        Chooses the reply to a request. The default answers each request with ``Reply()``.

    """

    base_url: str
    requests: list[RecordedRequest] = attrs.field(factory=list)
    handler: Handler = attrs.field(default=_default_handler)

    @property
    def prompts(self) -> list[str]:
        """List the text of each request.

        Returns
        -------
        list[str]
            One entry for each request.

        """
        return [request.prompt for request in self.requests]


def _handler_class(state: FakeLLM) -> type[http.server.BaseHTTPRequestHandler]:
    class ChatHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = orjson.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request = RecordedRequest(
                path=self.path,
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
            state.requests.append(request)
            reply = state.handler(request)
            payload = reply.payload(body.get("model"))
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Stay silent, so that the test output shows no access log."""

    return ChatHandler


@contextmanager
def serve_fake_llm() -> Generator[FakeLLM]:
    """Run the server on a free port of the loopback interface, and stop it at the end.

    Yields
    ------
    FakeLLM
        The state of the server.

    """
    state = FakeLLM(base_url="")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(state))
    state.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


SUITES = ("unit", "integration", "api", "contract")
KEY_VARIABLE = "FAKE_LLM_KEY"
FAKE_KEY = "fake-key-of-the-tests"
PROVIDER_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with the suite of its directory."""
    for item in items:
        suite = item.path.relative_to(item.config.rootpath).parts[1:2]
        if suite and suite[0] in SUITES:
            item.add_marker(getattr(pytest.mark, suite[0]))


@pytest.fixture
def fake_llm() -> Iterator[FakeLLM]:
    """Run a local server that answers like an OpenAI-compatible provider.

    Yields
    ------
    FakeLLM
        The requests that the server received, and the handler that decides its replies.

    """
    with serve_fake_llm() as server:
        yield server


@pytest.fixture
def llm_model(fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> ModelOptions:
    """Point a model at the local server, and remove the keys of the real providers from the environment.

    A test that sends a request through this model reaches the local server only. A test that forgets the model finds
    no key, so a real provider never receives a request from a test.

    Returns
    -------
    ModelOptions
        The model: provider ``openai-generic``, with the address of the local server and a fake key.

    """
    for name in PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(KEY_VARIABLE, FAKE_KEY)
    return ModelOptions(
        provider="openai-generic",
        model="fake-model",
        api_key_env=KEY_VARIABLE,
        base_url=fake_llm.base_url,
    )


@pytest.fixture(scope="session")
def binary_data_and_model() -> tuple[PandasFrame, ExplainableBoostingClassifier]:
    """Fit a binary classifier of three features with few values, so that the graphs stay short.

    Returns
    -------
    tuple[PandasFrame, ExplainableBoostingClassifier]
        The training features and the fitted model.

    """
    features, target = training_data()
    ebm = ExplainableBoostingClassifier(interactions=0, random_state=0).fit(features, target)
    return features, ebm


@pytest.fixture
def describe_request(
    binary_data_and_model: tuple[PandasFrame, ExplainableBoostingClassifier],
    llm_model: ModelOptions,
) -> DescribeEBMRequest:
    """Build a request to describe the binary classifier through the local server.

    Returns
    -------
    DescribeEBMRequest
        The request, with a short summary.

    """
    _, ebm = binary_data_and_model
    return DescribeEBMRequest(ebm, model=llm_model, num_sentences=3)
