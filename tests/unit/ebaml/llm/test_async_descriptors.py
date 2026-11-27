"""Tests for the async form of each LLM call, and for its agreement with the sync form.

Each test of agreement runs the sync and the async function on the same request with the same fake replies. The fakes
replace the sync function ``<name>_sync`` and the async function ``<name>`` of ``baml_sdk`` with one recording fake.
"""

import asyncio
import io
import threading
from collections.abc import Callable
from typing import Any

import pandas as pd  # ruff: ignore[banned-api] -- the fixtures of the fitted models hold pandas frames
import pytest
from baml_bridge.errors import BamlError
from structlog.testing import capture_logs

import baml_sdk
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import (
    APIKeyRejectedError,
    LLMReplyError,
    LLMRequestError,
    MissingAPIKeyError,
    ProviderUnreachableError,
    RateLimitedError,
    VisionUnsupportedError,
)
from ebaml.llm import async_descriptors
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from tests.unit.ebaml.conftest import instance_of
from tests.unit.ebaml.llm.conftest import ASYNC, FORMS, SYNC, Form, fake_both_forms

USAGE: baml_sdk.TokenUsage = baml_sdk.TokenUsage(input_tokens=3, output_tokens=5, samples=[])


def _counted(text: str) -> baml_sdk.Counted:
    return baml_sdk.Counted(text=text, usage=USAGE)


def _missing_key() -> BamlError:
    value = baml_sdk.ai.errors.InvalidRequest(
        provider="openrouter",
        detail="openrouter needs an API key: set OPENROUTER_API_KEY",
    )
    return BamlError(value, class_name="ai.errors.InvalidRequest")


def _rejected_key() -> BamlError:
    value = baml_sdk.ai.errors.ProviderError(provider="openai", message="bad key", status_code=401)
    return BamlError(value, class_name="ai.errors.ProviderError")


def _unreachable() -> BamlError:
    value = baml_sdk.ai.errors.NetworkFailure(provider="openai", detail="connection refused")
    return BamlError(value, class_name="ai.errors.NetworkFailure")


def _rejected_image() -> BamlError:
    value = baml_sdk.ai.errors.InvalidRequest(
        provider="openrouter",
        status_code=404,
        detail="No endpoints found that support image input",
    )
    return BamlError(value, class_name="ai.errors.InvalidRequest")


def _rate_limited() -> BamlError:
    value = baml_sdk.ai.errors.RateLimited(provider="openrouter", retry_after_ms=2000, status_code=429)
    return BamlError(value, class_name="ai.errors.RateLimited")


def _cut_off() -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason="MaxTokens", raw_output="")
    return BamlError(value, class_name="ai.errors.FinishReasonError")


def _unparsed() -> BamlError:
    return BamlError(
        baml_sdk.baml.errors.ParseError(message="no JSON in the reply"),
        class_name="baml.errors.ParseError",
    )


# The error table of the plan of 0.15.0: the error that each failure of a graph request gives, and whether it stops
# describe_ebm at once. A reply that stays cut off, or that does not parse, concerns one graph.
FAILURES: dict[str, tuple[Callable[[], BamlError], type[LLMRequestError], bool]] = {
    "missing key": (_missing_key, MissingAPIKeyError, True),
    "rejected key": (_rejected_key, APIKeyRejectedError, True),
    "unreachable": (_unreachable, ProviderUnreachableError, True),
    "rejected image": (_rejected_image, VisionUnsupportedError, True),
    "rate limited": (_rate_limited, RateLimitedError, False),
    "cut off": (_cut_off, LLMReplyError, False),
    "unparsed": (_unparsed, LLMReplyError, False),
}


class Recorder:
    """Records the calls of the fake SDK functions, and answers each call with a scripted reply."""

    def __init__(self, replies: dict[str, Callable[[dict[str, Any]], Any]]) -> None:
        """Keep the reply of each function name."""
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._replies = replies
        self._lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the sync and the async form of each function of the replies."""
        for name in self._replies:
            fake_both_forms(monkeypatch, name, self._fake(name))

    def _fake(self, name: str) -> Callable[..., Any]:
        def fake(**kwargs: Any) -> Any:  # ruff: ignore[any-type] -- the reply has the type of the SDK function
            with self._lock:
                self.calls.append((name, kwargs))
            return self._replies[name](kwargs)

        return fake

    def names(self) -> list[str]:
        """List the names of the calls, in order.

        Returns
        -------
        list[str]
            The names.

        """
        return [name for name, _ in self.calls]


def _graph_reply(kwargs: dict[str, Any]) -> baml_sdk.Counted:
    # the text names the feature, so that a test can tell the descriptions apart
    graph: str = kwargs["graph"]
    return _counted(f"graph {graph.split(chr(10), maxsplit=1)[0][:20]}")


def _replies() -> dict[str, Callable[[dict[str, Any]], Any]]:
    rating = baml_sdk.FeatureDifficulty(feature="a", difficulty=baml_sdk.Difficulty.Hard, reason="r")
    return {
        "describe_graph_cot": _graph_reply,
        "summarize_ebm": lambda _kwargs: _counted("summary"),
        "explain_local": lambda _kwargs: baml_sdk.LocalExplanationText(
            counterfactual="c",
            sufficient_reason="s",
            completions="k",
            usage=USAGE,
        ),
        "propose_difficulty": lambda _kwargs: baml_sdk.DifficultyProposal(difficulties=[rating], usage=USAGE),
    }


@pytest.fixture
def fitted(continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame]) -> ExplainableBoostingClassifier:
    return continuous_ebm[0]


def _requests(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> dict[str, Callable[[Form], Any]]:
    ebm, data = continuous_ebm
    instance = instance_of(data, 0)
    return {
        "describe_graph": lambda form: form.describe_graph(DescribeGraphRequest(ebm, 0)),
        "describe_ebm": lambda form: form.describe_ebm(DescribeEBMRequest(ebm, max_concurrency=1)),
        "explain_instance": lambda form: form.explain_instance(ExplainInstanceRequest(ebm, instance, "G")),
        "propose_difficulties": lambda form: form.propose_difficulties(ProposeDifficultiesRequest(ebm, instance)),
    }


@pytest.mark.parametrize("function", ["describe_graph", "describe_ebm", "explain_instance", "propose_difficulties"])
def test_both_forms_give_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    function: str,
) -> None:
    Recorder(_replies()).install(monkeypatch)
    call = _requests(continuous_ebm)[function]
    assert call(SYNC) == call(ASYNC)


@pytest.mark.parametrize("function", ["describe_graph", "describe_ebm", "explain_instance", "propose_difficulties"])
def test_both_forms_make_the_same_sdk_calls(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    function: str,
) -> None:
    call = _requests(continuous_ebm)[function]
    recorded = []
    for form in FORMS:
        recorder = Recorder(_replies())
        recorder.install(monkeypatch)
        call(form)
        recorded.append(recorder.calls)
    sync_calls, async_calls = recorded
    # max_concurrency=1 runs the graphs of describe_ebm one at a time, so the order of the calls is fixed too
    assert sync_calls == async_calls


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_both_forms_of_describe_graph_raise_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    failure: str,
) -> None:
    make_error, _, _ = FAILURES[failure]

    def fail(_kwargs: dict[str, Any]) -> baml_sdk.Counted:
        raise make_error()

    Recorder({"describe_graph_cot": fail}).install(monkeypatch)
    raised = []
    for form in FORMS:
        with pytest.raises(Exception) as caught:  # ruff: ignore[pytest-raises-too-broad] -- the test compares the errors of both forms
            form.describe_graph(DescribeGraphRequest(fitted, 0))
        raised.append(caught.value)
    sync_error, async_error = raised
    assert (type(sync_error), str(sync_error)) == (type(async_error), str(async_error))


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_describe_ebm_raises_the_error_of_the_table_when_every_graph_fails(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
    failure: str,
) -> None:
    make_error, expected, _ = FAILURES[failure]

    def fail(_kwargs: dict[str, Any]) -> baml_sdk.Counted:
        raise make_error()

    recorder = Recorder({**_replies(), "describe_graph_cot": fail})
    recorder.install(monkeypatch)
    with pytest.raises(expected) as caught:
        form.describe_ebm(DescribeEBMRequest(fitted))
    assert "To fix it:" in str(caught.value)
    assert "summarize_ebm" not in recorder.names()


@pytest.mark.parametrize("failure", sorted(name for name, (_, _, stops) in FAILURES.items() if stops))
def test_an_error_that_affects_every_request_stops_describe_ebm_before_the_other_graphs(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
    failure: str,
) -> None:
    make_error, expected, _ = FAILURES[failure]

    def fail(_kwargs: dict[str, Any]) -> baml_sdk.Counted:
        raise make_error()

    recorder = Recorder({**_replies(), "describe_graph_cot": fail})
    recorder.install(monkeypatch)
    with pytest.raises(expected):
        form.describe_ebm(DescribeEBMRequest(fitted, max_concurrency=1))
    # three features, one at a time: the first error cancels the graphs that have not started
    assert recorder.names().count("describe_graph_cot") < len(fitted.feature_names_in_)


def _fail_feature(feature: str, make_error: Callable[[], BamlError]) -> Callable[[dict[str, Any]], baml_sdk.Counted]:
    def reply(kwargs: dict[str, Any]) -> baml_sdk.Counted:
        if f"Feature Name: {feature}\n" in kwargs["graph"]:
            raise make_error()
        return _graph_reply(kwargs)

    return reply


@pytest.mark.parametrize("failure", sorted(name for name, (_, _, stops) in FAILURES.items() if not stops))
def test_a_failed_feature_is_named_in_the_result_and_the_others_are_described(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
    failure: str,
) -> None:
    make_error, expected, _ = FAILURES[failure]
    recorder = Recorder({**_replies(), "describe_graph_cot": _fail_feature("b", make_error)})
    recorder.install(monkeypatch)
    description = form.describe_ebm(DescribeEBMRequest(fitted))
    assert description.text == "summary"
    assert [(failed.feature, type(failed.error)) for failed in description.failed_features] == [("b", expected)]


def test_the_summary_prompt_names_the_failed_feature_and_leaves_out_its_description(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
) -> None:
    # contract 4: the summary says which features it does not cover
    recorder = Recorder({**_replies(), "describe_graph_cot": _fail_feature("b", _rate_limited)})
    recorder.install(monkeypatch)
    form.describe_ebm(DescribeEBMRequest(fitted))
    (summary,) = [kwargs for name, kwargs in recorder.calls if name == "summarize_ebm"]
    assert summary["failed_features"] == ["b"]
    assert [line.split(":")[0] for line in summary["graph_descriptions"].split("\n\n")] == ["a", "c"]


def test_the_failed_features_are_logged(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
) -> None:
    Recorder({**_replies(), "describe_graph_cot": _fail_feature("b", _rate_limited)}).install(monkeypatch)
    with capture_logs() as logs:
        form.describe_ebm(DescribeEBMRequest(fitted))
    assert [log["features"] for log in logs if log["event"] == "features_failed"] == [["b"]]


def test_no_failure_gives_no_failed_feature_and_no_warning(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
) -> None:
    recorder = Recorder(_replies())
    recorder.install(monkeypatch)
    with capture_logs() as logs:
        description = form.describe_ebm(DescribeEBMRequest(fitted))
    assert description.failed_features == []
    assert not [log for log in logs if log["event"] == "features_failed"]
    (summary,) = [kwargs for name, kwargs in recorder.calls if name == "summarize_ebm"]
    assert summary["failed_features"] == []


def test_when_every_feature_fails_the_error_of_the_first_feature_is_raised(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
) -> None:
    def fail(kwargs: dict[str, Any]) -> baml_sdk.Counted:
        raise _rate_limited() if "Feature Name: a\n" in kwargs["graph"] else _unparsed()

    Recorder({**_replies(), "describe_graph_cot": fail}).install(monkeypatch)
    with pytest.raises(RateLimitedError):
        form.describe_ebm(DescribeEBMRequest(fitted))


def test_a_bug_in_a_graph_stops_describe_ebm(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    form: Form,
) -> None:
    def fail(_kwargs: dict[str, Any]) -> baml_sdk.Counted:
        msg = "a bug"
        raise RuntimeError(msg)

    Recorder({**_replies(), "describe_graph_cot": fail}).install(monkeypatch)
    with pytest.raises(RuntimeError, match="a bug"):
        form.describe_ebm(DescribeEBMRequest(fitted))


def test_describe_ebm_async_shows_one_step_for_each_graph_and_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    error_stream: io.StringIO,
) -> None:
    Recorder(_replies()).install(monkeypatch)
    asyncio.run(async_descriptors.describe_ebm_async(DescribeEBMRequest(fitted, progress_bar=True)))
    lines = error_stream.getvalue().splitlines()
    assert lines[0].startswith("describe_ebm ")
    assert "(0 of 4)" in lines[0]
    assert "(4 of 4)" in lines[-1]


def test_describe_ebm_async_shows_no_progress_bar_by_default(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    error_stream: io.StringIO,
) -> None:
    Recorder(_replies()).install(monkeypatch)
    asyncio.run(async_descriptors.describe_ebm_async(DescribeEBMRequest(fitted)))
    assert not error_stream.getvalue()


def test_describe_ebm_async_counts_the_summary_after_the_graphs(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    error_stream: io.StringIO,
) -> None:
    def summary(_kwargs: dict[str, Any]) -> baml_sdk.Counted:
        # the bar shows the three graphs and not the summary while the summary runs
        assert "(3 of 4)" in error_stream.getvalue()
        assert "(4 of 4)" not in error_stream.getvalue()
        return _counted("summary")

    Recorder({**_replies(), "summarize_ebm": summary}).install(monkeypatch)
    asyncio.run(async_descriptors.describe_ebm_async(DescribeEBMRequest(fitted, progress_bar=True)))
    assert "(4 of 4)" in error_stream.getvalue()


def test_describe_ebm_async_counts_a_failed_graph_as_a_step(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ExplainableBoostingClassifier,
    error_stream: io.StringIO,
) -> None:
    Recorder({**_replies(), "describe_graph_cot": _fail_feature("b", _rate_limited)}).install(monkeypatch)
    asyncio.run(async_descriptors.describe_ebm_async(DescribeEBMRequest(fitted, progress_bar=True)))
    assert "(4 of 4)" in error_stream.getvalue()
