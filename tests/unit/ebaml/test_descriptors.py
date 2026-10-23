"""Tests for the LLM descriptors."""

from types import SimpleNamespace

import baml_sdk
import pytest
from baml_bridge.errors import BamlError
from structlog.testing import capture_logs

from ebaml import descriptors, local
from ebaml.custom_types import Audience
from ebaml.exceptions import EBMGraphSimplificationError, InvalidOptionError
from ebaml.graphs import EBMGraph

DEFAULTS = {
    "expert_description": "expert",
    "graph_description": "g",
    "dataset_description": "d",
    "task_description": "t",
}

pytestmark: pytest.MarkDecorator = pytest.mark.usefixtures("default_options")


@pytest.fixture
def counted() -> baml_sdk.Counted:
    return baml_sdk.Counted(text="answer", usage=baml_sdk.TokenUsage(input_tokens=3, output_tokens=5))


@pytest.fixture
def fake_ebm() -> SimpleNamespace:
    return SimpleNamespace(feature_names_in_=["f0", "f1"], term_importances=lambda: [1.0, 2.0])


@pytest.fixture
def default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baml_sdk, "default_graph_options", lambda: baml_sdk.GraphPromptOptions(**DEFAULTS))


def test_prompt_options_defaults() -> None:
    assert descriptors._prompt_options({}).model_dump() == DEFAULTS


def test_prompt_options_overrides_and_ignores_none() -> None:
    opts = descriptors._prompt_options({"task_description": "new", "graph_description": None, "other": 1})
    assert opts.model_dump() == {**DEFAULTS, "task_description": "new"}


def test_graph_text_filters_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    class FakeGraph:
        @staticmethod
        def to_text(*, include_description: bool = True) -> str:
            seen["to_text"] = include_description
            return "text"

    def from_ebm(ebm: str, feature_index: int, *, use_feature_bounds: bool = True) -> FakeGraph:
        seen["from_ebm"] = (ebm, feature_index, use_feature_bounds)
        return FakeGraph()

    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(from_ebm))
    kwargs = {"use_feature_bounds": False, "include_description": False, "task_description": "x"}
    assert descriptors._graph_text("ebm", 4, kwargs) == "text"
    assert seen == {"from_ebm": ("ebm", 4, False), "to_text": False}


def test_log_usage(counted: baml_sdk.Counted) -> None:
    with capture_logs() as logs:
        assert descriptors._log_usage("what", counted) == "answer"
    assert [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs] == [("what", 3, 5)]


def test_describe_graph(monkeypatch: pytest.MonkeyPatch, fake_ebm: SimpleNamespace, counted: baml_sdk.Counted) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda _ebm, idx, kwargs: f"graph{idx}:{sorted(kwargs)}")
    calls = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", lambda **kw: calls.update(kw) or counted)
    assert descriptors.describe_graph(fake_ebm, 1, num_sentences=2, task_description="T") == "answer"
    assert calls["graph"] == "graph1:['task_description']"
    assert calls["num_sentences"] == 2
    assert calls["opts"].task_description == "T"


def test_describe_ebm(monkeypatch: pytest.MonkeyPatch, fake_ebm: SimpleNamespace, counted: baml_sdk.Counted) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda _ebm, idx, _kwargs: f"graph{idx}")
    graph_calls = []
    summary_calls = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", lambda **kw: graph_calls.append(kw) or counted)
    monkeypatch.setattr(baml_sdk, "summarize_ebm", lambda **kw: summary_calls.update(kw) or counted)
    assert descriptors.describe_ebm(fake_ebm, num_sentences=9, max_concurrency=2) == "answer"
    assert sorted(call["graph"] for call in graph_calls) == ["graph0", "graph1"]
    assert all(call["num_sentences"] == 7 for call in graph_calls)
    assert summary_calls["feature_importances"] == "f0: 1.00\nf1: 2.00"
    assert summary_calls["graph_descriptions"] == "f0: answer\n\nf1: answer"
    assert summary_calls["num_sentences"] == 9
    assert summary_calls["opts"].model_dump() == DEFAULTS


def test_describe_ebm_sums_usage(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda *_args: "g")
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", lambda **_kw: counted)
    monkeypatch.setattr(baml_sdk, "summarize_ebm", lambda **_kw: counted)
    with capture_logs() as logs:
        descriptors.describe_ebm(fake_ebm)
    # two graphs and the summary, 3 input and 5 output tokens each
    assert (logs[-1]["input_tokens"], logs[-1]["output_tokens"]) == (9, 15)


def test_describe_ebm_retries_a_truncated_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda _ebm, _idx, kwargs: "short" if kwargs else "x" * 300)
    seen = []

    def cot(*, graph: str, **_kw: object) -> baml_sdk.Counted:
        seen.append(graph)
        if graph != "short":
            error = _finish_error("MaxTokens")
            raise error
        return counted

    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    monkeypatch.setattr(baml_sdk, "summarize_ebm", lambda **_kw: counted)
    assert descriptors.describe_ebm(fake_ebm) == "answer"
    assert seen.count("short") == 2


def test_describe_graph_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda *_args: "g")
    calls = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", lambda **kw: calls.update(kw) or counted)
    descriptors.describe_graph(fake_ebm, 0)
    assert calls["num_sentences"] == 7


def test_describe_ebm_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", lambda *_args: "g")
    calls = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", lambda **_kw: counted)
    monkeypatch.setattr(baml_sdk, "summarize_ebm", lambda **kw: calls.update(kw) or counted)
    descriptors.describe_ebm(fake_ebm)
    assert calls["num_sentences"] == 30


def test_accepted_kwargs() -> None:
    def func(a: int, b: int) -> int:
        return a * b + 1

    assert descriptors._accepted_kwargs(func, {"a": 1, "c": 3}) == {"a": 1}


def test_feature_importances_to_text() -> None:
    ebm = SimpleNamespace(feature_names_in_=["a", "b"], term_importances=lambda: [1.234, 0.5])
    assert descriptors._feature_importances_to_text(ebm) == "a: 1.23\nb: 0.50"


def _facts() -> local.LocalFacts:
    prediction = local.Prediction(label=1, probability=0.8, lower=0.7, upper=0.9, logit=1.4, logit_std=0.2)
    return local.LocalFacts(prediction, [], None, None, [], 0.95)


def test_explain_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def compute(ebm: str, instance: dict, options: local.LocalOptions) -> local.LocalFacts:
        seen["compute"] = (ebm, instance, options)
        return _facts()

    monkeypatch.setattr(descriptors, "compute_local_facts", compute)
    monkeypatch.setattr(
        descriptors,
        "facts_to_texts",
        lambda _facts, audience: {
            "prediction": "p",
            "counterfactual": f"c-{audience}",
            "sufficient_reason": "s",
            "completions": "k",
        },
    )
    text = baml_sdk.LocalExplanationText(
        counterfactual="cf",
        sufficient_reason="sr",
        completions="cc",
        usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=2),
    )
    calls = {}
    monkeypatch.setattr(baml_sdk, "explain_local", lambda **kw: calls.update(kw) or text)
    with capture_logs() as logs:
        result = descriptors.explain_instance(
            "ebm",
            {"a": 1},
            "GLOBAL",
            audience="data_scientist",
            confidence_level=0.9,
            expert_description="E",
        )
    assert (result.counterfactual, result.sufficient_reason, result.completions) == ("cf", "sr", "cc")
    assert seen["compute"] == ("ebm", {"a": 1}, local.LocalOptions(confidence_level=0.9))
    assert calls["audience"] == "data_scientist"
    assert calls["global_explanation"] == "GLOBAL"
    assert calls["facts"].counterfactual == "c-data_scientist"
    assert calls["num_sentences"] == 5
    assert calls["opts"].expert_description == "E"
    assert [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs] == [("explain_instance", 1, 2)]


def test_explain_instance_defaults_to_layperson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(descriptors, "compute_local_facts", lambda *_a: _facts())
    audiences = []
    monkeypatch.setattr(
        descriptors,
        "facts_to_texts",
        lambda _f, audience: (
            audiences.append(audience)
            or dict.fromkeys(("prediction", "counterfactual", "sufficient_reason", "completions"), "x")
        ),
    )
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0),
    )
    calls = {}
    monkeypatch.setattr(baml_sdk, "explain_local", lambda **kw: calls.update(kw) or text)
    descriptors.explain_instance("ebm", {}, "G")
    assert calls["audience"] == "layperson"
    assert audiences == [Audience.LAYPERSON]


def test_explain_instance_rejects_unknown_audience() -> None:
    with pytest.raises(InvalidOptionError):
        descriptors.explain_instance("ebm", {}, "G", audience="child")


def _finish_error(reason: str) -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason=reason, raw_output="")
    return BamlError(value, class_name="ai.errors.FinishReasonError")


def test_describe_graph_simplifies_after_truncated_reply(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
    counted: baml_sdk.Counted,
) -> None:
    texts = []

    def graph_text(_ebm: object, _idx: int, kwargs: dict) -> str:
        texts.append(kwargs)
        return "x" * (900 if not kwargs else 300)

    replies = [_finish_error("MaxTokens"), counted]

    def cot(**_kw: object) -> baml_sdk.Counted:
        reply = replies.pop(0)
        if isinstance(reply, BamlError):
            raise reply
        return reply

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    assert descriptors.describe_graph(fake_ebm, 0) == "answer"
    # estimate is 300 tokens, so the retry asks for 75% of it with no tolerance
    assert texts == [{}, {"max_tokens": 225, "overbudget_percent": 0}]


def test_describe_graph_does_not_retry_other_errors(monkeypatch: pytest.MonkeyPatch, fake_ebm: SimpleNamespace) -> None:
    def cot(**_kw: object) -> baml_sdk.Counted:
        error = _finish_error("Refused")
        raise error

    monkeypatch.setattr(descriptors, "_graph_text", lambda *_args: "g")
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    with pytest.raises(BamlError):
        descriptors.describe_graph(fake_ebm, 0)


def test_describe_graph_reraises_when_graph_cannot_shrink(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: SimpleNamespace,
) -> None:
    def graph_text(_ebm: object, _idx: int, kwargs: dict) -> str:
        if kwargs:
            msg = "too complex"
            raise EBMGraphSimplificationError(msg)
        return "g"

    def cot(**_kw: object) -> baml_sdk.Counted:
        error = _finish_error("MaxTokens")
        raise error

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    with pytest.raises(BamlError):
        descriptors.describe_graph(fake_ebm, 0)
