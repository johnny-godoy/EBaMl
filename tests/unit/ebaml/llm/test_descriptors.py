"""Tests for the LLM descriptors."""

import inspect
from typing import Any

import pytest
from baml_bridge.errors import BamlError
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
from structlog.testing import capture_logs

import baml_sdk
from ebaml.core.custom_types import Audience, FeatureType, Normalization
from ebaml.core.exceptions import EBMGraphSimplificationError, InvalidOptionError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.llm import descriptors
from ebaml.llm.options import GraphOptions, PromptOptions
from ebaml.local import facts as local
from tests.unit.ebaml.conftest import appending, fake_model, recording, returning

FAKE_EBM: ExplainableBoostingClassifier = fake_model()

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
def fake_ebm() -> ExplainableBoostingClassifier:
    return fake_model(feature_names_in_=["f0", "f1"], term_importances=lambda: [1.0, 2.0])


@pytest.fixture
def default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baml_sdk, "default_graph_options", lambda: baml_sdk.GraphPromptOptions(**DEFAULTS))


def test_prompt_options_defaults() -> None:
    assert descriptors._prompt_options(PromptOptions()).model_dump() == DEFAULTS


def test_prompt_options_overrides_and_ignores_none() -> None:
    opts = descriptors._prompt_options(PromptOptions(task_description="new", graph_description=None))
    assert opts.model_dump() == {**DEFAULTS, "task_description": "new"}


def test_prompt_options_override_every_description() -> None:
    options = PromptOptions(
        expert_description="e",
        graph_description="g2",
        dataset_description="d2",
        task_description="t2",
    )
    assert descriptors._prompt_options(options).model_dump() == {
        "expert_description": "e",
        "graph_description": "g2",
        "dataset_description": "d2",
        "task_description": "t2",
    }


def test_graph_text_forwards_every_option(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeGraph:
        @staticmethod
        def to_text(**kwargs: object) -> str:
            seen["to_text"] = kwargs
            return "text"

    def from_ebm(
        ebm: ExplainableBoostingClassifier,
        feature_index: int,
        normalization: Normalization,
        *,
        use_feature_bounds: bool,
    ) -> FakeGraph:
        seen["from_ebm"] = (ebm, feature_index, normalization, use_feature_bounds)
        return FakeGraph()

    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(from_ebm))
    options = GraphOptions(
        normalization=Normalization.Z_SCORE,
        use_feature_bounds=False,
        include_description=False,
        feature_format=FeatureType.BOOLEAN,
        x_axis_precision=2,
        y_axis_precision=3,
        confidence_bounds=False,
        confidence_level=0.8,
        max_tokens=11,
        overbudget_percent=7,
    )
    assert descriptors._graph_text(FAKE_EBM, 4, options) == "text"
    assert seen == {
        "from_ebm": (FAKE_EBM, 4, Normalization.Z_SCORE, False),
        "to_text": {
            "include_description": False,
            "feature_format": FeatureType.BOOLEAN,
            "x_axis_precision": 2,
            "y_axis_precision": 3,
            "confidence_bounds": False,
            "confidence_level": 0.8,
            "max_tokens": 11,
            "overbudget_percent": 7,
            "check_budget": baml_sdk.check_token_budget,
        },
    }


def test_log_usage(counted: baml_sdk.Counted) -> None:
    with capture_logs() as logs:
        assert descriptors._log_usage("what", counted) == "answer"
    assert [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs] == [("what", 3, 5)]


def test_describe_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def graph_text(_ebm: object, idx: int, options: GraphOptions) -> str:
        return f"graph{idx}:{options.max_tokens}"

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", recording(calls, counted))
    result = descriptors.describe_graph(
        fake_ebm,
        1,
        num_sentences=2,
        graph=GraphOptions(max_tokens=5),
        prompt=PromptOptions(task_description="T"),
    )
    assert result == "answer"
    assert calls["graph"] == "graph1:5"
    assert calls["num_sentences"] == 2
    assert calls["opts"].task_description == "T"


def test_describe_ebm(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def graph_text(_ebm: object, idx: int, _options: GraphOptions) -> str:
        return f"graph{idx}"

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", appending(graph_calls, counted))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", recording(summary_calls, counted))
    assert descriptors.describe_ebm(fake_ebm, num_sentences=9, max_concurrency=2) == "answer"
    assert sorted(call["graph"] for call in graph_calls) == ["graph0", "graph1"]
    assert all(call["num_sentences"] == 7 for call in graph_calls)
    assert summary_calls["feature_importances"] == "f0: 1.00\nf1: 2.00"
    assert summary_calls["graph_descriptions"] == "f0: answer\n\nf1: answer"
    assert summary_calls["num_sentences"] == 9
    assert summary_calls["opts"].model_dump() == DEFAULTS


def test_describe_ebm_sums_usage(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", returning(counted))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", returning(counted))
    with capture_logs() as logs:
        descriptors.describe_ebm(fake_ebm)
    # two graphs and the summary, 3 input and 5 output tokens each
    assert (logs[-1]["input_tokens"], logs[-1]["output_tokens"]) == (9, 15)


def test_describe_ebm_retries_a_truncated_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def graph_text(_ebm: object, _idx: int, options: GraphOptions) -> str:
        return "x" * 300 if options == GraphOptions() else "short"

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    seen = []

    def cot(*, graph: str, **_kw: object) -> baml_sdk.Counted:
        seen.append(graph)
        if graph != "short":
            error = _finish_error("MaxTokens")
            raise error
        return counted

    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    monkeypatch.setattr(baml_sdk, "summarize_ebm", returning(counted))
    assert descriptors.describe_ebm(fake_ebm) == "answer"
    assert seen.count("short") == 2


def test_describe_graph_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", recording(calls, counted))
    descriptors.describe_graph(fake_ebm, 0)
    assert calls["num_sentences"] == 7


def test_describe_ebm_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", returning(counted))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", recording(calls, counted))
    descriptors.describe_ebm(fake_ebm)
    assert calls["num_sentences"] == 30


def test_feature_importances_to_text() -> None:
    ebm = fake_model(feature_names_in_=["a", "b"], term_importances=lambda: [1.234, 0.5])
    assert descriptors._feature_importances_to_text(ebm) == "a: 1.23\nb: 0.50"


def _facts() -> local.LocalFacts:
    prediction = local.Prediction(label=1, probability=0.8, lower=0.7, upper=0.9, logit=1.4, logit_std=0.2)
    return local.LocalFacts(prediction, [], None, None, [], 0.95)


def test_explain_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def compute(ebm: str, instance: dict[str, object], options: local.LocalOptions) -> local.LocalFacts:
        seen["compute"] = (ebm, instance, options)
        return _facts()

    monkeypatch.setattr(descriptors, "compute_local_facts", compute)

    def facts_to_texts(_facts: local.LocalFacts, audience: Audience) -> dict[str, str]:
        return {"prediction": "p", "counterfactual": f"c-{audience}", "sufficient_reason": "s", "completions": "k"}

    monkeypatch.setattr(descriptors, "facts_to_texts", facts_to_texts)
    text = baml_sdk.LocalExplanationText(
        counterfactual="cf",
        sufficient_reason="sr",
        completions="cc",
        usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=2),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local", recording(calls, text))
    with capture_logs() as logs:
        result = descriptors.explain_instance(
            FAKE_EBM,
            {"a": 1},
            "GLOBAL",
            audience="data_scientist",
            local=local.LocalOptions(confidence_level=0.9),
            prompt=PromptOptions(expert_description="E"),
        )
    assert (result.counterfactual, result.sufficient_reason, result.completions) == ("cf", "sr", "cc")
    assert seen["compute"] == (FAKE_EBM, {"a": 1}, local.LocalOptions(confidence_level=0.9))
    assert calls["audience"] == "data_scientist"
    assert calls["global_explanation"] == "GLOBAL"
    assert calls["facts"].counterfactual == "c-data_scientist"
    assert calls["num_sentences"] == 5
    assert calls["opts"].expert_description == "E"
    assert [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs] == [("explain_instance", 1, 2)]


def test_explain_instance_defaults_to_layperson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    audiences: list[Audience] = []

    def facts_to_texts(_facts: local.LocalFacts, audience: Audience) -> dict[str, str]:
        audiences.append(audience)
        return dict.fromkeys(("prediction", "counterfactual", "sufficient_reason", "completions"), "x")

    monkeypatch.setattr(descriptors, "facts_to_texts", facts_to_texts)
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local", recording(calls, text))
    descriptors.explain_instance(FAKE_EBM, {}, "G")
    assert calls["audience"] == "layperson"
    assert audiences == [Audience.LAYPERSON]


def test_explain_instance_rejects_unknown_audience() -> None:
    with pytest.raises(InvalidOptionError):
        descriptors.explain_instance(FAKE_EBM, {}, "G", audience="child")


def _finish_error(reason: str) -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason=reason, raw_output="")
    return BamlError(value, class_name="ai.errors.FinishReasonError")


def test_describe_graph_simplifies_after_truncated_reply(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    texts = []

    def graph_text(_ebm: object, _idx: int, options: GraphOptions) -> str:
        texts.append(options)
        return "x" * (900 if options == GraphOptions() else 300)

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
    assert texts == [GraphOptions(), GraphOptions(max_tokens=225, overbudget_percent=0)]


def test_describe_graph_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def cot(**_kw: object) -> baml_sdk.Counted:
        error = _finish_error("Refused")
        raise error

    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    with pytest.raises(BamlError):
        descriptors.describe_graph(fake_ebm, 0)


def test_describe_graph_reraises_when_graph_cannot_shrink(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def graph_text(_ebm: object, _idx: int, options: GraphOptions) -> str:
        if options != GraphOptions():
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


def _finish_error_of(class_name: str, reason: str) -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason=reason, raw_output="")
    return BamlError(value, class_name=class_name)


FINISH_CLASS = "ai.errors.FinishReasonError"
MAX_TOKENS = "MaxTokens"


@pytest.mark.parametrize(
    ("class_name", "reason", "expected"),
    [
        (FINISH_CLASS, MAX_TOKENS, True),
        (FINISH_CLASS, "Refused", False),
        (FINISH_CLASS, "Complete", False),  # sorts before the reason
        (FINISH_CLASS, "ToolUse", False),  # sorts after the reason
        ("ai.errors.A", MAX_TOKENS, False),  # sorts before the class name
        ("ai.errors.Z", MAX_TOKENS, False),  # sorts after the class name
    ],
)
def test_is_truncated(class_name: str, reason: str, *, expected: bool) -> None:
    assert descriptors._is_truncated(_finish_error_of(class_name, reason)) is expected


def test_describe_graph_stops_after_four_retries(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    graphs = iter(f"graph{index}" for index in range(10))
    calls = []

    def cot(*, graph: str, **_kw: object) -> baml_sdk.Counted:
        calls.append(graph)
        error = _finish_error("MaxTokens")
        raise error

    def graph_text(*_args: object) -> str:
        return next(graphs)

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    with capture_logs() as logs, pytest.raises(BamlError):
        descriptors.describe_graph(fake_ebm, 0)
    # the first call and four retries, each on a new graph
    assert calls == [f"graph{index}" for index in range(5)]
    assert [log["retry"] for log in logs if log["event"] == "truncated_reply"] == [1, 2, 3, 4]


def test_describe_graph_asks_once_when_the_error_is_not_a_truncation(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    calls = []

    def cot(**_kw: object) -> baml_sdk.Counted:
        calls.append(1)
        error = _finish_error("Refused")
        raise error

    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", cot)
    with pytest.raises(BamlError):
        descriptors.describe_graph(fake_ebm, 0)
    assert calls == [1]


def test_describe_ebm_adds_the_usage_of_every_call(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    # The numbers share bits, so a bitwise or / xor in place of the sum gives another total.
    graph = baml_sdk.Counted(text="g", usage=baml_sdk.TokenUsage(input_tokens=6, output_tokens=7))
    summary = baml_sdk.Counted(text="s", usage=baml_sdk.TokenUsage(input_tokens=3, output_tokens=5))
    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", returning(graph))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", returning(summary))
    with capture_logs() as logs:
        descriptors.describe_ebm(fake_ebm)
    assert (logs[-1]["input_tokens"], logs[-1]["output_tokens"]) == (3 + 6 * 2, 5 + 7 * 2)


def test_describe_ebm_runs_eight_graphs_at_a_time_by_default(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    workers = []

    class RecordingPool(descriptors.ThreadPoolExecutor):
        def __init__(self, max_workers: int) -> None:
            workers.append(max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(descriptors, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(descriptors, "_graph_text", returning("g"))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", returning(counted))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", returning(counted))
    descriptors.describe_ebm(fake_ebm)
    assert workers == [8]


def test_describe_ebm_passes_the_options_to_every_graph_and_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    graph_options: list[GraphOptions] = []
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}

    def graph_text(_ebm: object, _idx: int, options: GraphOptions) -> str:
        graph_options.append(options)
        return "g"

    monkeypatch.setattr(descriptors, "_graph_text", graph_text)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot", appending(graph_calls, counted))
    monkeypatch.setattr(baml_sdk, "summarize_ebm", recording(summary_calls, counted))
    options = GraphOptions(max_tokens=9)
    descriptors.describe_ebm(fake_ebm, graph=options, prompt=PromptOptions(dataset_description="D"))
    assert graph_options == [options, options]
    assert [call["opts"].dataset_description for call in graph_calls] == ["D", "D"]
    assert summary_calls["opts"].dataset_description == "D"


def test_explain_instance_uses_the_default_local_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[local.LocalOptions] = []

    def compute(_ebm: object, _instance: object, options: local.LocalOptions) -> local.LocalFacts:
        seen.append(options)
        return _facts()

    monkeypatch.setattr(descriptors, "compute_local_facts", compute)
    monkeypatch.setattr(
        descriptors,
        "facts_to_texts",
        returning(dict.fromkeys(("prediction", "counterfactual", "sufficient_reason", "completions"), "x")),
    )
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0),
    )
    monkeypatch.setattr(baml_sdk, "explain_local", returning(text))
    descriptors.explain_instance(FAKE_EBM, {}, "G")
    assert seen == [local.LocalOptions()]


def test_option_arguments_are_keyword_only() -> None:
    for function in (descriptors.describe_graph, descriptors.describe_ebm, descriptors.explain_instance):
        kinds = {
            name: parameter.kind
            for name, parameter in inspect.signature(function).parameters.items()
            if name in {"graph", "prompt", "local"}
        }
        assert set(kinds.values()) == {inspect.Parameter.KEYWORD_ONLY}
