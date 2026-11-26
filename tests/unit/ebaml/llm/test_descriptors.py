"""Tests for the LLM descriptors."""

import ast
import base64
import io
import re
import sys
import time
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, override

import orjson
import pytest
from baml_bridge.errors import BamlError
from structlog.testing import capture_logs

import baml_sdk
from ebaml.core.custom_types import FeatureType, GraphMode, Normalization, VisionFallback
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ebaml.core.exceptions import (
    EBMGraphSimplificationError,
    InvalidOptionError,
    MissingAPIKeyError,
    MissingExtraError,
    UnsupportedModelError,
    VisionFallbackWarning,
    VisionUnsupportedError,
)
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.graph_image import GraphImageOptions, graph_png
from ebaml.graphs.model_kinds import model_kind
from ebaml.graphs.token_budget import check_token_budget, estimate_tokens
from ebaml.graphs.token_weights import ModelKey, TokenWeights, WeightsEntry, WeightsTable
from ebaml.llm import descriptors
from ebaml.llm.descriptors import EBMDescription
from ebaml.llm.graph_views import RenderedGraph
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.token_learning import TokenLearner
from ebaml.local import fact_types as local
from ebaml.local.audience import Audience, DataScientist, Layperson
from ebaml.local.fact_text import FactTexts
from tests.unit.ebaml.conftest import (
    PARTIAL_KINDS,
    REMOVED_FEATURE,
    PandasFrame,
    appending,
    fake_model,
    fake_sdk,
    instance_of,
    recording,
    returning,
    same_type,
    sdk_fields,
    sdk_function_names,
    sdk_signature,
)

FAKE_EBM: ExplainableBoostingClassifier = fake_model()

DEFAULTS = {
    "expert_description": "expert",
    "graph_description": "g",
    "dataset_description": "d",
    "task_description": "t",
}

pytestmark: pytest.MarkDecorator = pytest.mark.usefixtures("default_options")

# a graph for the fakes of _render_graph; the tests of the text mode read only its text
PLAIN_GRAPH: EBMGraph = EBMGraph("f", "continuous", [(0.0, 1.0)], [0.0], [0.0])


def rendered(text: str) -> RenderedGraph:
    return RenderedGraph(extracted=PLAIN_GRAPH, shown=PLAIN_GRAPH, text=text)


def rendering(graph_text: Callable[[object, int, GraphOptions, object], str]) -> Callable[..., RenderedGraph]:
    """Turn a fake of the graph text into a fake of ``_render_graph``.

    Returns
    -------
    Callable[..., RenderedGraph]
        A function with the arguments of ``_render_graph`` that renders the text of ``graph_text``.

    """

    def render(ebm: object, feature_index: int, options: GraphOptions, model: object) -> RenderedGraph:
        return rendered(graph_text(ebm, feature_index, options, model))

    return render


@pytest.fixture
def counted() -> baml_sdk.Counted:
    return baml_sdk.Counted(text="answer", usage=baml_sdk.TokenUsage(input_tokens=3, output_tokens=5, samples=[]))


@pytest.fixture
def fake_ebm() -> ExplainableBoostingClassifier:
    return fake_model(
        feature_names_in_=["f0", "f1"],
        term_names_=["f0", "f1"],
        term_features_=[(0,), (1,)],
        term_importances=lambda: [1.0, 2.0],
    )


@pytest.fixture
def default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        baml_sdk,
        "default_graph_options_sync",
        fake_sdk("default_graph_options_sync", lambda: baml_sdk.GraphPromptOptions(**DEFAULTS)),
    )


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


def test_render_graph_forwards_every_option(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def to_text(_graph: EBMGraph, *, on_render: Callable[[EBMGraph], None], **kwargs: object) -> str:
        seen["to_text"] = kwargs
        on_render(PLAIN_GRAPH)
        return "text"

    def from_ebm(
        ebm: ExplainableBoostingClassifier,
        feature_index: int,
        normalization: Normalization,
        *,
        use_feature_bounds: bool,
        classes: Sequence[str] | None,
    ) -> EBMGraph:
        seen["from_ebm"] = (ebm, feature_index, normalization, use_feature_bounds, classes)
        return PLAIN_GRAPH

    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(from_ebm))
    monkeypatch.setattr(EBMGraph, "to_text", to_text)
    options = GraphOptions(
        normalization=Normalization.Z_SCORE,
        use_feature_bounds=False,
        classes=("b", "a"),
        include_description=False,
        feature_format=FeatureType.BOOLEAN,
        x_axis_precision=2,
        y_axis_precision=3,
        confidence_bounds=False,
        confidence_level=0.8,
        max_tokens=11,
        overbudget_percent=7,
    )
    result = descriptors._render_graph(FAKE_EBM, 4, options, None)
    assert (result.text, result.shown) == ("text", PLAIN_GRAPH)
    assert seen == {
        "from_ebm": (FAKE_EBM, 4, Normalization.Z_SCORE, False, ("b", "a")),
        "to_text": {
            "include_description": False,
            "feature_format": FeatureType.BOOLEAN,
            "x_axis_precision": 2,
            "y_axis_precision": 3,
            "confidence_bounds": False,
            "confidence_level": 0.8,
            "max_tokens": 11,
            "overbudget_percent": 7,
            "check_budget": check_token_budget,
        },
    }


def test_log_usage(counted: baml_sdk.Counted) -> None:
    with capture_logs() as logs:
        assert descriptors._log_usage("what", counted, truncated_attempts=2) == "answer"
    assert [(log["what"], log["input_tokens"], log["output_tokens"], log["truncated_attempts"]) for log in logs] == [
        ("what", 3, 5, 2),
    ]


def test_describe_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def graph_text(_ebm: object, idx: int, options: GraphOptions, _model: object) -> str:
        return f"graph{idx}:{options.max_tokens}"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", recording(calls, counted)),
    )
    result = descriptors.describe_graph(
        DescribeGraphRequest(
            fake_ebm,
            1,
            num_sentences=2,
            graph=GraphOptions(max_tokens=5),
            prompt=PromptOptions(task_description="T"),
        ),
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
    def graph_text(_ebm: object, idx: int, _options: GraphOptions, _model: object) -> str:
        return f"graph{idx}"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    assert descriptors.describe_ebm(DescribeEBMRequest(fake_ebm, num_sentences=9, max_concurrency=2)) == EBMDescription(
        "answer",
        [],
        [],
    )
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
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    with capture_logs() as logs:
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    # two graphs and the summary, 3 input and 5 output tokens each
    assert (logs[-1]["input_tokens"], logs[-1]["output_tokens"]) == (9, 15)


def test_describe_ebm_retries_a_truncated_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def graph_text(_ebm: object, _idx: int, options: GraphOptions, _model: object) -> str:
        return "x" * 300 if options == GraphOptions() else "short"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    seen = []

    def cot(*, graph: str, **_kw: object) -> baml_sdk.Counted:
        seen.append(graph)
        if graph != "short":
            error = _finish_error("MaxTokens")
            raise error
        return counted

    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    assert descriptors.describe_ebm(DescribeEBMRequest(fake_ebm)).text == "answer"
    assert seen.count("short") == 2


def test_describe_graph_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", recording(calls, counted)),
    )
    descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    assert calls["num_sentences"] == 7


def test_describe_ebm_defaults(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", recording(calls, counted)))
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert calls["num_sentences"] == 30


def _facts() -> local.LocalFacts:
    prediction = local.Prediction(decision=1, output=0.8, lower=0.7, upper=0.9, score=1.4, score_std=0.2, margin=1.4)
    return local.LocalFacts(prediction, [], [], None, [], 0.95, min_support=0.01, n_samples=10_000)


def test_explain_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def compute(ebm: str, instance: dict[str, object], options: local.LocalOptions) -> local.LocalFacts:
        seen["compute"] = (ebm, instance, options)
        return _facts()

    monkeypatch.setattr(descriptors, "compute_local_facts", compute)

    def facts_to_texts(_facts: local.LocalFacts, audience: Audience) -> FactTexts:
        return FactTexts("p", f"c-{audience.guidance()}", "s", "k")

    monkeypatch.setattr(descriptors, "facts_to_texts", facts_to_texts)
    text = baml_sdk.LocalExplanationText(
        counterfactual="cf",
        sufficient_reason="sr",
        completions="cc",
        usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=2, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", recording(calls, text)))
    with capture_logs() as logs:
        result = descriptors.explain_instance(
            ExplainInstanceRequest(
                FAKE_EBM,
                {"a": 1},
                "GLOBAL",
                audience="data_scientist",
                local=local.LocalOptions(confidence_level=0.9),
                prompt=PromptOptions(expert_description="E"),
            ),
        )
    assert (result.counterfactual, result.sufficient_reason, result.completions) == ("cf", "sr", "cc")
    assert seen["compute"] == (FAKE_EBM, {"a": 1}, local.LocalOptions(confidence_level=0.9))
    assert calls["audience_guidance"] == DataScientist().guidance()
    assert calls["global_explanation"] == "GLOBAL"
    assert calls["facts"].counterfactual == f"c-{DataScientist().guidance()}"
    assert calls["num_sentences"] == 5
    assert calls["opts"].expert_description == "E"
    assert [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs] == [("explain_instance", 1, 2)]


def test_explain_instance_defaults_to_layperson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    audiences: list[Audience] = []

    def facts_to_texts(_facts: local.LocalFacts, audience: Audience) -> FactTexts:
        audiences.append(audience)
        return FactTexts("x", "x", "x", "x")

    monkeypatch.setattr(descriptors, "facts_to_texts", facts_to_texts)
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", recording(calls, text)))
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G"))
    assert calls["audience_guidance"] == Layperson().guidance()
    assert [type(audience) for audience in audiences] == [Layperson]


def test_explain_instance_uses_a_custom_audience(monkeypatch: pytest.MonkeyPatch) -> None:
    class Regulator(DataScientist):
        @override
        def guidance(self) -> str:
            return "REGULATOR"

    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", recording(calls, text)))
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G", audience=Regulator()))
    assert calls["audience_guidance"] == "REGULATOR"
    assert "probability" in calls["facts"].prediction


def test_explain_instance_passes_the_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", recording(calls, text)))
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G"))
    assert calls["language"] == "English"
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G", language="French"))
    assert calls["language"] == "French"


def test_explain_instance_rejects_unknown_audience() -> None:
    with pytest.raises(InvalidOptionError):
        descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G", audience="child"))


def _finish_error(reason: str) -> BamlError:
    value = baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason=reason, raw_output="")
    return BamlError(value, class_name="ai.errors.FinishReasonError")


def test_describe_graph_simplifies_after_truncated_reply(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    texts = []

    def graph_text(_ebm: object, _idx: int, options: GraphOptions, _model: object) -> str:
        texts.append(options)
        return "1.5 " * (900 if options == GraphOptions() else 300)

    replies = [_finish_error("MaxTokens"), counted]

    def cot(**_kw: object) -> baml_sdk.Counted:
        reply = replies.pop(0)
        if isinstance(reply, BamlError):
            raise reply
        return reply

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    assert descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0)) == "answer"
    # the retry asks for 75% of the estimate of the first text, with no tolerance
    assert texts == [
        GraphOptions(),
        GraphOptions(max_tokens=int(estimate_tokens("1.5 " * 900) * 0.75), overbudget_percent=0),
    ]


def test_describe_graph_logs_the_truncated_attempts_with_the_usage(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    # regression: the usage showed the last attempt only, with no sign that cut-off attempts were left out
    replies = [_finish_error("MaxTokens"), _finish_error("MaxTokens"), counted]

    def cot(**_kw: object) -> baml_sdk.Counted:
        reply = replies.pop(0)
        if isinstance(reply, BamlError):
            raise reply
        return reply

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with capture_logs() as logs:
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    assert [log["truncated_attempts"] for log in logs if log["event"] == "usage"] == [2]


def test_describe_graph_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def cot(**_kw: object) -> baml_sdk.Counted:
        error = _finish_error("Refused")
        raise error

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with pytest.raises(BamlError):
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))


def test_describe_graph_reraises_when_graph_cannot_shrink(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def graph_text(_ebm: object, _idx: int, options: GraphOptions, _model: object) -> str:
        if options != GraphOptions():
            msg = "too complex"
            raise EBMGraphSimplificationError(msg)
        return "g"

    def cot(**_kw: object) -> baml_sdk.Counted:
        error = _finish_error("MaxTokens")
        raise error

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with pytest.raises(BamlError):
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))


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

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with capture_logs() as logs, pytest.raises(BamlError):
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
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

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with pytest.raises(BamlError):
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    assert calls == [1]


def test_describe_ebm_adds_the_usage_of_every_call(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    # The numbers share bits, so a bitwise or / xor in place of the sum gives another total.
    graph = baml_sdk.Counted(text="g", usage=baml_sdk.TokenUsage(input_tokens=6, output_tokens=7, samples=[]))
    summary = baml_sdk.Counted(text="s", usage=baml_sdk.TokenUsage(input_tokens=3, output_tokens=5, samples=[]))
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(graph)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(summary)))
    with capture_logs() as logs:
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert (logs[-1]["input_tokens"], logs[-1]["output_tokens"]) == (3 + 6 * 2, 5 + 7 * 2)


def test_describe_ebm_adds_the_truncated_attempts_of_every_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    # each of the two graphs is cut off once, and a third attempt succeeds
    failures = {"g0": 1, "g1": 1}

    def cot(*, graph: str, **_kw: object) -> baml_sdk.Counted:
        if failures[graph]:
            failures[graph] -= 1
            reason = "MaxTokens"
            raise _finish_error(reason)
        return counted

    def graph_text(_ebm: object, idx: int, _options: GraphOptions, _model: object) -> str:
        return f"g{idx}"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    with capture_logs() as logs:
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert [log["truncated_attempts"] for log in logs if log["event"] == "usage"] == [2]


def test_describe_ebm_builds_the_prompt_options_once(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    # the two graphs and the summary share one set of prompt options, so BAML builds the defaults once
    defaults: list[dict[str, Any]] = []
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(
        baml_sdk,
        "default_graph_options_sync",
        fake_sdk("default_graph_options_sync", appending(defaults, baml_sdk.GraphPromptOptions(**DEFAULTS))),
    )
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert len(defaults) == 1


def test_describe_graph_builds_the_prompt_options_once(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    defaults: list[dict[str, Any]] = []
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(
        baml_sdk,
        "default_graph_options_sync",
        fake_sdk("default_graph_options_sync", appending(defaults, baml_sdk.GraphPromptOptions(**DEFAULTS))),
    )
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    assert len(defaults) == 1


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
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert workers == [8]


def _fake_summary_and_graphs(monkeypatch: pytest.MonkeyPatch, counted: baml_sdk.Counted) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))


def test_describe_ebm_shows_one_step_for_each_graph_and_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
    error_stream: io.StringIO,
) -> None:
    _fake_summary_and_graphs(monkeypatch, counted)
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm, progress_bar=True))
    lines = error_stream.getvalue().splitlines()
    assert [f"({step} of 3)" in lines[step] for step in range(4)] == [True] * 4
    assert all(line.startswith("describe_ebm ") for line in lines)


def test_describe_ebm_shows_no_progress_bar_by_default(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
    error_stream: io.StringIO,
) -> None:
    _fake_summary_and_graphs(monkeypatch, counted)
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert not error_stream.getvalue()


def test_describe_ebm_counts_the_summary_after_the_graphs(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
    error_stream: io.StringIO,
) -> None:
    def summary(**_kw: object) -> baml_sdk.Counted:
        # the bar shows both graphs and not the summary while the summary runs
        assert "(2 of 3)" in error_stream.getvalue()
        assert "(3 of 3)" not in error_stream.getvalue()
        return counted

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", summary))
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm, progress_bar=True))
    assert "(3 of 3)" in error_stream.getvalue()


def test_describe_graphs_returns_the_descriptions_in_the_order_of_the_features(counted: baml_sdk.Counted) -> None:
    finished = []

    def describe(feature_index: int, _graph: RenderedGraph) -> descriptors._Described:
        # the first graph finishes last
        time.sleep(0.1 if feature_index == 0 else 0)
        return descriptors._Described(
            baml_sdk.Counted(text=f"d{feature_index}", usage=counted.usage),
            truncated_attempts=feature_index,
        )

    described = descriptors._describe_graphs(
        describe,
        [0, 1],
        [rendered("a"), rendered("b")],
        2,
        lambda: finished.append(len(finished)),
    )
    assert [item.counted.text for item in described] == ["d0", "d1"]
    assert finished == [0, 1]


def test_describe_graphs_cancels_the_graphs_that_have_not_started_after_a_failure() -> None:
    started = []

    def describe(feature_index: int, _graph: RenderedGraph) -> descriptors._Described:
        started.append(feature_index)
        if feature_index == 0:
            msg = "first graph failed"
            raise RuntimeError(msg)
        # the second graph runs while the caller cancels the third, which has not started
        time.sleep(0.2)
        return descriptors._Described(baml_sdk.Counted(text="d", usage=_no_usage()), truncated_attempts=0)

    with pytest.raises(RuntimeError, match="first graph failed"):
        descriptors._describe_graphs(describe, [0, 1, 2], [rendered(text) for text in "abc"], 1, lambda: None)
    assert 2 not in started


def _no_usage() -> baml_sdk.TokenUsage:
    return baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[])


def test_describe_ebm_passes_the_options_to_every_graph_and_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    graph_options: list[GraphOptions] = []
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}

    def graph_text(_ebm: object, _idx: int, options: GraphOptions, _model: object) -> str:
        graph_options.append(options)
        return "g"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    options = GraphOptions(max_tokens=9)
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm, graph=options, prompt=PromptOptions(dataset_description="D")))
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
        returning(FactTexts("x", "x", "x", "x")),
    )
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[]),
    )
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", returning(text)))
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G"))
    assert seen == [local.LocalOptions()]


# The SDK classes that the faked functions take or return. The fakes build real instances, so these checks compare the
# classes of the runtime package with the stub that `baml generate` wrote next to it.
SDK_CLASSES = [
    "Counted",
    "TokenUsage",
    "TokenSample",
    "LocalExplanationText",
    "LocalFactTexts",
    "GraphPromptOptions",
    "DifficultyProposal",
    "FeatureDifficulty",
]
FAKED_FUNCTIONS = [
    "default_graph_options_sync",
    "describe_graph_cot_sync",
    "summarize_ebm_sync",
    "explain_local_sync",
    "propose_difficulty_sync",
]


@pytest.mark.parametrize("name", SDK_CLASSES)
def test_an_sdk_class_has_the_fields_of_the_stub(name: str) -> None:
    # the generated module can define a class before a class that its fields name, so resolve each forward reference
    sdk_class = getattr(baml_sdk, name)
    hints = typing.get_type_hints(sdk_class)
    runtime = {field: hints[field] for field in sdk_class.model_fields}
    stub = sdk_fields(name)
    assert runtime.keys() == stub.keys()
    assert all(same_type(runtime[field], stub[field]) for field in stub)


@pytest.mark.parametrize("name", FAKED_FUNCTIONS)
def test_a_faked_sdk_function_returns_an_sdk_class(name: str) -> None:
    returns = sdk_signature(name).return_annotation
    assert isinstance(returns, type)
    assert returns.__name__ in SDK_CLASSES


def test_the_stub_signature_has_the_parameters_that_the_descriptors_pass() -> None:
    parameters = sdk_signature("describe_graph_cot_sync").parameters
    assert list(parameters)[:3] == ["graph", "num_sentences", "opts"]
    assert parameters["graph"].annotation == str | None
    assert parameters["png_base64"].annotation == str | None
    assert parameters["opts"].annotation is baml_sdk.GraphPromptOptions
    assert sdk_signature("describe_graph_cot_sync").return_annotation is baml_sdk.Counted


def test_a_fake_with_a_parameter_that_the_sdk_function_lacks_is_rejected(counted: baml_sdk.Counted) -> None:
    def cot(*, graph_text: str, **_kw: object) -> baml_sdk.Counted:
        assert graph_text
        return counted

    with pytest.raises(AssertionError, match="graph_text"):
        fake_sdk("describe_graph_cot_sync", cot)


def test_a_call_that_does_not_bind_to_the_stub_is_rejected(counted: baml_sdk.Counted) -> None:
    fake = fake_sdk("describe_graph_cot_sync", returning(counted))
    with pytest.raises(TypeError):
        fake(graph_text="g", num_sentences=1, opts=baml_sdk.default_graph_options_sync())


def test_an_argument_of_the_wrong_type_is_rejected(counted: baml_sdk.Counted) -> None:
    fake = fake_sdk("describe_graph_cot_sync", returning(counted))
    with pytest.raises(AssertionError, match="num_sentences"):
        fake(graph="g", num_sentences="seven", opts=baml_sdk.default_graph_options_sync())


def test_a_result_of_the_wrong_type_is_rejected() -> None:
    fake = fake_sdk("describe_graph_cot_sync", returning("answer"))
    with pytest.raises(AssertionError, match="returned 'answer'"):
        fake(graph="g", num_sentences=1, opts=baml_sdk.default_graph_options_sync())


PROVIDERS = ["openai", "anthropic"]


def _invalid_argument() -> BamlError:
    value = baml_sdk.baml.errors.InvalidArgument(message="unknown provider")
    return BamlError(value, class_name="baml.errors.InvalidArgument")


@pytest.fixture
def checked_models(monkeypatch: pytest.MonkeyPatch) -> list[baml_sdk.ModelSpec]:
    """Fake ``check_model`` and ``supported_providers`` of the SDK, and record the specs that the check gets.

    Returns
    -------
    list[baml_sdk.ModelSpec]
        The list that the fake fills with the spec of each check.

    """
    specs: list[baml_sdk.ModelSpec] = []

    def check_model(*, spec: baml_sdk.ModelSpec) -> bool:
        specs.append(spec)
        return True

    monkeypatch.setattr(baml_sdk, "check_model_sync", fake_sdk("check_model_sync", check_model))
    monkeypatch.setattr(
        baml_sdk,
        "supported_providers_sync",
        fake_sdk("supported_providers_sync", returning(PROVIDERS)),
    )
    return specs


def test_a_call_with_no_model_sends_no_model_spec(
    checked_models: list[baml_sdk.ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert [call["model"] for call in graph_calls] == [None, None, None]
    assert summary_calls["model"] is None
    assert checked_models == []


def test_model_spec_copies_every_field_of_the_model_options(checked_models: list[baml_sdk.ModelSpec]) -> None:
    options = ModelOptions(
        "openai-generic",
        "m",
        api_key_env="GATEWAY_KEY",
        base_url="http://gateway.invalid/v1",
        temperature=0.2,
        max_output_tokens=99,
        headers={"x-team": "ebm"},
    )
    spec = descriptors._model_spec(options)
    assert spec == baml_sdk.ModelSpec(
        provider="openai-generic",
        model="m",
        api_key_env="GATEWAY_KEY",
        base_url="http://gateway.invalid/v1",
        temperature=0.2,
        max_output_tokens=99,
        headers={"x-team": "ebm"},
    )
    assert checked_models == [spec]


def test_model_spec_leaves_the_unset_fields_to_the_defaults_of_the_call(
    checked_models: list[baml_sdk.ModelSpec],
) -> None:
    spec = descriptors._model_spec(ModelOptions("openai", "m"))
    assert spec == baml_sdk.ModelSpec(provider="openai", model="m")
    assert spec is not None
    assert spec.headers is None
    assert checked_models == [spec]


def test_a_string_model_gives_the_spec_of_the_equal_model_options(checked_models: list[baml_sdk.ModelSpec]) -> None:
    assert descriptors._model_spec("anthropic/claude-haiku-4-5") == descriptors._model_spec(
        ModelOptions("anthropic", "claude-haiku-4-5"),
    )
    assert len(checked_models) == 2


def test_describe_graph_sends_the_model_to_the_llm(
    checked_models: list[baml_sdk.ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", recording(calls, counted)),
    )
    descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0, model="ollama/llama3.1"))
    assert calls["model"] == baml_sdk.ModelSpec(provider="ollama", model="llama3.1")
    assert len(checked_models) == 1


def test_describe_ebm_sends_the_model_to_every_graph_and_the_summary(
    checked_models: list[baml_sdk.ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    descriptors.describe_ebm(
        DescribeEBMRequest(fake_ebm, model=ModelOptions("anthropic", "claude-haiku-4-5", temperature=0.1)),
    )
    spec = baml_sdk.ModelSpec(provider="anthropic", model="claude-haiku-4-5", temperature=0.1)
    assert [call["model"] for call in graph_calls] == [spec, spec]
    assert summary_calls["model"] == spec
    assert len(checked_models) == 1  # one check for the whole call, not one for each graph


def test_explain_instance_sends_the_model_to_the_llm(
    checked_models: list[baml_sdk.ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    monkeypatch.setattr(descriptors, "facts_to_texts", returning(FactTexts("x", "x", "x", "x")))
    text = baml_sdk.LocalExplanationText(
        counterfactual="",
        sufficient_reason="",
        completions="",
        usage=baml_sdk.TokenUsage(input_tokens=0, output_tokens=0, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", recording(calls, text)))
    descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G", model="openai/gpt-5"))
    assert calls["model"] == baml_sdk.ModelSpec(provider="openai", model="gpt-5")
    assert len(checked_models) == 1


def test_an_unknown_provider_raises_before_any_paid_call(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def check_model(**_kwargs: object) -> bool:
        raise _invalid_argument()

    monkeypatch.setattr(baml_sdk, "check_model_sync", fake_sdk("check_model_sync", check_model))
    monkeypatch.setattr(
        baml_sdk,
        "supported_providers_sync",
        fake_sdk("supported_providers_sync", returning(PROVIDERS)),
    )
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    paid: list[dict[str, Any]] = []
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(paid, counted)),
    )
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", appending(paid, counted)))
    with pytest.raises(InvalidOptionError, match="nope") as caught:
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm, model="nope/m"))
    assert "anthropic" in str(caught.value)
    assert isinstance(caught.value.__cause__, BamlError)
    assert paid == []


def test_an_other_error_of_the_model_check_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def check_model(**_kwargs: object) -> bool:
        reason = "Refused"
        raise _finish_error(reason)

    monkeypatch.setattr(baml_sdk, "check_model_sync", fake_sdk("check_model_sync", check_model))
    with pytest.raises(BamlError, match="FinishReasonError"):
        descriptors._model_spec("openai/gpt-5")


def test_a_malformed_model_string_raises_before_any_paid_call(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    with pytest.raises(InvalidOptionError, match="provider/model"):
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0, model="gpt-5"))


def _missing_key_error() -> BamlError:
    value = baml_sdk.ai.errors.InvalidRequest(
        provider="openrouter",
        detail="openrouter needs an API key: pass api_key, or set OPENROUTER_API_KEY",
    )
    return BamlError(value, class_name="ai.errors.InvalidRequest")


def test_describe_graph_explains_a_missing_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def cot(**_kw: object) -> baml_sdk.Counted:
        raise _missing_key_error()

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with pytest.raises(MissingAPIKeyError, match=r"export OPENROUTER_API_KEY=\.\.\.") as raised:
        descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0))
    assert isinstance(raised.value.__cause__, BamlError)


def test_describe_ebm_explains_a_missing_key_of_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    def summary(**_kw: object) -> baml_sdk.Counted:
        raise _missing_key_error()

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", summary))
    with pytest.raises(MissingAPIKeyError):
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))


def test_describe_ebm_explains_a_missing_key_of_a_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    def cot(**_kw: object) -> baml_sdk.Counted:
        raise _missing_key_error()

    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    with pytest.raises(MissingAPIKeyError):
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))


def test_explain_instance_explains_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def explain(**_kw: object) -> baml_sdk.LocalExplanationText:
        raise _missing_key_error()

    monkeypatch.setattr(descriptors, "compute_local_facts", returning(_facts()))
    monkeypatch.setattr(descriptors, "facts_to_texts", returning(FactTexts("p", "c", "s", "k")))
    monkeypatch.setattr(baml_sdk, "explain_local_sync", fake_sdk("explain_local_sync", explain))
    with pytest.raises(MissingAPIKeyError):
        descriptors.explain_instance(ExplainInstanceRequest(FAKE_EBM, {}, "G"))


def test_describe_ebm_warns_about_the_skipped_interaction_terms(
    monkeypatch: pytest.MonkeyPatch,
    counted: baml_sdk.Counted,
) -> None:
    # the descriptions skipped the interaction terms without a word, so a user could not know they were left out
    ebm = fake_model(
        feature_names_in_=["f0", "f1"],
        term_names_=["f0", "f1", "f0 & f1"],
        term_features_=[(0,), (1,), (0, 1)],
        term_importances=lambda: [1.0, 2.0, 0.5],
    )
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    with capture_logs() as logs:
        described = descriptors.describe_ebm(DescribeEBMRequest(ebm))
    # a program reads the omitted parts from the result, without a log or the text
    assert described == EBMDescription("answer", ["f0 & f1"], [])
    assert [log for log in logs if log["log_level"] == "warning"] == [
        {
            "event": "interactions_skipped",
            "terms": ["f0 & f1"],
            "reason": "the descriptions cover the main effects only",
            "log_level": "warning",
        },
    ]
    # a reader of the summary cannot see the log, so the summary prompt names the terms too
    assert summary_calls["omitted_terms"] == ["f0 & f1"]
    assert summary_calls["omitted_features"] == []
    assert [call["omitted_terms"] for call in graph_calls] == [[], []]


def test_describe_ebm_does_not_warn_without_interactions(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(counted)))
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    with capture_logs() as logs:
        descriptors.describe_ebm(DescribeEBMRequest(fake_ebm))
    assert all(log["event"] not in {"interactions_skipped", "features_skipped"} for log in logs)
    assert summary_calls["omitted_terms"] == []
    assert summary_calls["omitted_features"] == []


def test_describe_graph_of_a_multiclass_classifier_sends_one_series_per_class(
    monkeypatch: pytest.MonkeyPatch,
    multiclass_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", recording(calls, counted)),
    )
    options = GraphOptions(classes=["low", "high"], overbudget_percent=0)
    assert descriptors.describe_graph(DescribeGraphRequest(multiclass_ebm, 0, graph=options)) == "answer"
    graph = calls["graph"]
    assert "The model is a multiclass classifier" in graph
    assert [line for line in graph.splitlines() if line.startswith("Class:")] == ["Class: low", "Class: high"]


def test_describe_ebm_of_a_multiclass_classifier_describes_every_feature(
    monkeypatch: pytest.MonkeyPatch,
    multiclass_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
) -> None:
    graph_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    described = descriptors.describe_ebm(DescribeEBMRequest(multiclass_ebm, graph=GraphOptions(overbudget_percent=0)))
    assert described == EBMDescription("answer", [], [])
    assert len(graph_calls) == len(multiclass_ebm.feature_names_in_)
    for call in graph_calls:
        assert [line for line in call["graph"].splitlines() if line.startswith("Class:")] == [
            "Class: high",
            "Class: low",
            "Class: mid",
        ]


@pytest.mark.parametrize(("feature_index", "terms"), [(0, ["f0 & f1"]), (2, ["f1 & f2"])])
def test_describe_graph_warns_about_the_interaction_terms_of_its_feature(
    monkeypatch: pytest.MonkeyPatch,
    counted: baml_sdk.Counted,
    feature_index: int,
    terms: list[str],
) -> None:
    # describe_graph skipped the interaction terms of its feature without a word
    ebm = fake_model(
        feature_names_in_=["f0", "f1", "f2"],
        term_names_=["f0", "f1", "f2", "f0 & f1", "f1 & f2"],
        term_features_=[(0,), (1,), (2,), (0, 1), (1, 2)],
    )
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", recording(calls, counted)),
    )
    with capture_logs() as logs:
        descriptors.describe_graph(DescribeGraphRequest(ebm, feature_index))
    assert [log["terms"] for log in logs if log["event"] == "interactions_skipped"] == [terms]
    # the prompt names the terms, so the description says that it leaves them out
    assert calls["omitted_terms"] == terms


def test_describe_ebm_skips_a_feature_without_a_term(
    monkeypatch: pytest.MonkeyPatch,
    counted: baml_sdk.Counted,
) -> None:
    # a model fit with exclude=["f0"] made describe_ebm read the graph of "f0", which has no term
    ebm = fake_model(
        feature_names_in_=["f0", "f1"],
        term_names_=["f1"],
        term_features_=[(1,)],
        term_importances=lambda: [2.0],
    )

    def graph_text(_ebm: object, idx: int, _options: GraphOptions, _model: object) -> str:
        return f"graph{idx}"

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    graph_calls: list[dict[str, Any]] = []
    summary_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(graph_calls, counted)),
    )
    monkeypatch.setattr(
        baml_sdk,
        "summarize_ebm_sync",
        fake_sdk("summarize_ebm_sync", recording(summary_calls, counted)),
    )
    with capture_logs() as logs:
        described = descriptors.describe_ebm(DescribeEBMRequest(ebm))
    assert described == EBMDescription("answer", [], ["f0"])
    assert [call["graph"] for call in graph_calls] == ["graph1"]
    assert summary_calls["feature_importances"] == "f1: 2.00"
    assert summary_calls["graph_descriptions"] == "f1: answer"
    # the summary and the log name the feature that has no graph
    assert summary_calls["omitted_features"] == ["f0"]
    assert summary_calls["omitted_terms"] == []
    assert [log for log in logs if log["log_level"] == "warning"] == [
        {
            "event": "features_skipped",
            "features": ["f0"],
            "reason": "the features have no main-effect term",
            "log_level": "warning",
        },
    ]


def _sdk_calls(monkeypatch: pytest.MonkeyPatch, counted: baml_sdk.Counted) -> dict[str, list[dict[str, Any]]]:
    """Replace the SDK functions of the three entry points by fakes that record their calls.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        The keyword arguments of each call, by the name of the SDK function.

    """
    calls: dict[str, list[dict[str, Any]]] = {
        "describe_graph_cot_sync": [],
        "summarize_ebm_sync": [],
        "explain_local_sync": [],
    }
    for name in ("describe_graph_cot_sync", "summarize_ebm_sync"):
        monkeypatch.setattr(baml_sdk, name, fake_sdk(name, appending(calls[name], counted)))
    explained = baml_sdk.LocalExplanationText(
        counterfactual="c",
        sufficient_reason="s",
        completions="k",
        usage=counted.usage,
    )
    monkeypatch.setattr(
        baml_sdk,
        "explain_local_sync",
        fake_sdk("explain_local_sync", appending(calls["explain_local_sync"], explained)),
    )
    return calls


@pytest.mark.parametrize("mode", list(GraphMode))
@pytest.mark.parametrize("kind", PARTIAL_KINDS)
def test_each_entry_point_names_or_rejects_a_model_with_an_interaction(
    monkeypatch: pytest.MonkeyPatch,
    partial_models: dict[tuple[str, str], tuple[ExplainableBoostingClassifier | ExplainableBoostingRegressor, Any]],
    counted: baml_sdk.Counted,
    kind: str,
    mode: GraphMode,
) -> None:
    # contract 4: an explanation covers every term in its scope, or it names what it leaves out
    ebm, data = partial_models[kind, "interaction"]
    (term, features) = next(
        (name, features)
        for name, features in zip(ebm.term_names_, ebm.term_features_, strict=True)
        if len(features) > 1
    )
    outside = next(index for index, _ in enumerate(ebm.feature_names_in_) if index not in features)
    calls = _sdk_calls(monkeypatch, counted)
    graph_options = GraphOptions(max_tokens=20_000, mode=mode)

    # describe_ebm: named in the summary prompt and in the log, in every mode
    with capture_logs() as logs:
        described = descriptors.describe_ebm(DescribeEBMRequest(ebm, graph=graph_options))
    assert (described.skipped_terms, described.skipped_features) == ([term], [])
    assert [call["omitted_terms"] for call in calls["summarize_ebm_sync"]] == [[term]]
    assert [log["terms"] for log in logs if log["event"] == "interactions_skipped"] == [[term]]
    assert all(call["omitted_terms"] == [] for call in calls["describe_graph_cot_sync"])

    # describe_graph: named in the prompt of a feature of the term only
    calls["describe_graph_cot_sync"].clear()
    descriptors.describe_graph(DescribeGraphRequest(ebm, features[0], graph=graph_options))
    descriptors.describe_graph(DescribeGraphRequest(ebm, outside, graph=graph_options))
    assert [call["omitted_terms"] for call in calls["describe_graph_cot_sync"]] == [[term], []]

    # explain_instance: rejected before any LLM call
    with pytest.raises(UnsupportedModelError, match="without interactions"):
        descriptors.explain_instance(ExplainInstanceRequest(ebm, instance_of(data, 0), "G"))
    assert calls["explain_local_sync"] == []


@pytest.mark.parametrize("kind", PARTIAL_KINDS)
def test_each_entry_point_names_or_rejects_a_model_with_a_removed_term(
    monkeypatch: pytest.MonkeyPatch,
    partial_models: dict[tuple[str, str], tuple[ExplainableBoostingClassifier | ExplainableBoostingRegressor, Any]],
    counted: baml_sdk.Counted,
    kind: str,
) -> None:
    ebm, data = partial_models[kind, "removed_term"]
    removed = list(ebm.feature_names_in_).index(REMOVED_FEATURE)
    calls = _sdk_calls(monkeypatch, counted)
    graph_options = GraphOptions(max_tokens=20_000)

    # describe_ebm: named in the summary prompt and in the log
    with capture_logs() as logs:
        described = descriptors.describe_ebm(DescribeEBMRequest(ebm, graph=graph_options))
    assert (described.skipped_terms, described.skipped_features) == ([], [REMOVED_FEATURE])
    assert [call["omitted_features"] for call in calls["summarize_ebm_sync"]] == [[REMOVED_FEATURE]]
    assert [call["omitted_terms"] for call in calls["summarize_ebm_sync"]] == [[]]
    assert [log["features"] for log in logs if log["event"] == "features_skipped"] == [[REMOVED_FEATURE]]
    assert len(calls["describe_graph_cot_sync"]) == len(ebm.feature_names_in_) - 1

    # describe_graph of the feature without a graph, and explain_instance: rejected before any LLM call
    calls["describe_graph_cot_sync"].clear()
    with pytest.raises(UnsupportedModelError, match="no main-effect term"):
        descriptors.describe_graph(DescribeGraphRequest(ebm, removed, graph=graph_options))
    with pytest.raises(UnsupportedModelError, match="main-effect term for every feature"):
        descriptors.explain_instance(ExplainInstanceRequest(ebm, instance_of(data, 0), "G"))
    assert calls["describe_graph_cot_sync"] == []
    assert calls["explain_local_sync"] == []


LUNA: ModelKey = ModelKey("openrouter", "openai/gpt-6-luna")


def _learner(tmp_path: Path, *, learning: bool = False, spread: float = 0.0) -> TokenLearner:
    # one token for each word, so that the estimate of a text is its number of words
    weights = TokenWeights(words=1.0, bytes=0.0, digits=0.0)
    table = WeightsTable(default=WeightsEntry(weights, spread, 1), models={LUNA: WeightsEntry(weights, spread, 1)})
    return TokenLearner(learning=learning, path=tmp_path / "learned.json", table=table)


def _sampled(text: str = "prompt g prompt", tokens: int = 33) -> baml_sdk.Counted:
    body = orjson.dumps({"messages": [{"role": "user", "content": text}]}).decode()
    sample = baml_sdk.TokenSample(provider=LUNA.provider, model=LUNA.model, body=body, input_tokens=tokens)
    return baml_sdk.Counted(text="answer", usage=baml_sdk.TokenUsage(input_tokens=1, output_tokens=2, samples=[sample]))


class _GraphRecorder:
    """A ``to_text`` that records its arguments."""

    seen: ClassVar[dict[str, Any]] = {}

    @staticmethod
    def to_text(_graph: EBMGraph, *, on_render: Callable[[EBMGraph], None], **kwargs: object) -> str:
        _GraphRecorder.seen = kwargs
        on_render(PLAIN_GRAPH)
        return "text"


def _record_graph(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(EBMGraph, "from_ebm", staticmethod(returning(PLAIN_GRAPH)))
    monkeypatch.setattr(EBMGraph, "to_text", _GraphRecorder.to_text)
    return _GraphRecorder.seen


def test_render_graph_checks_the_budget_with_the_default_weights_when_there_is_no_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_graph(monkeypatch)
    descriptors._render_graph(FAKE_EBM, 0, GraphOptions(), None)
    assert _GraphRecorder.seen["check_budget"] is check_token_budget


def test_render_graph_checks_the_budget_with_the_estimator_of_the_model_of_the_learner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _record_graph(monkeypatch)
    learner = _learner(tmp_path)
    spec = baml_sdk.ModelSpec(provider=LUNA.provider, model=LUNA.model)
    descriptors._render_graph(FAKE_EBM, 0, GraphOptions(token_learner=learner), spec)
    assert _GraphRecorder.seen["check_budget"] is learner.estimator_for(LUNA)


def test_render_graph_checks_the_budget_with_the_default_entry_when_the_model_is_not_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _record_graph(monkeypatch)
    descriptors._render_graph(FAKE_EBM, 0, GraphOptions(token_learner=_learner(tmp_path)), None)
    assert _GraphRecorder.seen["check_budget"].weights == TokenWeights(words=1.0, bytes=0.0, digits=0.0)


def test_describe_graph_retries_with_the_estimate_of_the_learner(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    counted: baml_sdk.Counted,
    tmp_path: Path,
) -> None:
    options = []

    def graph_text(_ebm: object, _idx: int, graph: GraphOptions, _model: object) -> str:
        options.append(graph)
        return "w " * (800 if graph.max_tokens == GraphOptions().max_tokens else 100)

    replies = [_finish_error("MaxTokens"), counted]

    def cot(**_kw: object) -> baml_sdk.Counted:
        reply = replies.pop(0)
        if isinstance(reply, BamlError):
            raise reply
        return reply

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    learner = _learner(tmp_path)
    descriptors.describe_graph(
        DescribeGraphRequest(
            fake_ebm,
            0,
            graph=GraphOptions(token_learner=learner),
            model=f"{LUNA.provider}/{LUNA.model}",
        ),
    )
    # the estimate of 800 words is 800 tokens, and the retry asks for 75% of it
    assert options[-1].max_tokens == 600


def test_describe_graph_teaches_the_learner_the_samples_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(_sampled())))
    learner = _learner(tmp_path, learning=True)
    descriptors.describe_graph(
        DescribeGraphRequest(
            fake_ebm,
            0,
            graph=GraphOptions(token_learner=learner),
            model=f"{LUNA.provider}/{LUNA.model}",
        ),
    )
    assert learner.estimator_for(LUNA).samples_seen == 1


def test_describe_graph_without_a_learner_keeps_the_samples_out_of_every_estimator(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(_sampled())))
    assert descriptors.describe_graph(DescribeGraphRequest(fake_ebm, 0)) == "answer"


def test_describe_ebm_teaches_the_learner_the_samples_of_every_graph(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(descriptors, "_render_graph", returning(rendered("g")))
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(_sampled())))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(_sampled("summary"))))
    learner = _learner(tmp_path, learning=True)
    descriptors.describe_ebm(
        DescribeEBMRequest(fake_ebm, graph=GraphOptions(token_learner=learner), model=f"{LUNA.provider}/{LUNA.model}"),
    )
    # one sample for each of the two graphs. The summary holds no graph, so it teaches the learner nothing.
    assert learner.estimator_for(LUNA).samples_seen == 2


def test_describe_graph_teaches_the_learner_the_graph_text_that_the_successful_attempt_sent(
    monkeypatch: pytest.MonkeyPatch,
    fake_ebm: ExplainableBoostingClassifier,
    tmp_path: Path,
) -> None:
    texts = iter(["long graph text", "g"])

    def graph_text(*_args: object) -> str:
        return next(texts)

    monkeypatch.setattr(descriptors, "_render_graph", rendering(graph_text))
    replies = [_finish_error("MaxTokens"), _sampled("prompt g prompt")]

    def cot(**_kw: object) -> baml_sdk.Counted:
        reply = replies.pop(0)
        if isinstance(reply, BamlError):
            raise reply
        return reply

    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cot))
    learner = _learner(tmp_path, learning=True)
    descriptors.describe_graph(
        DescribeGraphRequest(
            fake_ebm,
            0,
            graph=GraphOptions(token_learner=learner),
            model=f"{LUNA.provider}/{LUNA.model}",
        ),
    )
    # the request holds the shorter graph "g" that the retry sent, and not the first graph
    assert learner.estimator_for(LUNA).samples_seen == 1


def test_sample_graph_tokens_sends_the_graph_the_prompt_and_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    sample = baml_sdk.TokenSample(provider="p", model="m", body="{}", input_tokens=5)
    monkeypatch.setattr(
        baml_sdk,
        "sample_graph_tokens_sync",
        fake_sdk("sample_graph_tokens_sync", appending(calls, [sample])),
    )
    result = descriptors.sample_graph_tokens(
        "GRAPH",
        ModelOptions("openrouter", "some/model", max_output_tokens=16),
        prompt=PromptOptions(task_description="new"),
    )
    assert result == [sample]
    assert calls[0]["graph"] == "GRAPH"
    assert calls[0]["opts"].task_description == "new"
    assert (calls[0]["model"].model, calls[0]["model"].max_output_tokens) == ("some/model", 16)


def test_sample_graph_tokens_uses_the_default_clients_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        baml_sdk,
        "sample_graph_tokens_sync",
        fake_sdk("sample_graph_tokens_sync", appending(calls, [])),
    )
    assert descriptors.sample_graph_tokens("GRAPH") == []
    assert calls[0]["model"] is None


def test_sample_graph_tokens_turns_a_missing_key_into_an_error_with_fix_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_kw: object) -> list[baml_sdk.TokenSample]:
        raise _missing_key_error()

    monkeypatch.setattr(baml_sdk, "sample_graph_tokens_sync", fake_sdk("sample_graph_tokens_sync", refuse))
    with pytest.raises(MissingAPIKeyError):
        descriptors.sample_graph_tokens("GRAPH")


def _sdk_names_used_by_descriptors() -> set[str]:
    tree = ast.parse(Path(descriptors.__file__).read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "baml_sdk"
    }


def test_the_sdk_functions_that_descriptors_calls_are_plain_functions() -> None:
    # A toolchain newer than the baml-bridge of uv.lock writes `async def` for every function. The fakes of the other
    # tests return a value at once, so only this check sees that the real call would return a coroutine.
    functions = _sdk_names_used_by_descriptors() & sdk_function_names()
    assert functions, "descriptors.py calls no function of baml_sdk"
    for name in sorted(functions):
        sdk_signature(name)


def test_an_async_sdk_function_fails_the_fake_with_the_way_to_fix_it(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = ast.parse("async def describe(x: int) -> str: ...\ndef describe_sync(x: int) -> str: ...\n")
    monkeypatch.setattr("tests.unit.ebaml.conftest._sdk_stub", lambda: stub)
    monkeypatch.setattr("tests.unit.ebaml.conftest._sdk_functions", lambda: {"describe_sync": stub.body[1]})
    monkeypatch.setattr("tests.unit.ebaml.conftest._sdk_async_functions", lambda: frozenset({"describe"}))
    with pytest.raises(AssertionError, match=r"baml_sdk\.describe is async.*_sync. twin"):
        sdk_signature("describe")


def test_chat_with_mcp_sends_the_server_command_the_model_and_the_step_limit(
    checked_models: list[baml_sdk.ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(baml_sdk, "chat_with_mcp_sync", fake_sdk("chat_with_mcp_sync", appending(calls, 3)))
    answered = descriptors.chat_with_mcp("python", ("-c", "server"), "openai/gpt-4o-mini", max_steps=4)
    assert answered == 3
    assert calls[0]["command"] == "python"
    assert calls[0]["args"] == ["-c", "server"]
    assert (calls[0]["model"].provider, calls[0]["model"].model) == ("openai", "gpt-4o-mini")
    assert calls[0]["max_steps"] == 4
    assert [spec.model for spec in checked_models] == ["gpt-4o-mini"]


def test_chat_with_mcp_uses_the_default_clients_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(baml_sdk, "chat_with_mcp_sync", fake_sdk("chat_with_mcp_sync", appending(calls, 0)))
    assert descriptors.chat_with_mcp("python", []) == 0
    assert calls[0]["model"] is None
    assert calls[0]["max_steps"] == 12


def test_chat_with_mcp_turns_a_missing_key_into_an_error_with_fix_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_kw: object) -> int:
        raise _missing_key_error()

    monkeypatch.setattr(baml_sdk, "chat_with_mcp_sync", fake_sdk("chat_with_mcp_sync", refuse))
    with pytest.raises(MissingAPIKeyError):
        descriptors.chat_with_mcp("python", [])


def test_propose_difficulties_sends_the_inputs_and_reads_the_reply(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    ebm, data = continuous_ebm
    reply = baml_sdk.DifficultyProposal(
        difficulties=[
            baml_sdk.FeatureDifficulty(feature="a", difficulty=baml_sdk.Difficulty.Hard, reason="years"),
            baml_sdk.FeatureDifficulty(feature="nope", difficulty=baml_sdk.Difficulty.Easy, reason="unknown"),
        ],
        usage=baml_sdk.TokenUsage(input_tokens=3, output_tokens=4, samples=[]),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        baml_sdk,
        "propose_difficulty_sync",
        fake_sdk("propose_difficulty_sync", recording(calls, reply)),
    )
    request = ProposeDifficultiesRequest(ebm, instance_of(data, 0), prompt=PromptOptions(dataset_description="D"))
    with capture_logs() as logs:
        proposal = descriptors.propose_difficulties(request)
    assert calls["features"].splitlines()[0].startswith('- "a" (continuous), current value')
    assert calls["opts"].dataset_description == "D"
    assert [(item.feature, item.multiplier) for item in proposal.difficulties] == [("a", 2.0)]
    assert proposal.costs.proposed_by_model
    usage = [(log["what"], log["input_tokens"], log["output_tokens"]) for log in logs if log["event"] == "usage"]
    assert usage == [("propose_difficulties", 3, 4)]


# The graph modes and the fallback for a model that reads no images.

OPENROUTER_NO_IMAGES = (
    '{"error":{"message":"No endpoints found that support image input","code":404,'
    '"metadata":{"failed_routing_step":"Filter by Image Support"}}}'
)


def _rejected_image() -> BamlError:
    # the record of the live OpenRouter answer to an image for a text-only model
    value = baml_sdk.ai.errors.InvalidRequest(
        provider="openrouter",
        detail="http 404",
        status_code=404,
        raw_body=OPENROUTER_NO_IMAGES,
    )
    return BamlError(value, class_name="ai.errors.InvalidRequest")


class _ImageRejecter:
    """A fake of ``describe_graph_cot_sync`` that rejects every request with an image, and records each request."""

    def __init__(self, counted: baml_sdk.Counted) -> None:
        self.counted = counted
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: object) -> baml_sdk.Counted:
        self.requests.append(kwargs)
        if kwargs["png_base64"] is not None:
            raise _rejected_image()
        return self.counted


def _graph_request(
    ebm: ExplainableBoostingClassifier,
    *,
    mode: GraphMode = GraphMode.VISION,
    fallback: VisionFallback = VisionFallback.RAISE,
    model: ModelOptions | None = None,
) -> DescribeGraphRequest:
    graph = GraphOptions(max_tokens=20_000, mode=mode, vision_fallback=fallback)
    return DescribeGraphRequest(ebm, 0, graph=graph, model=model)


def _record_cot(monkeypatch: pytest.MonkeyPatch, counted: baml_sdk.Counted) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        baml_sdk,
        "describe_graph_cot_sync",
        fake_sdk("describe_graph_cot_sync", appending(calls, counted)),
    )
    return calls


def _decoded_png(call: dict[str, Any]) -> bytes:
    return base64.b64decode(call["png_base64"])


def test_text_mode_sends_the_graph_text_and_no_image(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    calls = _record_cot(monkeypatch, counted)
    descriptors.describe_graph(_graph_request(continuous_ebm[0], mode=GraphMode.TEXT))
    ((call,),) = [calls]
    assert call["png_base64"] is None
    assert "Feature Name: a" in call["graph"]


def test_vision_mode_sends_the_image_of_the_extracted_graph_and_no_text(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    ebm = continuous_ebm[0]
    calls = _record_cot(monkeypatch, counted)
    descriptors.describe_graph(_graph_request(ebm))
    ((call,),) = [calls]
    assert call["graph"] is None
    options = GraphImageOptions(score_axis=model_kind(ebm).score_axis)
    assert _decoded_png(call) == graph_png(EBMGraph.from_ebm(ebm, 0), options)


def test_mixed_mode_sends_the_text_and_the_image_of_the_graph_that_the_text_renders(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    ebm = continuous_ebm[0]
    calls = _record_cot(monkeypatch, counted)
    # a small budget simplifies the graph, so that the image of the text differs from the image of the extracted graph
    graph = GraphOptions(max_tokens=1500, overbudget_percent=0, mode=GraphMode.MIXED)
    descriptors.describe_graph(DescribeGraphRequest(ebm, 0, graph=graph))
    ((call,),) = [calls]
    shown: list[EBMGraph] = []
    text = EBMGraph.from_ebm(ebm, 0).to_text(
        max_tokens=1500,
        overbudget_percent=0,
        check_budget=check_token_budget,
        on_render=shown.append,
    )
    options = GraphImageOptions(score_axis=model_kind(ebm).score_axis)
    assert call["graph"] == text
    assert _decoded_png(call) == graph_png(shown[0], options)
    assert _decoded_png(call) != graph_png(EBMGraph.from_ebm(ebm, 0), options)


def test_the_image_has_no_bounds_when_the_text_has_none(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    ebm = continuous_ebm[0]
    calls = _record_cot(monkeypatch, counted)
    graph = GraphOptions(max_tokens=20_000, mode=GraphMode.VISION, confidence_bounds=False)
    descriptors.describe_graph(DescribeGraphRequest(ebm, 0, graph=graph))
    options = GraphImageOptions(score_axis=model_kind(ebm).score_axis, confidence_level=None)
    assert _decoded_png(calls[0]) == graph_png(EBMGraph.from_ebm(ebm, 0), options)


def test_a_rejected_image_raises_the_vision_error_after_one_request(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    rejecter = _ImageRejecter(counted)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", rejecter))
    model = ModelOptions("openrouter", "liquid/lfm-3-1.7b:free")
    with pytest.raises(VisionUnsupportedError, match=re.escape("openrouter/liquid/lfm-3-1.7b:free reads no images")):
        descriptors.describe_graph(_graph_request(continuous_ebm[0], model=model))
    assert len(rejecter.requests) == 1


def test_a_rejected_image_with_the_text_fallback_describes_the_text_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    rejecter = _ImageRejecter(counted)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", rejecter))
    with pytest.warns(VisionFallbackWarning, match="sends the graphs as text") as warned:
        text = descriptors.describe_graph(_graph_request(continuous_ebm[0], fallback=VisionFallback.TEXT))
    assert text == "answer"
    assert len(warned) == 1
    assert [request["png_base64"] is None for request in rejecter.requests] == [False, True]
    assert "Feature Name: a" in rejecter.requests[1]["graph"]


@pytest.mark.parametrize("mode", [GraphMode.VISION, GraphMode.MIXED])
def test_a_model_that_reads_no_images_by_its_options_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
    mode: GraphMode,
) -> None:
    calls = _record_cot(monkeypatch, counted)
    model = ModelOptions("openai", "text-only", vision=False)
    with pytest.raises(VisionUnsupportedError, match=r"ModelOptions\(vision=False\)"):
        descriptors.describe_graph(_graph_request(continuous_ebm[0], mode=mode, model=model))
    assert calls == []


def test_a_model_that_reads_no_images_by_its_options_gets_text_with_the_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    calls = _record_cot(monkeypatch, counted)
    model = ModelOptions("openai", "text-only", vision=False)
    with pytest.warns(VisionFallbackWarning):
        descriptors.describe_graph(_graph_request(continuous_ebm[0], fallback=VisionFallback.TEXT, model=model))
    assert [call["png_base64"] for call in calls] == [None]


@pytest.mark.parametrize("model", [ModelOptions("openai", "m", vision=True), ModelOptions("openai", "m")])
def test_a_model_that_may_read_images_gets_the_image(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
    model: ModelOptions,
) -> None:
    calls = _record_cot(monkeypatch, counted)
    descriptors.describe_graph(_graph_request(continuous_ebm[0], model=model))
    assert calls[0]["png_base64"] is not None


def test_one_rejected_image_turns_every_later_graph_of_describe_ebm_to_text(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    ebm = continuous_ebm[0]
    rejecter = _ImageRejecter(counted)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", rejecter))
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", returning(counted)))
    graph = GraphOptions(max_tokens=20_000, mode=GraphMode.VISION, vision_fallback=VisionFallback.TEXT)
    with pytest.warns(VisionFallbackWarning) as warned:
        descriptors.describe_ebm(DescribeEBMRequest(ebm, graph=graph, max_concurrency=1))
    # one rejected request, then one text request for each of the three graphs
    assert [request["png_base64"] is None for request in rejecter.requests] == [False, True, True, True]
    assert len(warned) == 1


def test_one_rejected_image_stops_describe_ebm_with_the_raise_fallback(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    rejecter = _ImageRejecter(counted)
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", rejecter))
    summaries = _record_summaries(monkeypatch, counted)
    graph = GraphOptions(max_tokens=20_000, mode=GraphMode.MIXED)
    with pytest.raises(VisionUnsupportedError):
        descriptors.describe_ebm(DescribeEBMRequest(continuous_ebm[0], graph=graph, max_concurrency=1))
    assert len(rejecter.requests) == 1
    assert summaries == []


def _record_summaries(monkeypatch: pytest.MonkeyPatch, counted: baml_sdk.Counted) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(baml_sdk, "summarize_ebm_sync", fake_sdk("summarize_ebm_sync", appending(calls, counted)))
    return calls


def test_a_vision_error_of_a_request_without_an_image_is_raised_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    requests: list[dict[str, Any]] = []

    def reject_all(**kwargs: object) -> baml_sdk.Counted:
        requests.append(kwargs)
        raise _rejected_image()

    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", reject_all))
    request = _graph_request(continuous_ebm[0], mode=GraphMode.TEXT, fallback=VisionFallback.TEXT)
    with pytest.raises(VisionUnsupportedError):
        descriptors.describe_graph(request)
    assert len(requests) == 1


def test_a_cut_off_reply_in_vision_mode_is_not_retried_with_the_same_image(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
) -> None:
    requests: list[dict[str, Any]] = []

    def cut_off(**kwargs: object) -> baml_sdk.Counted:
        requests.append(kwargs)
        raise _finish_error(MAX_TOKENS)

    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cut_off))
    with pytest.raises(BamlError):
        descriptors.describe_graph(_graph_request(continuous_ebm[0]))
    assert len(requests) == 1


def test_a_cut_off_reply_in_mixed_mode_is_retried_with_a_shorter_text_and_its_image(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    requests: list[dict[str, Any]] = []

    def cut_off_once(**kwargs: object) -> baml_sdk.Counted:
        requests.append(kwargs)
        if len(requests) == 1:
            raise _finish_error(MAX_TOKENS)
        return counted

    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", cut_off_once))
    descriptors.describe_graph(_graph_request(continuous_ebm[0], mode=GraphMode.MIXED))
    first, second = requests
    assert len(second["graph"]) < len(first["graph"])
    assert second["png_base64"] != first["png_base64"]


def test_a_request_with_an_image_teaches_the_learner_nothing(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(baml_sdk, "describe_graph_cot_sync", fake_sdk("describe_graph_cot_sync", returning(_sampled())))
    learner = _learner(tmp_path, learning=True)
    graph = GraphOptions(max_tokens=20_000, mode=GraphMode.MIXED, token_learner=learner)
    descriptors.describe_graph(DescribeGraphRequest(continuous_ebm[0], 0, graph=graph))
    assert not (tmp_path / "learned.json").exists()


def test_a_call_in_vision_mode_without_the_extra_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    continuous_ebm: tuple[ExplainableBoostingClassifier, PandasFrame],
    counted: baml_sdk.Counted,
) -> None:
    calls = _record_cot(monkeypatch, counted)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "ebaml.graphs.graph_image")
    with pytest.raises(MissingExtraError, match=r"ebaml\[vision\]"):
        descriptors.describe_graph(_graph_request(continuous_ebm[0]))
    assert calls == []
