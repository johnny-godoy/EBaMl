"""Tests for the pipelines of the LLM calls, driven step by step with plain replies and no fake of ``baml_sdk``."""

import functools
from collections.abc import Callable, Generator
from typing import Any

import pytest
from baml_bridge.errors import BamlError

import baml_sdk
from ebaml.core.custom_types import RenderTiming
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import EBMGraphTokenLimitError, InvalidOptionError, MissingAPIKeyError
from ebaml.llm import graph_conversation, pipelines
from ebaml.llm.graph_views import RenderedGraph
from ebaml.llm.options import GraphOptions
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.llm.sdk_steps import (
    FanOut,
    LocalWork,
    ModelIdentity,
    Outcome,
    PipelineRun,
    SDKCall,
    SDKTypes,
    SDKValue,
    Step,
)
from tests.unit.ebaml.conftest import FittedClassifier, instance_of

TYPES: SDKTypes = SDKTypes(model_spec=baml_sdk.ModelSpec, local_fact_texts=baml_sdk.LocalFactTexts)
OPTIONS: baml_sdk.GraphPromptOptions = baml_sdk.GraphPromptOptions(
    expert_description="expert",
    graph_description="g",
    dataset_description="d",
    task_description="t",
)
USAGE: baml_sdk.TokenUsage = baml_sdk.TokenUsage(input_tokens=3, output_tokens=5, samples=[])

type Answer = Callable[[Step], Any]


def drive[R](pipeline: Generator[Step, Any, R], answer: Answer, steps: list[Step]) -> R:
    """Run a pipeline, and answer each step with ``answer``. An answer that is an exception is thrown into the pipeline.

    Returns
    -------
    R
        The value that the pipeline returns.

    """
    run = PipelineRun(pipeline)
    while (step := run.step) is not None:
        steps.append(step)
        reply = answer(step)
        run.advance(Outcome(error=reply) if isinstance(reply, Exception) else Outcome(reply=reply))
    return run.result()


def counted(text: str) -> baml_sdk.Counted:
    return baml_sdk.Counted(text=text, usage=USAGE)


def answer_by_name(replies: dict[str, Any]) -> Answer:
    """Answer each call with the reply of its name, and each fan-out with the result of its pipelines.

    Returns
    -------
    Answer
        The function that answers a step.

    """

    def answer(step: Step) -> Any:  # ruff: ignore[any-type] -- a reply has the type of its step
        if isinstance(step, LocalWork):
            return step.run()
        if isinstance(step, FanOut):
            return [
                step.feature_outcome(Outcome.of(functools.partial(drive, pipeline, answer, [])))
                for pipeline in step.pipelines
            ]
        reply = replies[step.name]
        return reply.pop(0) if isinstance(reply, list) else reply

    return answer


def names(steps: list[Step]) -> list[str]:
    return [step.name if isinstance(step, SDKCall | LocalWork) else "fan_out" for step in steps]


@pytest.fixture
def fitted(
    continuous_ebm: FittedClassifier,
) -> ExplainableBoostingClassifier:
    return continuous_ebm[0]


def test_describe_graph_reads_the_defaults_then_asks_for_the_description(fitted: ExplainableBoostingClassifier) -> None:
    steps: list[Step] = []
    replies = {"default_graph_options": OPTIONS, "describe_graph_cot": counted("rises")}
    text = drive(
        pipelines.describe_graph_pipeline(DescribeGraphRequest(fitted, 0), TYPES),
        answer_by_name(replies),
        steps,
    )
    assert text == "rises"
    assert names(steps) == ["default_graph_options", "describe_graph_cot"]


def test_describe_graph_sends_the_graph_text_and_the_prompt_options(fitted: ExplainableBoostingClassifier) -> None:
    steps: list[Step] = []
    replies = {"default_graph_options": OPTIONS, "describe_graph_cot": counted("rises")}
    drive(pipelines.describe_graph_pipeline(DescribeGraphRequest(fitted, 0), TYPES), answer_by_name(replies), steps)
    (call,) = [step for step in steps if isinstance(step, SDKCall) and step.name == "describe_graph_cot"]
    assert call.arguments["opts"] == OPTIONS
    assert "a" in call.arguments["graph"]
    assert call.arguments["png_base64"] is None


def test_a_cut_off_reply_is_asked_again_with_a_shorter_graph(fitted: ExplainableBoostingClassifier) -> None:
    cut_off = BamlError(
        baml_sdk.ai.errors.FinishReasonError(provider="p", finish_reason="MaxTokens", raw_output=""),
        class_name="ai.errors.FinishReasonError",
    )
    steps: list[Step] = []
    replies = {"default_graph_options": OPTIONS, "describe_graph_cot": [cut_off, counted("rises")]}
    drive(pipelines.describe_graph_pipeline(DescribeGraphRequest(fitted, 0), TYPES), answer_by_name(replies), steps)
    first, second = (
        step.arguments["graph"] for step in steps if isinstance(step, SDKCall) and step.name != "default_graph_options"
    )
    assert len(second) < len(first)


def test_a_provider_error_becomes_the_error_with_its_fix_steps(fitted: ExplainableBoostingClassifier) -> None:
    missing_key = BamlError(
        baml_sdk.ai.errors.InvalidRequest(
            provider="openrouter",
            detail="openrouter needs an API key: set OPENROUTER_API_KEY",
        ),
        class_name="ai.errors.InvalidRequest",
    )
    replies = {"default_graph_options": OPTIONS, "describe_graph_cot": missing_key}
    with pytest.raises(MissingAPIKeyError, match="export OPENROUTER_API_KEY"):
        drive(pipelines.describe_graph_pipeline(DescribeGraphRequest(fitted, 0), TYPES), answer_by_name(replies), [])


def test_a_model_is_checked_before_the_first_paid_call() -> None:
    steps: list[Step] = []
    spec = drive(pipelines.model_spec_pipeline("openai/gpt-5", TYPES), answer_by_name({"check_model": True}), steps)
    assert names(steps) == ["check_model"]
    assert (spec.provider, spec.model) == ("openai", "gpt-5")


def test_an_unknown_provider_lists_the_providers_that_baml_knows() -> None:
    unknown = BamlError(
        baml_sdk.baml.errors.InvalidArgument(message="unknown provider"),
        class_name="baml.errors.InvalidArgument",
    )
    steps: list[Step] = []
    replies = {"check_model": unknown, "supported_providers": ["openai"]}
    with pytest.raises(InvalidOptionError, match="openai"):
        drive(pipelines.model_spec_pipeline("nobody/model", TYPES), answer_by_name(replies), steps)
    assert names(steps) == ["check_model", "supported_providers"]


def test_no_model_needs_no_check() -> None:
    steps: list[Step] = []
    assert drive(pipelines.model_spec_pipeline(None, TYPES), answer_by_name({}), steps) is None
    assert not steps


def test_describe_ebm_fans_out_one_pipeline_per_feature_then_summarizes(fitted: ExplainableBoostingClassifier) -> None:
    steps: list[Step] = []
    replies = {
        "default_graph_options": OPTIONS,
        "describe_graph_cot": counted("graph"),
        "summarize_ebm": counted("sum"),
    }
    description = drive(
        pipelines.describe_ebm_pipeline(DescribeEBMRequest(fitted), TYPES),
        answer_by_name(replies),
        steps,
    )
    assert description.text == "sum"
    assert names(steps) == ["default_graph_options", "render_graphs", "fan_out", "summarize_ebm"]
    (fan_out,) = [step for step in steps if isinstance(step, FanOut)]
    assert len(fan_out.pipelines) == len(fitted.feature_names_in_)
    assert fan_out.max_concurrency == 8


def _describe_with(fitted: ExplainableBoostingClassifier, timing: RenderTiming, answer: Answer) -> list[Step]:
    steps: list[Step] = []
    drive(pipelines.describe_ebm_pipeline(DescribeEBMRequest(fitted, render_timing=timing), TYPES), answer, steps)
    return steps


def _graph_answer(prompts: list[str], graph_steps: list[Step] | None = None) -> Answer:
    """Answer every step, and record the graph of each description and the steps of the pipelines of the graphs.

    Returns
    -------
    Answer
        The function that answers a step.

    """
    replies = {"default_graph_options": OPTIONS, "describe_graph_cot": counted("graph"), "summarize_ebm": counted("s")}
    nested = [] if graph_steps is None else graph_steps

    def answer(step: Step) -> SDKValue:
        if isinstance(step, LocalWork):
            return step.run()
        if isinstance(step, FanOut):
            return [
                step.feature_outcome(Outcome.of(functools.partial(drive, pipeline, answer, nested)))
                for pipeline in step.pipelines
            ]
        if step.name == "describe_graph_cot":
            prompts.append(step.arguments["graph"])
        return replies[step.name]

    return answer


def test_both_render_timings_send_the_same_graphs(fitted: ExplainableBoostingClassifier) -> None:
    before: list[str] = []
    with_requests: list[str] = []
    _describe_with(fitted, RenderTiming.BEFORE_REQUESTS, _graph_answer(before))
    _describe_with(fitted, RenderTiming.WITH_REQUESTS, _graph_answer(with_requests))
    assert before == with_requests
    assert len(before) == len(fitted.feature_names_in_)


def test_rendering_with_the_requests_renders_each_graph_in_the_pipeline_of_its_feature(
    fitted: ExplainableBoostingClassifier,
) -> None:
    graph_steps: list[Step] = []
    steps = _describe_with(fitted, RenderTiming.WITH_REQUESTS, _graph_answer([], graph_steps))
    assert names(steps) == ["default_graph_options", "fan_out", "summarize_ebm"]
    feature_count = len(fitted.feature_names_in_)
    assert names(graph_steps) == ["render_graph", "describe_graph_cot"] * feature_count


@pytest.mark.parametrize(
    ("timing", "expected"),
    [(RenderTiming.BEFORE_REQUESTS, 0), (RenderTiming.WITH_REQUESTS, 1)],
)
def test_a_graph_that_fails_to_render_stops_the_call_before_or_after_the_requests_of_the_other_graphs(
    fitted: ExplainableBoostingClassifier,
    monkeypatch: pytest.MonkeyPatch,
    timing: RenderTiming,
    expected: int,
) -> None:
    render = graph_conversation.render_graph

    def fail_on_the_second(
        ebm: ExplainableBoostingClassifier,
        feature_index: int,
        options: GraphOptions,
        model: ModelIdentity | None,
    ) -> RenderedGraph:
        if feature_index == 1:
            msg = "too long"
            raise EBMGraphTokenLimitError(msg)
        return render(ebm, feature_index, options, model)

    monkeypatch.setattr(graph_conversation, "render_graph", fail_on_the_second)
    prompts: list[str] = []
    with pytest.raises(EBMGraphTokenLimitError):
        _describe_with(fitted, timing, _graph_answer(prompts))
    assert len(prompts) == expected


def test_describe_ebm_summarizes_the_description_of_every_feature(fitted: ExplainableBoostingClassifier) -> None:
    steps: list[Step] = []
    replies = {
        "default_graph_options": OPTIONS,
        "describe_graph_cot": counted("graph"),
        "summarize_ebm": counted("sum"),
    }
    drive(pipelines.describe_ebm_pipeline(DescribeEBMRequest(fitted), TYPES), answer_by_name(replies), steps)
    summary = steps[-1]
    assert isinstance(summary, SDKCall)
    assert summary.arguments["graph_descriptions"] == "a: graph\n\nb: graph\n\nc: graph"


def test_explain_instance_asks_for_the_three_explanations(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    reply = baml_sdk.LocalExplanationText(counterfactual="c", sufficient_reason="s", completions="k", usage=USAGE)
    steps: list[Step] = []
    replies = {"default_graph_options": OPTIONS, "explain_local": reply}
    request = ExplainInstanceRequest(ebm, instance_of(data, 0), "global")
    explanation = drive(pipelines.explain_instance_pipeline(request, TYPES), answer_by_name(replies), steps)
    assert (explanation.counterfactual, explanation.sufficient_reason, explanation.completions) == ("c", "s", "k")
    assert names(steps) == ["compute_local_facts", "default_graph_options", "explain_local"]


def test_propose_difficulties_reads_the_ratings_of_the_reply(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    rating = baml_sdk.FeatureDifficulty(feature="a", difficulty=baml_sdk.Difficulty.Hard, reason="r")
    reply = baml_sdk.DifficultyProposal(difficulties=[rating], usage=USAGE)
    steps: list[Step] = []
    replies = {"default_graph_options": OPTIONS, "propose_difficulty": reply}
    request = ProposeDifficultiesRequest(ebm, instance_of(data, 0))
    proposal = drive(pipelines.propose_difficulties_pipeline(request, TYPES), answer_by_name(replies), steps)
    assert [difficulty.feature for difficulty in proposal.difficulties] == ["a"]
    assert names(steps) == ["default_graph_options", "propose_difficulty"]
