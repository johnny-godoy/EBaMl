"""The graph modes cross the real BAML runtime: what each mode sends, and what a model that reads no images does.

Each test answers one question about the requests that reach the local server, so each has one reason to fail.
"""

import base64
from collections.abc import Mapping
from typing import cast

import attrs
import pytest

from ebaml import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    EBMGraph,
    GraphImageOptions,
    GraphOptions,
    ModelOptions,
    VisionFallbackWarning,
    VisionUnsupportedError,
    describe_ebm,
    describe_graph,
    graph_png,
    model_kind,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from tests.conftest import FakeLLM, RecordedRequest, Reply
from tests.unit.ebaml.conftest import PandasFrame

type Fitted = tuple[PandasFrame, ExplainableBoostingClassifier]

# the answer of OpenRouter to an image for a model that reads no images, as a live request showed it
NO_IMAGES: Reply = Reply(status=404, error_message="No endpoints found that support image input")
PNG_DATA_URL = "data:image/png;base64,"


def _image_urls(request: RecordedRequest) -> list[str]:
    """List the addresses of the image parts of the messages of the user, in the wire format of ``openai-generic``.

    Returns
    -------
    list[str]
        The ``image_url.url`` of each image part, in order.

    """
    urls: list[str] = []
    for message in request.messages:
        content = message["content"]
        if message["role"] != "user" or not isinstance(content, list):
            continue
        urls.extend(
            part["image_url"]["url"]
            for part in cast("list[Mapping[str, Mapping[str, str]]]", content)
            if part.get("type") == "image_url"
        )
    return urls


def _has_image(request: RecordedRequest) -> bool:
    return bool(_image_urls(request))


def _reject_images(request: RecordedRequest) -> Reply:
    return NO_IMAGES if _has_image(request) else Reply()


def _graph(
    ebm: ExplainableBoostingClassifier,
    model: ModelOptions,
    graph: GraphOptions | None = None,
) -> DescribeGraphRequest:
    return DescribeGraphRequest(ebm, 0, num_sentences=3, graph=graph or GraphOptions(), model=model)


def test_a_request_in_vision_mode_carries_the_png_of_the_graph_as_an_image_part(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision")))
    expected = graph_png(EBMGraph.from_ebm(ebm, 0), GraphImageOptions(score_axis=model_kind(ebm).score_axis))
    (url,) = _image_urls(fake_llm.requests[0])
    assert url == PNG_DATA_URL + base64.b64encode(expected).decode()


def test_a_request_in_vision_mode_carries_no_graph_text(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision")))
    assert all("Feature Name:" not in request.prompt for request in fake_llm.requests)


def test_each_turn_of_the_chain_in_mixed_mode_carries_the_image_and_the_text(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_graph(_graph(ebm, llm_model, GraphOptions(mode="mixed")))
    assert [(len(_image_urls(request)), "Feature Name:" in request.prompt) for request in fake_llm.requests] == [
        (1, True),
    ] * 3


def test_a_request_in_text_mode_carries_no_image(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    describe_graph(_graph(ebm, llm_model))
    assert not any(_has_image(request) for request in fake_llm.requests)


def test_a_rejected_image_raises_the_vision_error_with_its_fix_steps(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = _reject_images
    with pytest.raises(VisionUnsupportedError, match=r'GraphOptions\(mode="text"\)'):
        describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision")))


def test_a_rejected_image_is_sent_once_and_not_retried(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = _reject_images
    with pytest.raises(VisionUnsupportedError):
        describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision")))
    assert len(fake_llm.requests) == 1


def test_a_rejected_image_with_the_text_fallback_emits_one_warning(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = _reject_images
    with pytest.warns(VisionFallbackWarning) as warned:
        describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision", vision_fallback="text")))
    assert len(warned) == 1


def test_a_rejected_image_with_the_text_fallback_sends_the_graph_text_without_an_image(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = _reject_images
    with pytest.warns(VisionFallbackWarning):
        describe_graph(_graph(ebm, llm_model, GraphOptions(mode="vision", vision_fallback="text")))
    retried = fake_llm.requests[1:]
    assert [(_has_image(request), "Feature Name:" in request.prompt) for request in retried] == [(False, True)] * 3


def test_a_model_that_reads_no_images_by_its_options_gets_no_request_with_the_raise_fallback(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    with pytest.raises(VisionUnsupportedError):
        describe_graph(_graph(ebm, attrs.evolve(llm_model, vision=False), GraphOptions(mode="mixed")))
    assert fake_llm.requests == []


def test_a_model_that_reads_no_images_by_its_options_gets_only_text_with_the_text_fallback(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    with pytest.warns(VisionFallbackWarning):
        describe_graph(
            _graph(ebm, attrs.evolve(llm_model, vision=False), GraphOptions(mode="vision", vision_fallback="text")),
        )
    assert [_has_image(request) for request in fake_llm.requests] == [False] * 3


def test_one_rejected_image_turns_the_whole_describe_ebm_call_to_text(
    binary_data_and_model: Fitted,
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    _, ebm = binary_data_and_model
    fake_llm.handler = _reject_images
    request = DescribeEBMRequest(
        ebm,
        num_sentences=3,
        max_concurrency=1,
        graph=GraphOptions(mode="vision", vision_fallback="text"),
        model=llm_model,
    )
    with pytest.warns(VisionFallbackWarning):
        describe_ebm(request)
    # one rejected request, then three turns for each graph in text mode, then the draft and the shortened summary
    graph_count = len(ebm.term_features_)
    assert [_has_image(request) for request in fake_llm.requests] == [True] + [False] * (3 * graph_count + 2)
