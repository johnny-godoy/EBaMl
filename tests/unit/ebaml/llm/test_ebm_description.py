"""Tests for the partial results of describe_ebm: which errors concern one feature, and how the outcomes split."""

import pytest
from baml_bridge.errors import BamlError
from structlog.testing import capture_logs

import baml_sdk
from ebaml.core.exceptions import LLMReplyError, MissingAPIKeyError, RateLimitedError
from ebaml.llm.ebm_description import FeatureFailure, feature_error, settle_outcomes
from ebaml.llm.sdk_steps import Described, FeatureOutcome

USAGE: baml_sdk.TokenUsage = baml_sdk.TokenUsage(input_tokens=1, output_tokens=1, samples=[])


def _described(text: str) -> FeatureOutcome:
    return FeatureOutcome(described=Described(baml_sdk.Counted(text=text, usage=USAGE), truncated_attempts=0))


def test_an_error_of_one_feature_is_kept() -> None:
    error = RateLimitedError("slow down")
    assert feature_error(error) is error


def test_an_error_that_affects_every_request_is_raised() -> None:
    with pytest.raises(MissingAPIKeyError):
        feature_error(MissingAPIKeyError("no key"))


def test_an_unexplained_baml_error_becomes_a_reply_error_of_the_feature() -> None:
    cause = BamlError(baml_sdk.baml.errors.ParseError(message="no JSON"), class_name="baml.errors.ParseError")
    error = feature_error(cause)
    assert isinstance(error, LLMReplyError)
    assert error.__cause__ is cause
    assert "To fix it:" in str(error)


def test_a_bug_is_raised() -> None:
    with pytest.raises(KeyError):
        feature_error(KeyError("bug"))


def test_the_outcomes_split_into_descriptions_and_failures_in_order() -> None:
    error = RateLimitedError("slow down")
    settled = settle_outcomes(["a", "b", "c"], [_described("A"), FeatureOutcome(error=error), _described("C")])
    assert settled.failed_features == [FeatureFailure("b", error)]
    assert settled.graph_descriptions() == "a: A\n\nc: C"


def test_the_failed_features_are_logged_once() -> None:
    with capture_logs() as logs:
        settle_outcomes(["a", "b"], [_described("A"), FeatureOutcome(error=RateLimitedError("slow down"))])
    assert [log["features"] for log in logs if log["event"] == "features_failed"] == [["b"]]


def test_no_failure_logs_nothing() -> None:
    with capture_logs() as logs:
        settled = settle_outcomes(["a"], [_described("A")])
    assert settled.failed_features == []
    assert not logs


def test_when_every_feature_fails_the_error_of_the_first_is_raised() -> None:
    first, second = RateLimitedError("first"), RateLimitedError("second")
    with pytest.raises(RateLimitedError, match="first"):
        settle_outcomes(["a", "b"], [FeatureOutcome(error=first), FeatureOutcome(error=second)])


def test_no_feature_gives_an_empty_result() -> None:
    settled = settle_outcomes([], [])
    assert (settled.described, settled.failed_features) == ({}, [])
