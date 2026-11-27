"""The description of an EBM, and the partial results of ``describe_ebm`` when the request of some features fails.

``describe_ebm`` describes the features at the same time. The error of one feature either concerns that feature only, or
affects every request of the call (``LLMRequestError.affects_every_request``). ``feature_error`` decides, and
``settle_outcomes`` splits the outcomes of the features into descriptions and failures.
"""

import functools
from collections.abc import Sequence

import attrs
import structlog
from baml_bridge.errors import BamlError

from ebaml.core.exceptions import LLMReplyError, LLMRequestError
from ebaml.llm.sdk_steps import Described, FeatureOutcome

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

_QUOTE_CHARACTERS = 300


@attrs.frozen
class FeatureFailure:
    """A feature whose graph description failed, and the error of its request.

    Attributes
    ----------
    feature: str
        The name of the feature.
    error: LLMRequestError
        The error. It concerns this feature only: an error that concerns every request stops ``describe_ebm``.

    """

    feature: str
    error: LLMRequestError


@functools.singledispatch
def feature_error(error: Exception) -> LLMRequestError:
    """Turn the error of one graph pipeline into the error of its feature, or raise it when it stops the whole call.

    An ``LLMRequestError`` concerns one feature when its class says that it does not affect every request. A
    ``BamlError`` that ``provider_errors`` does not explain is a reply that stays cut off or does not parse, so it
    concerns one feature, and the function returns an ``LLMReplyError``. The function raises any other error again,
    and an ``LLMRequestError`` that affects every request, because such an error stops the whole call.

    Parameters
    ----------
    error: Exception
        The error of the pipeline. The registered functions of the ``LLMRequestError`` and the ``BamlError`` classes
        return the error of the feature.

    """
    raise error


@feature_error.register
def _feature_error_of_request(error: LLMRequestError) -> LLMRequestError:
    if error.affects_every_request:
        raise error
    return error


@feature_error.register
def _feature_error_of_reply(error: BamlError) -> LLMRequestError:
    reply_error = LLMReplyError(
        "The reply of the LLM for one graph was cut off by max_tokens after the retries, or did not parse.\n"
        f"The BAML runtime said: {str(error)[:_QUOTE_CHARACTERS]}\n"
        "To fix it:\n"
        '  1. Raise max_output_tokens, for example model=ModelOptions("provider", "model", max_output_tokens=4000).\n'
        "  2. Lower max_tokens in GraphOptions, so that the graph text and the reply are shorter.\n"
        "  3. Run the call again, or choose another model, if the reply did not parse.",
    )
    reply_error.__cause__ = error
    return reply_error


@attrs.frozen
class SettledOutcomes:
    """The descriptions of the features that succeeded, by name, and the features that failed, in order."""

    described: dict[str, Described]
    failed_features: list[FeatureFailure]

    def graph_descriptions(self) -> str:
        """Join the descriptions for the summary prompt.

        Returns
        -------
        str
            One paragraph for each feature that succeeded: its name and its description.

        """
        return "\n\n".join(f"{name}: {item.counted.text}" for name, item in self.described.items())


def settle_outcomes(feature_names: Sequence[str], outcomes: Sequence[FeatureOutcome]) -> SettledOutcomes:
    """Split the outcomes of the graphs into descriptions and failures, and log the failures.

    If every feature failed, the summary has nothing to cover, so the function raises the error of the first feature.

    Parameters
    ----------
    feature_names: Sequence[str]
        The name of each feature, in the order of the outcomes.
    outcomes: Sequence[FeatureOutcome]
        The outcome of the graph of each feature.

    Returns
    -------
    SettledOutcomes
        The descriptions and the failures.

    """
    pairs = list(zip(feature_names, outcomes, strict=True))
    described = {name: outcome.described for name, outcome in pairs if outcome.described is not None}
    failed = [FeatureFailure(name, outcome.error) for name, outcome in pairs if outcome.error is not None]
    if failed and not described:
        raise failed[0].error
    if failed:
        logger.warning("features_failed", features=[failure.feature for failure in failed])
    return SettledOutcomes(described, failed)


@attrs.frozen
class EBMDescription:
    """The description of an EBM, and the parts of the model that it leaves out.

    A reader of ``text`` learns from the text itself what the description leaves out. A program reads the same
    information from the three lists, with no need to parse a log or the text.

    Attributes
    ----------
    text: str
        The description. Pass it as ``global_explanation`` to ``explain_instance``.
    skipped_terms: list[str]
        The names of the interaction terms that the description leaves out. The list is empty for a model without
        interactions.
    skipped_features: list[str]
        The names of the features that have no main-effect term, because ``exclude`` or ``remove_terms`` left them
        out. The description has no graph for them.
    failed_features: list[FeatureFailure]
        The features whose graph description failed, each with its error, in the order of the features. The summary
        names them and does not cover them. The list is empty when every graph description succeeded.

    """

    text: str
    skipped_terms: list[str]
    skipped_features: list[str]
    failed_features: list[FeatureFailure] = attrs.field(factory=list, kw_only=True)
