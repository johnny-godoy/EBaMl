"""The tool that explains one prediction with the exact local facts. It calls no LLM."""

from collections.abc import Mapping

import attrs

from ebaml.core.custom_types import FeatureValue
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.local.audience import resolve_audience
from ebaml.local.fact_text import FactTexts, facts_to_texts
from ebaml.local.fact_types import LocalFacts, LocalOptions
from ebaml.local.facts import compute_local_facts
from ebaml.mcp.model_store import ModelStore


@attrs.frozen
class PredictionExplanation:
    """The facts of one prediction and their texts.

    Attributes
    ----------
    facts: LocalFacts
        The prediction, the contributions and the three local explanations.
    texts: FactTexts
        The facts as the JSON texts that the prompts read, for the audience of the call.

    """

    facts: LocalFacts
    texts: FactTexts


def explain_prediction(
    store: ModelStore,
    *,
    instance: Mapping[str, FeatureValue] | None = None,
    row: int | None = None,
    confidence_level: float = 0.95,
    audience: str = "layperson",
) -> PredictionExplanation:
    """Compute the local facts of one prediction.

    Parameters
    ----------
    store: ModelStore
        The session.
    instance: Mapping[str, FeatureValue], optional
        The value of every feature, by feature name.
    row: int, optional
        The position of a row of the data of ``--data``. Give exactly one of ``instance`` and ``row``.
    confidence_level: float, default=0.95
        See ``LocalOptions.confidence_level``.
    audience: str, default="layperson"
        The name of a built-in audience. It decides the keys of the texts.

    Returns
    -------
    PredictionExplanation
        The facts and their texts.

    Raises
    ------
    InvalidInstanceError
        If the call gives both or none of ``instance`` and ``row``.

    """
    if (instance is None) == (row is None):
        msg = "Give exactly one of instance and row."
        raise InvalidInstanceError(msg)
    values = store.row_instance(row) if row is not None else dict(instance or {})
    facts = compute_local_facts(store.ebm, values, LocalOptions(confidence_level=confidence_level))
    return PredictionExplanation(facts=facts, texts=facts_to_texts(facts, resolve_audience(audience)))
