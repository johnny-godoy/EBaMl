"""Put the local facts of one EBM prediction in words, as JSON texts for the LLM prompts.

``ebaml.local`` computes the facts. This module only renders them, for one ``Audience`` (``ebaml.local.audience``): a
layperson gets words in place of the model numbers, so the LLM has no number to repeat, and a data scientist gets the
numbers. The prompts that read the texts are in ``baml_src/local_explanations.baml``.
"""

from collections.abc import Mapping

import attrs
import orjson

from ebaml.core.custom_types import JSONValue
from ebaml.local.audience import Audience, Layperson
from ebaml.local.facts import LocalFacts


def _dump(data: Mapping[str, JSONValue]) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()


@attrs.frozen
class FactTexts:
    """The JSON text of each explanation, as the prompts read them."""

    prediction: str
    counterfactual: str
    sufficient_reason: str
    completions: str


_DEFAULT_AUDIENCE = Layperson()


def facts_to_texts(facts: LocalFacts, audience: Audience = _DEFAULT_AUDIENCE) -> FactTexts:
    """Render the facts as one JSON text per explanation, for the prompts.

    The audience decides which keys each fact gets. For a layperson the model numbers (scores, probabilities,
    intervals) are replaced by words, so the LLM has none to repeat. A data scientist gets the numbers.

    Parameters
    ----------
    facts: LocalFacts
        The facts of the prediction.
    audience: Audience, default=Layperson()
        Who will read the explanation. Pass an instance of any subclass of ``Audience``.

    Returns
    -------
    FactTexts
        The texts of the prediction, the counterfactual, the sufficient reason and the completions.

    """
    p = facts.prediction
    prediction: dict[str, JSONValue] = {"predicted_class": p.decision}
    prediction["contributions_largest_first"] = [
        {"feature": c.feature, "value": c.value} | audience.contribution(c, p) for c in facts.contributions
    ]
    prediction |= audience.prediction(p, facts.confidence_level)

    cf = facts.counterfactual
    counterfactual: dict[str, JSONValue] = {"flip_possible": cf is not None}
    if cf is not None:
        counterfactual["changes"] = [
            {"feature": c.feature, "from": c.from_value, "to": c.to_value} | audience.change(c) for c in cf.changes
        ]
        counterfactual["minimum_number_of_changes"] = len(cf.changes)
        counterfactual |= audience.flip(cf, facts.confidence_level)

    all_names = [c.feature for c in facts.contributions]
    sr = facts.sufficient_reason
    sufficient: dict[str, JSONValue] = {"exists": sr is not None}
    if sr is not None:
        sufficient["features"] = sr.features
        sufficient["features_not_needed"] = [f for f in all_names if f not in sr.features]
        sufficient |= audience.margin(sr)

    completions = [
        {"known_features": s.fixed, "share_with_same_prediction": audience.share(s.share)} for s in facts.completions
    ]
    return FactTexts(
        prediction=_dump(prediction),
        counterfactual=_dump(counterfactual),
        sufficient_reason=_dump(sufficient),
        completions=_dump({"steps": completions}),
    )
