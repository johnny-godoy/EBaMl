"""Put the local facts of one EBM prediction in words, as JSON texts for the LLM prompts.

``ebaml.local`` computes the facts. This module only renders them, for one ``Audience`` (``ebaml.local.audience``): a
layperson gets words in place of the model numbers, so the LLM has no number to repeat, and a data scientist gets the
numbers. The prompts that read the texts are in ``baml_src/local_explanations.baml``.
"""

from collections.abc import Mapping

import attrs
import orjson

from ebaml.core.custom_types import JSONMapping, JSONObject, JSONValue
from ebaml.local.audience import Audience, Layperson
from ebaml.local.costs import UnitCost
from ebaml.local.fact_types import Counterfactual, LocalFacts
from ebaml.local.local_kinds import OutputKind


def _dump(data: JSONMapping) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()


@attrs.frozen
class FactTexts:
    """The JSON text of each explanation, as the prompts read them."""

    prediction: str
    counterfactual: str
    sufficient_reason: str
    completions: str


# the key of the decision: a class for a classifier, an answer to a question about the value for a regressor
_DECISION_KEYS: Mapping[OutputKind, str] = {
    OutputKind.PROBABILITY: "predicted_class",
    OutputKind.CLASS_PROBABILITY: "predicted_class",
    OutputKind.VALUE: "decision",
}

_DEFAULT_AUDIENCE = Layperson()


def _changes(plan: Counterfactual, audience: Audience) -> list[JSONValue]:
    return [{"feature": c.feature, "from": c.from_value, "to": c.to_value} | audience.change(c) for c in plan.changes]


def _other_plan(
    plan: Counterfactual,
    audience: Audience,
    *,
    unit_costs: bool,
) -> JSONObject:  # nosemgrep: ebaml-return-attrs-not-container -- a fragment of a JSON object
    """Render a plan after the first: its changes, its cost and whether it is the cheapest after the earlier plans.

    Returns
    -------
    JSONObject
        The keys of the plan.

    """
    other: JSONObject = {
        "changes": _changes(plan, audience),
        "number_of_changes": len(plan.changes),
        "is_cheapest_without_the_changes_of_earlier_plans": plan.minimality.is_optimal,
    }
    if not unit_costs:
        other["total_cost"] = round(plan.cost, 4)
    return other


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
    decision_key = _DECISION_KEYS[p.output_kind]
    prediction: JSONObject = {decision_key: p.decision}
    prediction["contributions_largest_first"] = [
        {"feature": c.feature, "value": c.value} | audience.contribution(c, p) for c in facts.contributions
    ]
    prediction |= audience.prediction(p, facts.confidence_level)

    # the first plan is the cheapest one; the others follow as other_plans
    cf = facts.counterfactuals[0] if facts.counterfactuals else None
    unit_costs = facts.recourse.costs == UnitCost()
    counterfactual: JSONObject = {"flip_possible": cf is not None}
    if cf is not None:
        counterfactual["changes"] = _changes(cf, audience)
        counterfactual["number_of_changes"] = len(cf.changes)
        counterfactual["is_optimal"] = cf.minimality.is_optimal
        # with unit costs the bound of the cost is a number of changes; with other costs it says nothing about them
        if unit_costs:
            counterfactual |= audience.lower_bound(round(cf.cost_lower_bound), "changes")
        # the class after the changes and its probability share one object, so the LLM cannot read the probability
        # as one of the old class
        counterfactual["after_the_changes"] = {decision_key: cf.new_decision} | audience.class_output(
            cf.new_output,
            p.output_kind,
        )
    # the conditions of the search are facts too, with or without a result: the prompt claims a condition only when
    # its key says so
    counterfactual |= audience.flip(facts.confidence_level)
    counterfactual |= audience.support(facts.min_support)
    # the limits of the person are conditions of the search too, so the prompt names them when the keys exist
    if facts.recourse.immutable:
        counterfactual["features_kept_unchanged"] = list(facts.recourse.immutable)
    if facts.recourse.allowed:
        counterfactual["allowed_values"] = facts.recourse.describe()
    # with other costs than the default, the plan is the cheapest one and not the smallest, so the facts say so
    if not unit_costs:
        counterfactual["costs_of_the_changes"] = facts.recourse.costs.describe()
        if cf is not None:
            counterfactual["total_cost"] = round(cf.cost, 4)
    if facts.recourse.max_changes is not None:
        counterfactual["max_changes"] = facts.recourse.max_changes
    if len(facts.counterfactuals) > 1:
        counterfactual["other_plans"] = [
            _other_plan(plan, audience, unit_costs=unit_costs) for plan in facts.counterfactuals[1:]
        ]

    all_names = [c.feature for c in facts.contributions]
    sr = facts.sufficient_reason
    sufficient: JSONObject = {"exists": sr is not None}
    if sr is not None:
        sufficient["features"] = sr.features
        sufficient["is_optimal"] = sr.minimality.is_optimal
        sufficient |= audience.lower_bound(sr.lower_bound, "features")
        sufficient["features_not_needed"] = [f for f in all_names if f not in sr.features]
        sufficient |= audience.margin(sr, p.output_kind)
    sufficient |= audience.guarantee(facts.confidence_level)
    sufficient |= audience.support(facts.min_support)

    completions = [
        {"known_features": s.fixed, "share_with_same_prediction": audience.share(s.share)} for s in facts.completions
    ]
    return FactTexts(
        prediction=_dump(prediction),
        counterfactual=_dump(counterfactual),
        sufficient_reason=_dump(sufficient),
        completions=_dump({"steps": completions} | audience.sample(facts.n_samples)),
    )
