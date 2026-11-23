"""The prompts and the facts agree on the name of each key that crosses from Python into a prompt.

A prompt reads a fact by its key. If Python renames a key and the prompt keeps the old one, the prompt reads a key that
no fact has, and the LLM gets no instruction for the new one. ``.claude/rules/renames.md`` asks for one name on both
sides; this test reads both sources of truth.
"""

import re
from pathlib import Path

import orjson

from ebaml.core.custom_types import JSONValue
from ebaml.local.audience import DataScientist, Layperson
from ebaml.local.costs import UserCost
from ebaml.local.fact_text import facts_to_texts
from ebaml.local.fact_types import (
    Change,
    CompletionStep,
    Contribution,
    Counterfactual,
    LocalFacts,
    Prediction,
    SufficientReason,
)
from ebaml.local.minimality import MinimalityStatus
from ebaml.local.recourse import Levels, RecourseOptions

# the options that turn on every optional key: equal costs give the bound of the number of changes, and other costs
# give the costs, the cap and the limits of the person
RECOURSE_OPTIONS: tuple[RecourseOptions, ...] = (
    RecourseOptions(plans=2),
    RecourseOptions(
        immutable=["f1"],
        allowed={"f2": Levels(["a"])},
        costs=UserCost(per_feature={"f0": 2.0}),
        max_changes=2,
        plans=2,
    ),
)
PROMPTS = Path(__file__).parents[2] / "baml_src" / "local_explanations.baml"
# a quoted snake_case word of a prompt names a key of the facts
QUOTED_KEY: re.Pattern[str] = re.compile(r'"([a-z]+(?:_[a-z0-9]+)+)"')


def _facts(confidence_level: float, recourse: RecourseOptions) -> LocalFacts:
    """Build facts with two plans, a sufficient reason and completions, under the given recourse options.

    Returns
    -------
    LocalFacts
        The facts.

    """
    plan = Counterfactual(
        [Change("f0", "1", "[2, 3)", -0.5, 2)],
        new_decision=0,
        new_output=0.7,
        minimality=MinimalityStatus.UNPROVED,
        cost=2.0,
        cost_lower_bound=1.0,
    )
    return LocalFacts(
        prediction=Prediction(1, 0.8, 0.7, 0.9, 1.4, 0.2, 1.4),
        contributions=[Contribution("f0", "1", 0.6, 0.1)],
        counterfactuals=[plan, plan],
        sufficient_reason=SufficientReason(["f0"], 0.5, MinimalityStatus.UNPROVED, 1),
        completions=[CompletionStep([], 0.5)],
        confidence_level=confidence_level,
        min_support=0.01,
        n_samples=100,
        recourse=recourse,
    )


def _keys(value: JSONValue) -> set[str]:
    """Collect every key of a JSON value, at any depth.

    Returns
    -------
    set[str]
        The keys.

    """
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def test_each_key_that_a_prompt_reads_is_a_key_of_the_facts() -> None:
    emitted: set[str] = set()
    for audience in (Layperson(), DataScientist()):
        for confidence_level in (0.0, 0.95):
            for recourse in RECOURSE_OPTIONS:
                texts = facts_to_texts(_facts(confidence_level, recourse), audience)
                for text in (texts.prediction, texts.counterfactual, texts.sufficient_reason, texts.completions):
                    emitted |= _keys(orjson.loads(text))
    read = set(QUOTED_KEY.findall(PROMPTS.read_text()))
    assert read - emitted == set()


def test_each_claim_key_has_the_name_of_the_field_that_it_renders() -> None:
    texts = facts_to_texts(_facts(0.95, RECOURSE_OPTIONS[0]), DataScientist())
    for text in (texts.counterfactual, texts.sufficient_reason):
        claims = {key for key in orjson.loads(text) if key.startswith("is_")}
        assert claims == {"is_optimal"}
        assert all(isinstance(getattr(MinimalityStatus, claim), property) for claim in claims)
