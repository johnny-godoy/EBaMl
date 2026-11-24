"""The tool that explains one prediction with the exact local facts. It calls no LLM."""

import enum
import functools
import math
import types
from collections.abc import Collection, Mapping
from typing import NotRequired, TypedDict

import attrs

from ebaml.core.custom_types import FeatureValue, coerce_enum
from ebaml.core.exceptions import InvalidInstanceError
from ebaml.edit.edit_session import CURRENT, RevisionInfo
from ebaml.local.audience import resolve_audience
from ebaml.local.costs import CostStrategy, DataCost, ScaledCost, UnitCost
from ebaml.local.fact_text import FactTexts, facts_to_texts
from ebaml.local.fact_types import LocalFacts, LocalOptions
from ebaml.local.facts import compute_local_facts
from ebaml.local.recourse import AllowedValues, Levels, RangeMatch, RecourseOptions, ValueRange
from ebaml.mcp.model_store import ModelStore


class CostStrategyName(enum.StrEnum):
    """The base cost strategy that a tool takes by name, under the difficulties of an agent."""

    UNIT = "unit"
    DATA = "data"


# the base strategies that a tool takes by name; the difficulties of an agent scale them
BASE_STRATEGIES: Mapping[CostStrategyName, CostStrategy] = types.MappingProxyType(
    {CostStrategyName.UNIT: UnitCost(), CostStrategyName.DATA: DataCost()},
)


class AllowedValuesArgument(TypedDict):
    """The allowed values of one feature, as a tool takes them.

    Give ``levels`` for a nominal or ordinal feature. Give ``low``, ``high`` or both, and ``match``, for a continuous
    feature. A missing end has no limit.
    """

    levels: NotRequired[list[str]]
    low: NotRequired[float]
    high: NotRequired[float]
    match: NotRequired[RangeMatch]


def _allowed_values(argument: AllowedValuesArgument) -> AllowedValues:
    levels = argument.get("levels")
    # one check that narrows the two shapes of the JSON argument where it enters ebaml
    if levels is not None:
        return Levels(levels)
    return ValueRange(
        low=argument.get("low", -math.inf),
        high=argument.get("high", math.inf),
        match=argument.get("match", RangeMatch.OVERLAP),
    )


@attrs.frozen(kw_only=True)
class RecourseArguments:
    """The recourse arguments of the tool ``explain_prediction``, as an agent gives them.

    Attributes
    ----------
    immutable: Collection[str]
        The features that the counterfactual must not change.
    allowed: Mapping[str, AllowedValuesArgument]
        The values that some features may take.
    difficulty: Mapping[str, float]
        A cost multiplier for some features, on top of ``cost_strategy``.
    cost_strategy: CostStrategyName
        The base cost of a change, as a member or its text: ``unit`` (every change costs 1) or ``data`` (the distance
        in the training data).
    difficulty_from_model: bool
        True when a language model proposed ``difficulty`` (the tool ``propose_difficulties``).
    max_changes: int | None
        The most changes of a plan.
    plans: int
        How many diverse plans to return.

    """

    immutable: Collection[str] = ()
    allowed: Mapping[str, AllowedValuesArgument] = attrs.field(factory=dict)
    difficulty: Mapping[str, float] = attrs.field(factory=dict)
    cost_strategy: CostStrategyName = attrs.field(
        default=CostStrategyName.UNIT,
        converter=functools.partial(coerce_enum, CostStrategyName),
    )
    difficulty_from_model: bool = False
    max_changes: int | None = None
    plans: int = 1


def recourse_options(arguments: RecourseArguments) -> RecourseOptions:
    """Build the recourse options from the arguments of the tool.

    Parameters
    ----------
    arguments: RecourseArguments
        The arguments.

    Returns
    -------
    RecourseOptions
        The options. The costs are the base strategy, times the difficulties when the agent gives some.

    """
    base = BASE_STRATEGIES[arguments.cost_strategy]
    costs = (
        ScaledCost(base, arguments.difficulty, proposed_by_model=arguments.difficulty_from_model)
        if arguments.difficulty
        else base
    )
    return RecourseOptions(
        immutable=arguments.immutable,
        allowed={name: _allowed_values(argument) for name, argument in arguments.allowed.items()},
        costs=costs,
        max_changes=arguments.max_changes,
        plans=arguments.plans,
    )


@attrs.frozen
class PredictionExplanation:
    """The facts of one prediction and their texts.

    Attributes
    ----------
    facts: LocalFacts
        The prediction, the contributions and the three local explanations.
    texts: FactTexts
        The facts as the JSON texts that the prompts read, for the audience of the call.
    session: RevisionInfo
        The revision that the facts explain, the head, and the number of edits that the head holds.

    """

    facts: LocalFacts
    texts: FactTexts
    session: RevisionInfo


@attrs.frozen(kw_only=True)
class PredictionRequest:
    """What to explain, and how.

    Attributes
    ----------
    instance: Mapping[str, FeatureValue], optional
        The value of every feature, by feature name.
    row: int, optional
        The position of a row of the data of ``--data``. Give exactly one of ``instance`` and ``row``.
    confidence_level: float, default=0.95
        See ``LocalOptions.confidence_level``.
    audience: str, default="layperson"
        The name of a built-in audience. It decides the keys of the texts.
    revision: str, default="current"
        ``original``, ``current`` or the id of a revision of the history. The facts use the model at this revision.
    recourse: RecourseOptions, default=RecourseOptions()
        The limits and the costs of the counterfactual.

    """

    instance: Mapping[str, FeatureValue] | None = None
    row: int | None = None
    confidence_level: float = 0.95
    audience: str = "layperson"
    revision: str = CURRENT
    recourse: RecourseOptions = attrs.field(factory=RecourseOptions)


def instance_values(
    store: ModelStore,
    instance: Mapping[str, FeatureValue] | None,
    row: int | None,
) -> dict[str, FeatureValue]:  # nosemgrep: ebaml-return-attrs-not-container -- the keys are feature names
    """Read the instance of a tool: the values that the agent gives, or a row of the data of ``--data``.

    Parameters
    ----------
    store: ModelStore
        The session.
    instance: Mapping[str, FeatureValue] | None
        The value of every feature, by name.
    row: int | None
        The position of a row of the data.

    Returns
    -------
    dict[str, FeatureValue]
        The value of every feature, by name.

    Raises
    ------
    InvalidInstanceError
        If the agent gives both or none of ``instance`` and ``row``.

    """
    if (instance is None) == (row is None):
        msg = "Give exactly one of instance and row."
        raise InvalidInstanceError(msg)
    return store.row_instance(row) if row is not None else dict(instance or {})


def explain_prediction(store: ModelStore, request: PredictionRequest) -> PredictionExplanation:
    """Compute the local facts of one prediction.

    Parameters
    ----------
    store: ModelStore
        The session.
    request: PredictionRequest
        The instance or the row, the revision, the confidence level, the audience and the recourse options.

    Returns
    -------
    PredictionExplanation
        The facts and their texts.

    """
    values = instance_values(store, request.instance, request.row)
    read = store.read(request.revision)
    options = LocalOptions(confidence_level=request.confidence_level, recourse=request.recourse)
    facts = compute_local_facts(read.model, values, options)
    return PredictionExplanation(
        facts=facts,
        texts=facts_to_texts(facts, resolve_audience(request.audience)),
        session=read.info,
    )
