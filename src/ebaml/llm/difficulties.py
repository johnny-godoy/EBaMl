"""The proposal of how hard each change of a recourse plan is, from an LLM, as a cost strategy that the caller checks.

``propose_difficulties`` in ``ebaml.llm.descriptors`` sends the inputs of an instance to the BAML function
``ProposeFeatureDifficulty``. This module writes the inputs as text and reads the reply. It runs no search: the caller
reads the proposal, corrects it, and passes its ``costs`` to ``RecourseOptions``. A difficulty that an LLM proposed is
an assumption, so the ``ScaledCost`` records its source and the facts say so.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import attrs
import structlog

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.terms import main_effect_features
from ebaml.local.costs import DIFFICULTY_MULTIPLIERS, CostStrategy, Difficulty, ScaledCost

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)


@runtime_checkable
class Rating(Protocol):
    """One rating of the reply, as ``baml_sdk.FeatureDifficulty`` holds it: a feature, a level and a reason."""

    @property
    def feature(self) -> str:
        """The name of the input that the reply rates."""
        ...

    @property
    def difficulty(self) -> str:
        """The level of the scale, as its text."""
        ...

    @property
    def reason(self) -> str:
        """Why the reply chose the level."""
        ...


@attrs.frozen
class FeatureDifficulty:
    """The difficulty that the LLM proposed for the change of one input.

    Attributes
    ----------
    feature: str
        The input.
    difficulty: Difficulty
        The level of the scale.
    multiplier: float
        The cost multiplier of the level (``DIFFICULTY_MULTIPLIERS``).
    reason: str
        Why the LLM chose the level, in one sentence.

    """

    feature: str
    difficulty: Difficulty
    multiplier: float
    reason: str


@attrs.frozen
class DifficultyProposal:
    """The difficulties that the LLM proposed, and the cost strategy that they give.

    Attributes
    ----------
    difficulties: list[FeatureDifficulty]
        One proposal for each input of the model that the reply rates, in the order of the reply.
    costs: ScaledCost
        The base strategy times the multiplier of each rated input. Its ``proposed_by_model`` is true. An input that
        the reply does not rate keeps the cost of the base strategy.

    """

    difficulties: list[FeatureDifficulty]
    costs: ScaledCost


def features_text(ebm: ExplainableBoostingModel, instance: Mapping[str, FeatureValue]) -> str:
    """Write the inputs of an instance for the prompt: one line with the name, the type and the value of each input.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM. Only the inputs with a main-effect term are listed, because only they can be part of a plan.
    instance: Mapping[str, FeatureValue]
        The value of every feature, by name.

    Returns
    -------
    str
        The lines, such as ``- income (continuous), current value 40``.

    """
    lines = []
    for feature_index in main_effect_features(ebm):
        name = ebm.feature_names_in_[feature_index]
        feature_type = EBMGraph.from_ebm(ebm, feature_index).feature_type.value
        lines.append(f"- {name} ({feature_type}), current value {instance[name]}")
    return "\n".join(lines)


def read_proposal(
    ratings: Sequence[Rating],
    feature_names: Sequence[str],
    base: CostStrategy,
) -> DifficultyProposal:
    """Turn the ratings of the reply into a proposal, and drop each rating of a name that the model does not have.

    Parameters
    ----------
    ratings: Sequence[Rating]
        The ratings of the reply.
    feature_names: Sequence[str]
        The inputs of the model.
    base: CostStrategy
        The strategy whose costs the multipliers scale.

    Returns
    -------
    DifficultyProposal
        The difficulties of the known inputs, the first rating of each, and the strategy that they give.

    """
    known = set(feature_names)
    difficulties: list[FeatureDifficulty] = []
    for rating in ratings:
        feature = rating.feature
        if feature not in known or feature in {item.feature for item in difficulties}:
            # a name that the model lacks cannot get a cost, and a second rating of a name would hide the first
            logger.warning("difficulty_dropped", feature=feature, known=feature in known)
            continue
        difficulty = Difficulty(rating.difficulty)
        difficulties.append(FeatureDifficulty(feature, difficulty, DIFFICULTY_MULTIPLIERS[difficulty], rating.reason))
    multipliers = {item.feature: item.multiplier for item in difficulties}
    return DifficultyProposal(difficulties, ScaledCost(base, multipliers, proposed_by_model=True))
