"""Talk 2 EBM in BAML.

Import every public name from ``ebaml``. The names ``describe_graph``, ``describe_ebm``, ``explain_instance``,
``propose_difficulties``, ``LocalExplanation`` and ``EBMDescription`` load on first use, because the module that holds
them can run ``baml generate``. The names ``graph_png``, ``graph_svg``, ``graph_figure`` and ``GraphImageOptions``
load on first use too, because they need the extra ``vision``.
"""

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING, cast

# The hook module must load first, or the modules that load before it stay unchecked.
import ebaml.core.beartype_hook  # ruff: ignore[unused-import] -- the import installs the beartype hook as a side effect
from ebaml.core.custom_types import FeatureType, Normalization
from ebaml.core.exceptions import (
    APIKeyRejectedError,
    DataFileError,
    EBaMlError,
    EBMGraphError,
    EBMGraphShapeError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    EditError,
    GraphTextError,
    HistoryError,
    InvalidInstanceError,
    InvalidOptionError,
    LLMRequestError,
    MissingAPIKeyError,
    MissingExtraError,
    OptionRangeError,
    ProviderUnreachableError,
    RateLimitedError,
    RecourseOptionError,
    SDKUnavailableError,
    SearchLimitError,
    SelectionError,
    StaleRevisionError,
    TokenFitError,
    UnreviewedEditError,
    UnsupportedModelError,
)
from ebaml.edit.bin_edits import ExtendRange, Rebin, SplitBin
from ebaml.edit.constraints import (
    Bounded,
    Constraint,
    FixedScore,
    MaxStep,
    Monotone,
    Unused,
    Violation,
    check_constraints,
)
from ebaml.edit.edit_effect import (
    AffectedProfile,
    ClassifierMetrics,
    ConfusionMatrix,
    EditEffect,
    FeatureProfile,
    MulticlassMetrics,
    RegressorMetrics,
    ScopeMetrics,
    Subgroup,
    affected_profile,
    edit_effect,
)
from ebaml.edit.edit_report import BinChange, ClassShift, EditReport
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import (
    Align,
    AssignScores,
    Clip,
    Edit,
    FitLine,
    Interpolate,
    LimitSteps,
    Monotonize,
    Recenter,
    RemoveFeature,
    RemoveTerms,
    Scale,
    SetScores,
    Shift,
    ZeroTerm,
)
from ebaml.edit.gamchanger_export import GamChangerExport, save_gamchanger
from ebaml.edit.gamchanger_files import GamChangerCommit, apply_gamchanger, load_gamchanger
from ebaml.edit.history import EditEntry, HistoryEntry, HistoryRow, SaveEntry
from ebaml.edit.selections import (
    Bins,
    Categories,
    EveryBin,
    MissingBin,
    Selection,
    UnseenBin,
    ValueInterval,
    WholeGraph,
)
from ebaml.graphs.ebm_graph import EBMGraph, GraphSeries
from ebaml.graphs.model_kinds import (
    BinaryClassifier,
    ModelKind,
    MulticlassClassifier,
    Regressor,
    model_kind,
)
from ebaml.graphs.token_budget import check_token_budget
from ebaml.llm.difficulties import DifficultyProposal, FeatureDifficulty
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.requests import (
    DescribeEBMRequest,
    DescribeGraphRequest,
    ExplainInstanceRequest,
    ProposeDifficultiesRequest,
)
from ebaml.local.audience import (
    Audience,
    DataScientist,
    Layperson,
    likelihood_word,
    points_towards,
    resolve_audience,
    round_value,
    sample_size_word,
    strength_word,
)
from ebaml.local.costs import DIFFICULTY_MULTIPLIERS, CostStrategy, DataCost, Difficulty, ScaledCost, UnitCost, UserCost
from ebaml.local.covering_backend import (
    CoveringModel,
    MipBackend,
    ModelResult,
    ModelStatus,
    SolveSettings,
    Stop,
)
from ebaml.local.covering_problem import Choice, Cover, CoveringProblem
from ebaml.local.covering_solvers import (
    CoveringSolver,
    GreedyCover,
    LocalSearchCover,
    MipCover,
)
from ebaml.local.decisions import Decision, Interval, Threshold
from ebaml.local.fact_text import FactTexts, facts_to_texts
from ebaml.local.fact_types import (
    Change,
    CompletionStep,
    Contribution,
    Counterfactual,
    LocalFacts,
    LocalOptions,
    Prediction,
    SufficientReason,
)
from ebaml.local.facts import compute_local_facts
from ebaml.local.local_kinds import OutputKind
from ebaml.local.minimality import MinimalityStatus
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver
from ebaml.local.recourse import Levels, RangeMatch, RecourseOptions, ValueRange

if TYPE_CHECKING:
    from ebaml.graphs.graph_image import GraphImageOptions, graph_figure, graph_png, graph_svg
    from ebaml.llm.descriptors import (
        EBMDescription,
        LocalExplanation,
        describe_ebm,
        describe_graph,
        explain_instance,
        propose_difficulties,
    )

__version__: str = version("ebaml")

__all__ = [
    "DIFFICULTY_MULTIPLIERS",
    "APIKeyRejectedError",
    "AffectedProfile",
    "Align",
    "AssignScores",
    "Audience",
    "BinChange",
    "BinaryClassifier",
    "Bins",
    "Bounded",
    "Categories",
    "Change",
    "Choice",
    "ClassShift",
    "ClassifierMetrics",
    "Clip",
    "CompletionStep",
    "ConfusionMatrix",
    "Constraint",
    "Contribution",
    "CostStrategy",
    "Counterfactual",
    "Cover",
    "CoveringModel",
    "CoveringProblem",
    "CoveringSolver",
    "DataCost",
    "DataFileError",
    "DataScientist",
    "Decision",
    "DescribeEBMRequest",
    "DescribeGraphRequest",
    "Difficulty",
    "DifficultyProposal",
    "EBMDescription",
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphShapeError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "EBaMlError",
    "Edit",
    "EditEffect",
    "EditEntry",
    "EditError",
    "EditReport",
    "EditSession",
    "EveryBin",
    "ExplainInstanceRequest",
    "ExtendRange",
    "FactTexts",
    "FeatureDifficulty",
    "FeatureProfile",
    "FeatureType",
    "FitLine",
    "FixedScore",
    "GamChangerCommit",
    "GamChangerExport",
    "GraphImageOptions",
    "GraphOptions",
    "GraphSeries",
    "GraphTextError",
    "GreedyCover",
    "HistoryEntry",
    "HistoryError",
    "HistoryRow",
    "Interpolate",
    "Interval",
    "InvalidInstanceError",
    "InvalidOptionError",
    "LLMRequestError",
    "Layperson",
    "Levels",
    "LimitSteps",
    "LocalExplanation",
    "LocalFacts",
    "LocalOptions",
    "LocalSearchCover",
    "MaxStep",
    "MinimalityStatus",
    "MipBackend",
    "MipCover",
    "MissingAPIKeyError",
    "MissingBin",
    "MissingExtraError",
    "ModelKind",
    "ModelOptions",
    "ModelResult",
    "ModelStatus",
    "Monotone",
    "Monotonize",
    "MulticlassClassifier",
    "MulticlassMetrics",
    "Normalization",
    "OptionRangeError",
    "OrToolsBackend",
    "OrToolsSolver",
    "OutputKind",
    "Prediction",
    "PromptOptions",
    "ProposeDifficultiesRequest",
    "ProviderUnreachableError",
    "RangeMatch",
    "RateLimitedError",
    "Rebin",
    "Recenter",
    "RecourseOptionError",
    "RecourseOptions",
    "Regressor",
    "RegressorMetrics",
    "RemoveFeature",
    "RemoveTerms",
    "SDKUnavailableError",
    "SaveEntry",
    "Scale",
    "ScaledCost",
    "ScopeMetrics",
    "SearchLimitError",
    "Selection",
    "SelectionError",
    "SetScores",
    "Shift",
    "SolveSettings",
    "SplitBin",
    "StaleRevisionError",
    "Stop",
    "Subgroup",
    "SufficientReason",
    "Threshold",
    "TokenFitError",
    "UnitCost",
    "UnreviewedEditError",
    "UnseenBin",
    "UnsupportedModelError",
    "Unused",
    "UserCost",
    "ValueInterval",
    "ValueRange",
    "Violation",
    "WholeGraph",
    "ZeroTerm",
    "__version__",
    "affected_profile",
    "apply_gamchanger",
    "check_constraints",
    "check_token_budget",
    "compute_local_facts",
    "describe_ebm",
    "describe_graph",
    "edit_effect",
    "explain_instance",
    "facts_to_texts",
    "graph_figure",
    "graph_png",
    "graph_svg",
    "likelihood_word",
    "load_gamchanger",
    "model_kind",
    "points_towards",
    "propose_difficulties",
    "resolve_audience",
    "round_value",
    "sample_size_word",
    "save_gamchanger",
    "strength_word",
]


# nosemgrep: ebaml-no-object-type -- PEP 562 returns a function or a class chosen by name, so no single type fits
def __getattr__(name: str) -> object:
    """Load a name of ``ebaml.llm.descriptors`` or ``ebaml.graphs.graph_image`` on first use.

    The first module can run ``baml generate`` when it is imported, and the second needs the extra ``vision``, so
    ``import ebaml`` must not load them. The loaded name goes into the module namespace, and the next access skips this
    function.

    Parameters
    ----------
    name
        The name of the attribute.

    Returns
    -------
    object
        The function or class with this name.

    Raises
    ------
    AttributeError
        If ``name`` is not a public name that loads on first use.

    """
    # the names that load on first use, by the module that holds them
    lazy_modules = {
        "EBMDescription": "ebaml.llm.descriptors",
        "LocalExplanation": "ebaml.llm.descriptors",
        "describe_ebm": "ebaml.llm.descriptors",
        "describe_graph": "ebaml.llm.descriptors",
        "explain_instance": "ebaml.llm.descriptors",
        "propose_difficulties": "ebaml.llm.descriptors",
        "GraphImageOptions": "ebaml.graphs.graph_image",
        "graph_figure": "ebaml.graphs.graph_image",
        "graph_png": "ebaml.graphs.graph_image",
        "graph_svg": "ebaml.graphs.graph_image",
    }
    if name not in lazy_modules:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(importlib.import_module(lazy_modules[name]), name)
    globals()[name] = value
    return cast(object, value)  # getattr returns Any, and the hint names the real type


def __dir__() -> list[str]:
    """List the public names, including the names that load on first use.

    Returns
    -------
    list[str]
        The sorted names of ``__all__``.

    """
    return sorted(__all__)
