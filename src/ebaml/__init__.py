"""Talk 2 EBM in BAML.

Import every public name from ``ebaml``. The names ``describe_graph``, ``describe_ebm``, ``explain_instance`` and
``LocalExplanation`` and ``EBMDescription`` load on first use, because the module that holds them can run
``baml generate``.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, cast

# The hook module must load first, or the modules that load before it stay unchecked.
import ebaml.core.beartype_hook  # ruff: ignore[unused-import] -- the import installs the beartype hook as a side effect
from ebaml.core.custom_types import FeatureType, Normalization
from ebaml.core.exceptions import (
    APIKeyRejectedError,
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
    OptionRangeError,
    ProviderUnreachableError,
    RateLimitedError,
    SDKUnavailableError,
    SearchLimitError,
    SelectionError,
    StaleRevisionError,
    TokenFitError,
    UnreviewedEditError,
    UnsupportedModelError,
)
from ebaml.edit.edit_report import BinChange, EditReport
from ebaml.edit.edit_session import EditSession
from ebaml.edit.edits import (
    Align,
    Edit,
    FitLine,
    Interpolate,
    Monotonize,
    Recenter,
    RemoveFeature,
    RemoveTerms,
    Scale,
    SetScores,
    Shift,
    ZeroTerm,
)
from ebaml.edit.history import EditEntry, HistoryEntry, HistoryRow, SaveEntry
from ebaml.edit.selections import (
    Bins,
    Categories,
    MissingBin,
    Selection,
    UnseenBin,
    ValueInterval,
    WholeGraph,
)
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.model_kinds import (
    BinaryClassifier,
    ModelKind,
    MulticlassClassifier,
    Regressor,
    model_kind,
)
from ebaml.graphs.token_budget import check_token_budget
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions
from ebaml.llm.requests import DescribeEBMRequest, DescribeGraphRequest, ExplainInstanceRequest
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
from ebaml.local.facts import (
    Change,
    CompletionStep,
    Contribution,
    Counterfactual,
    LocalFacts,
    LocalOptions,
    Prediction,
    SufficientReason,
    compute_local_facts,
)
from ebaml.local.local_kinds import OutputKind
from ebaml.local.minimality import MinimalityStatus
from ebaml.local.ortools_backend import OrToolsBackend, OrToolsSolver

if TYPE_CHECKING:
    from ebaml.llm.descriptors import EBMDescription, LocalExplanation, describe_ebm, describe_graph, explain_instance

__version__: str = version("ebaml")

__all__ = [
    "APIKeyRejectedError",
    "Align",
    "Audience",
    "BinChange",
    "BinaryClassifier",
    "Bins",
    "Categories",
    "Change",
    "Choice",
    "CompletionStep",
    "Contribution",
    "Counterfactual",
    "Cover",
    "CoveringModel",
    "CoveringProblem",
    "CoveringSolver",
    "DataScientist",
    "Decision",
    "DescribeEBMRequest",
    "DescribeGraphRequest",
    "EBMDescription",
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphShapeError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "EBaMlError",
    "Edit",
    "EditEntry",
    "EditError",
    "EditReport",
    "EditSession",
    "ExplainInstanceRequest",
    "FactTexts",
    "FeatureType",
    "FitLine",
    "GraphOptions",
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
    "LocalExplanation",
    "LocalFacts",
    "LocalOptions",
    "LocalSearchCover",
    "MinimalityStatus",
    "MipBackend",
    "MipCover",
    "MissingAPIKeyError",
    "MissingBin",
    "ModelKind",
    "ModelOptions",
    "ModelResult",
    "ModelStatus",
    "Monotonize",
    "MulticlassClassifier",
    "Normalization",
    "OptionRangeError",
    "OrToolsBackend",
    "OrToolsSolver",
    "OutputKind",
    "Prediction",
    "PromptOptions",
    "ProviderUnreachableError",
    "RateLimitedError",
    "Recenter",
    "Regressor",
    "RemoveFeature",
    "RemoveTerms",
    "SDKUnavailableError",
    "SaveEntry",
    "Scale",
    "SearchLimitError",
    "Selection",
    "SelectionError",
    "SetScores",
    "Shift",
    "SolveSettings",
    "StaleRevisionError",
    "Stop",
    "SufficientReason",
    "Threshold",
    "TokenFitError",
    "UnreviewedEditError",
    "UnseenBin",
    "UnsupportedModelError",
    "ValueInterval",
    "WholeGraph",
    "ZeroTerm",
    "__version__",
    "check_token_budget",
    "compute_local_facts",
    "describe_ebm",
    "describe_graph",
    "explain_instance",
    "facts_to_texts",
    "likelihood_word",
    "model_kind",
    "points_towards",
    "resolve_audience",
    "round_value",
    "sample_size_word",
    "strength_word",
]


# nosemgrep: ebaml-no-object-type -- PEP 562 returns a function or a class chosen by name, so no single type fits
def __getattr__(name: str) -> object:
    """Load a name of ``ebaml.llm.descriptors`` on first use.

    That module can run ``baml generate`` when it is imported, so ``import ebaml`` must not load it. The loaded name
    goes into the module namespace, and the next access skips this function.

    Parameters
    ----------
    name
        The name of the attribute.

    Returns
    -------
    object
        The function or class of ``ebaml.llm.descriptors`` with this name.

    Raises
    ------
    AttributeError
        If ``name`` is not a public name that loads on first use.

    """
    lazy_names = {"EBMDescription", "LocalExplanation", "describe_ebm", "describe_graph", "explain_instance"}
    if name not in lazy_names:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from ebaml.llm import descriptors  # ruff: ignore[import-outside-top-level] -- the import must wait until first use

    value = getattr(descriptors, name)
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
