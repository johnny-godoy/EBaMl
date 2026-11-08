"""Talk 2 EBM in BAML.

Import every public name from ``ebaml``. The names ``describe_graph``, ``describe_ebm``, ``explain_instance`` and
``LocalExplanation`` load on first use, because the module that holds them can run ``baml generate``.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, cast

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# The hook must run before any submodule import, or those modules stay unchecked.
# The numeric tower lets an int satisfy a float hint, as PEP 484 says.
# Checking annotated variables is off, because structlog's lazy logger proxy does not match its own logger type.
beartype_this_package(conf=BeartypeConf(is_pep484_tower=True, claw_is_pep526=False))  # ruff: ignore[non-empty-init-module] -- the hook must run before the submodule imports below

from ebaml.core.custom_types import FeatureType, Normalization  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.core.exceptions import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
    APIKeyRejectedError,
    EBaMlError,
    EBMGraphError,
    EBMGraphShapeError,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    GraphTextError,
    InvalidInstanceError,
    InvalidOptionError,
    LLMRequestError,
    MissingAPIKeyError,
    OptionRangeError,
    ProviderUnreachableError,
    RateLimitedError,
    SDKUnavailableError,
    SearchLimitError,
    TokenFitError,
    UnsupportedModelError,
)
from ebaml.graphs.ebm_graph import EBMGraph  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.graphs.model_kinds import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
    BinaryClassifier,
    ModelKind,
    MulticlassClassifier,
    Regressor,
    model_kind,
)
from ebaml.graphs.token_budget import check_token_budget  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.local.audience import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
    Audience,
    DataScientist,
    Layperson,
    likelihood_word,
    points_towards,
    resolve_audience,
    round_value,
    strength_word,
)
from ebaml.local.covering_solvers import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
    CoveringSearch,
    Engine,
    Exact,
    Greedy,
    Heuristic,
    Stochastic,
)
from ebaml.local.decisions import Decision, Interval, Threshold  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.local.fact_text import FactTexts, facts_to_texts  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
from ebaml.local.facts import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
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
from ebaml.local.local_kinds import OutputKind  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first

if TYPE_CHECKING:
    from ebaml.llm.descriptors import LocalExplanation, describe_ebm, describe_graph, explain_instance

__version__: str = version("ebaml")

__all__ = [
    "APIKeyRejectedError",
    "Audience",
    "BinaryClassifier",
    "Change",
    "CompletionStep",
    "Contribution",
    "Counterfactual",
    "CoveringSearch",
    "DataScientist",
    "Decision",
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphShapeError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "EBaMlError",
    "Engine",
    "Exact",
    "FactTexts",
    "FeatureType",
    "GraphOptions",
    "GraphTextError",
    "Greedy",
    "Heuristic",
    "Interval",
    "InvalidInstanceError",
    "InvalidOptionError",
    "LLMRequestError",
    "Layperson",
    "LocalExplanation",
    "LocalFacts",
    "LocalOptions",
    "MissingAPIKeyError",
    "ModelKind",
    "ModelOptions",
    "MulticlassClassifier",
    "Normalization",
    "OptionRangeError",
    "OutputKind",
    "Prediction",
    "PromptOptions",
    "ProviderUnreachableError",
    "RateLimitedError",
    "Regressor",
    "SDKUnavailableError",
    "SearchLimitError",
    "Stochastic",
    "SufficientReason",
    "Threshold",
    "TokenFitError",
    "UnsupportedModelError",
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
    lazy_names = {"LocalExplanation", "describe_ebm", "describe_graph", "explain_instance"}
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
