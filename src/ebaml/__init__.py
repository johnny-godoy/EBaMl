"""Talk 2 EBM in BAML.

The names of this package are the public API. ``describe_graph``, ``describe_ebm``, ``explain_instance`` and
``LocalExplanation`` load on first use, because their module imports the generated ``baml_sdk`` and can run
``baml generate``. So ``import ebaml`` stays fast, and code that calls no LLM never loads the SDK.
"""

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, cast

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# The hook must run before any submodule import, or those modules stay unchecked.
# The numeric tower lets an int satisfy a float hint, as PEP 484 says.
# Checking annotated variables is off, because structlog's lazy logger proxy does not match its own logger type.
beartype_this_package(conf=BeartypeConf(is_pep484_tower=True, claw_is_pep526=False))  # ruff: ignore[non-empty-init-module] -- the hook must run before the submodule imports below

from ebaml.core.custom_types import Audience, FeatureType, Normalization  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
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
from ebaml.llm.options import GraphOptions, ModelOptions, PromptOptions  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
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

if TYPE_CHECKING:
    from collections.abc import Callable

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
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphShapeError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "EBaMlError",
    "FactTexts",
    "FeatureType",
    "GraphOptions",
    "GraphTextError",
    "InvalidInstanceError",
    "InvalidOptionError",
    "LLMRequestError",
    "LocalExplanation",
    "LocalFacts",
    "LocalOptions",
    "MissingAPIKeyError",
    "ModelKind",
    "ModelOptions",
    "MulticlassClassifier",
    "Normalization",
    "OptionRangeError",
    "Prediction",
    "PromptOptions",
    "ProviderUnreachableError",
    "RateLimitedError",
    "Regressor",
    "SDKUnavailableError",
    "SufficientReason",
    "UnsupportedModelError",
    "__version__",
    "compute_local_facts",
    "describe_ebm",
    "describe_graph",
    "explain_instance",
    "facts_to_texts",
    "model_kind",
]


def __getattr__(name: str) -> "Callable[..., str | LocalExplanation] | type[LocalExplanation]":
    """Load a name of ``ebaml.llm.descriptors`` on first use.

    Parameters
    ----------
    name: str
        The name to load.

    Returns
    -------
    Callable[..., str | LocalExplanation] | type[LocalExplanation]
        The function or the class.

    Raises
    ------
    AttributeError
        If the package has no such name.

    """
    if name not in {"LocalExplanation", "describe_ebm", "describe_graph", "explain_instance"}:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    # getattr has no static type; the four names are three functions and one class
    value = cast(
        "Callable[..., str | LocalExplanation] | type[LocalExplanation]",
        getattr(import_module("ebaml.llm.descriptors"), name),
    )
    globals()[name] = value  # the next access finds the name without this function
    return value
