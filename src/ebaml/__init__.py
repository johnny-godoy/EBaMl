"""Talk 2 EBM in BAML."""

from importlib.metadata import version

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
from ebaml.core.model_kinds import (  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first
    BinaryClassifier,
    ModelKind,
    MulticlassClassifier,
    Regressor,
    model_kind,
)
from ebaml.graphs.ebm_graph import EBMGraph  # ruff: ignore[module-import-not-at-top-of-file] -- the beartype hook must run first

__version__: str = version("ebaml")

__all__ = [
    "APIKeyRejectedError",
    "Audience",
    "BinaryClassifier",
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphShapeError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "EBaMlError",
    "FeatureType",
    "GraphTextError",
    "InvalidInstanceError",
    "InvalidOptionError",
    "LLMRequestError",
    "MissingAPIKeyError",
    "ModelKind",
    "MulticlassClassifier",
    "Normalization",
    "OptionRangeError",
    "ProviderUnreachableError",
    "RateLimitedError",
    "Regressor",
    "SDKUnavailableError",
    "UnsupportedModelError",
    "__version__",
    "model_kind",
]
