"""Talk 2 EBM in BAML."""

from importlib.metadata import version

from ebaml.custom_types import Audience, FeatureType, Normalization
from ebaml.exceptions import EBMGraphError, EBMGraphSimplificationError, EBMGraphTokenLimitError, InvalidOptionError
from ebaml.graphs import EBMGraph

__version__: str = version("ebaml")

__all__ = [
    "Audience",
    "EBMGraph",
    "EBMGraphError",
    "EBMGraphSimplificationError",
    "EBMGraphTokenLimitError",
    "FeatureType",
    "InvalidOptionError",
    "Normalization",
    "__version__",
]
