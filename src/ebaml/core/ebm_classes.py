"""The EBM classes of ``interpret``, imported in one place.

``interpret.glassbox`` has no ``__all__``, so pyrefly reads its public classes as private imports. This module holds the
only import from it, and lists the classes in ``__all__``. The rest of ``ebaml``, the scripts and the tests import the
classes from here.
"""

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

__all__ = ["ExplainableBoostingClassifier", "ExplainableBoostingRegressor"]
