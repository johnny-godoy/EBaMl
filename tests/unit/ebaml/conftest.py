"""Unfitted EBM instances with preset attributes, so tests can pass them where an EBM type is required."""

from typing import Any

from interpret.glassbox import ExplainableBoostingClassifier


def fake_model(**attributes: Any) -> ExplainableBoostingClassifier:  # ruff: ignore[any-type] -- attribute values are arbitrary test data
    """Build an EBM without fitting it, and set the given attributes on it.

    Returns
    -------
    ExplainableBoostingClassifier
        An instance that passes the runtime type check but has only the given attributes.

    """
    model = ExplainableBoostingClassifier.__new__(ExplainableBoostingClassifier)
    vars(model).update(attributes)
    return model
