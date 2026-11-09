"""Tests for the single import of the EBM classes of interpret."""

import interpret.glassbox

from ebaml.core import ebm_classes


def test_the_module_exports_the_classes_of_interpret() -> None:
    assert ebm_classes.__all__ == ["ExplainableBoostingClassifier", "ExplainableBoostingRegressor"]
    for name in ebm_classes.__all__:
        assert getattr(ebm_classes, name) is getattr(interpret.glassbox, name)
