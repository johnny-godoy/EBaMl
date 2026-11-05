"""Tests of the session state and the loading of the files that the user names."""

from pathlib import Path

import joblib  # ruff: ignore[banned-api] -- the test writes the joblib file that ebaml-mcp loads
import pandas as pd  # ruff: ignore[banned-api] -- the test builds the pandas frame that interpret fits on
import pytest
import skops.io
from interpret.glassbox import ExplainableBoostingClassifier  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports

from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.mcp.model_store import ModelStore, load_data, load_model
from tests.unit.ebaml.conftest import as_frame


class UntrustedType:
    """A type that skops does not trust by default."""


def test_load_model_reads_a_joblib_file(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
) -> None:
    ebm, _ = continuous_ebm
    joblib.dump(ebm, tmp_path / "model.joblib")
    assert list(load_model(tmp_path / "model.joblib").feature_names_in_) == ["a", "b", "c"]


def test_load_model_rejects_a_file_that_holds_no_ebm(tmp_path: Path) -> None:
    joblib.dump({"not": "a model"}, tmp_path / "other.joblib")
    with pytest.raises(UnsupportedModelError, match="holds a dict"):
        load_model(tmp_path / "other.joblib")


def test_load_model_reads_a_skops_file(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
    tmp_path: Path,
) -> None:
    ebm, _ = continuous_ebm
    skops.io.dump(ebm, tmp_path / "model.skops")
    assert list(load_model(tmp_path / "model.skops").feature_names_in_) == ["a", "b", "c"]


def test_load_model_rejects_a_skops_file_that_holds_no_ebm(tmp_path: Path) -> None:
    skops.io.dump({"not": "a model"}, tmp_path / "other.skops")
    with pytest.raises(UnsupportedModelError, match="holds a dict"):
        load_model(tmp_path / "other.skops")


def test_load_model_does_not_trust_other_types_in_a_skops_file(tmp_path: Path) -> None:
    """A skops file may name any type. The loader trusts only the EBM classes, so a file with another type fails."""
    skops.io.dump(UntrustedType(), tmp_path / "other_type.skops")
    with pytest.raises(UnsupportedModelError, match="does not trust"):
        load_model(tmp_path / "other_type.skops")


def test_load_data_reads_a_csv_file(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert load_data(tmp_path / "data.csv")["b"].to_list() == [2, 4]


def test_row_instance_without_data_is_an_error(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, _ = continuous_ebm
    with pytest.raises(InvalidInstanceError, match="--data"):
        ModelStore(ebm=ebm).row_instance(0)


def test_row_instance_checks_the_row_and_the_columns(
    continuous_ebm: tuple[ExplainableBoostingClassifier, pd.DataFrame],
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    with pytest.raises(InvalidInstanceError, match="row -1"):
        ModelStore(ebm=ebm, data=data).row_instance(-1)
    with pytest.raises(InvalidInstanceError, match="no column"):
        ModelStore(ebm=ebm, data=data.drop("b")).row_instance(0)
    assert set(ModelStore(ebm=ebm, data=data).row_instance(len(data) - 1)) == {"a", "b", "c"}
