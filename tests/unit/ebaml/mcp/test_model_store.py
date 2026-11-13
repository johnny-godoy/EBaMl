"""Tests of the session state and the loading of the files that the user names."""

import ast
from pathlib import Path

import polars as pl
import pytest
import skops.io

import ebaml
from ebaml.core.exceptions import InvalidInstanceError, UnsupportedModelError
from ebaml.mcp.model_store import ModelStore, load_data, load_model
from tests.unit.ebaml.conftest import FittedClassifier, as_frame, save_model


class UntrustedType:
    """A type that skops does not trust by default."""


def test_load_model_reads_a_joblib_file(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    ebm, _ = continuous_ebm
    save_model(ebm, tmp_path / "model.joblib")
    assert list(load_model(tmp_path / "model.joblib").feature_names_in_) == ["a", "b", "c"]


def test_load_model_rejects_a_file_that_holds_no_ebm(tmp_path: Path) -> None:
    save_model({"not": "a model"}, tmp_path / "other.joblib")
    with pytest.raises(UnsupportedModelError, match="holds a dict"):
        load_model(tmp_path / "other.joblib")


def test_load_model_reads_a_skops_file(
    continuous_ebm: FittedClassifier,
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


def test_load_data_reads_a_parquet_file_and_keeps_its_column_types(tmp_path: Path) -> None:
    """Parquet keeps the type of a column. A CSV reader would read the text column ``code`` as integers."""
    pl.DataFrame({"code": ["001", "002"], "b": [2, 4]}).write_parquet(tmp_path / "data.parquet")
    frame = load_data(tmp_path / "data.parquet")
    assert frame["code"].to_list() == ["001", "002"]
    assert frame["b"].to_list() == [2, 4]


def test_load_data_reads_a_csv_file(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert load_data(tmp_path / "data.csv")["b"].to_list() == [2, 4]


def test_load_data_chooses_the_reader_by_the_first_bytes_and_not_by_the_extension(tmp_path: Path) -> None:
    """A Parquet file named ``.csv``, and a CSV file named ``.parquet``, still load."""
    pl.DataFrame({"a": [1, 3], "b": [2, 4]}).write_parquet(tmp_path / "parquet_inside.csv")
    (tmp_path / "csv_inside.parquet").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert load_data(tmp_path / "parquet_inside.csv")["b"].to_list() == [2, 4]
    assert load_data(tmp_path / "csv_inside.parquet")["b"].to_list() == [2, 4]


def test_load_data_reads_a_csv_file_whose_first_column_starts_with_the_parquet_mark(tmp_path: Path) -> None:
    """The mark at the start alone does not make a Parquet file: the file must end with the mark too."""
    (tmp_path / "data.csv").write_text("PAR1,b\n1,2\n3,4\n", encoding="utf-8")
    assert load_data(tmp_path / "data.csv")["b"].to_list() == [2, 4]


def test_row_instance_without_data_is_an_error(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    with pytest.raises(InvalidInstanceError, match="--data"):
        ModelStore(ebm=ebm).row_instance(0)


def test_row_instance_checks_the_row_and_the_columns(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, frame = continuous_ebm
    data = as_frame(frame)
    with pytest.raises(InvalidInstanceError, match="row -1"):
        ModelStore(ebm=ebm, data=data).row_instance(-1)
    with pytest.raises(InvalidInstanceError, match="no column"):
        ModelStore(ebm=ebm, data=data.drop("b")).row_instance(0)
    assert set(ModelStore(ebm=ebm, data=data).row_instance(len(data) - 1)) == {"a", "b", "c"}


_LOADERS = {"joblib.load", "pickle.load", "pickle.loads", "skops.io.load", "skops.io.loads"}


def _loader_calls(node: ast.AST, function: str = "<module>") -> list[tuple[str, str]]:
    """Find each call that reads a model, a Parquet file or a CSV file below a syntax node.

    Returns
    -------
    list[tuple[str, str]]
        The name of the call and the name of the function that holds it, for each call.

    """
    found: list[tuple[str, str]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            found += _loader_calls(child, child.name)
            continue
        if isinstance(child, ast.Call):
            name = ast.unparse(child.func)
            if name in _LOADERS or name.endswith((".read_csv", ".read_parquet")):
                found.append((name, function))
        found += _loader_calls(child, function)
    return found


def test_only_load_model_reads_a_pickle_and_only_model_store_reads_files() -> None:
    """Loading a pickle runs code. The only reader is ``load_model``, which only the command line of the user calls."""
    package = Path(ebaml.__file__).parent
    calls = {
        source.relative_to(package).as_posix(): _loader_calls(ast.parse(source.read_text(encoding="utf-8")))
        for source in package.rglob("*.py")
    }
    readers = {source: found for source, found in calls.items() if found}
    assert readers == {
        "mcp/model_store.py": [
            ("skops.io.load", "_load_skops"),
            ("joblib.load", "load_model"),
            ("pl.read_parquet", "load_data"),
            ("pl.read_csv", "load_data"),
        ],
    }
