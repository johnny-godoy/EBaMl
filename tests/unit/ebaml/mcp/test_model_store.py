"""Tests of the session state and the loading of the files that the user names."""

import ast
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest
import skops.io

import ebaml
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import DataFileError, EditError, InvalidInstanceError, UnsupportedModelError
from ebaml.edit.edit_session import RevisionInfo
from ebaml.edit.edits import SetScores
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins
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


def _excel_like_bytes() -> bytes:
    """Build a small zip archive, which is the container of an Excel file.

    Returns
    -------
    bytes
        The archive.

    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sheet.xml", "<sheet/>")
    return buffer.getvalue()


_NEITHER_PARQUET_NOR_CSV = {
    "an Excel file": _excel_like_bytes(),
    "random bytes": bytes(range(256)),
    "text that is not UTF-8": "name,city\nAna,Concepción\n".encode("latin-1"),
}


@pytest.mark.parametrize("content", _NEITHER_PARQUET_NOR_CSV.values(), ids=_NEITHER_PARQUET_NOR_CSV.keys())
def test_load_data_names_the_file_and_the_formats_when_it_is_neither_parquet_nor_csv(
    content: bytes,
    tmp_path: Path,
) -> None:
    """Polars reads a binary file as a frame of no rows with the bytes as column names, so the error came much later."""
    (tmp_path / "rows.dat").write_bytes(content)
    with pytest.raises(DataFileError, match=r"rows\.dat.*neither Parquet nor UTF-8 CSV.*write_parquet.*write_csv"):
        load_data(tmp_path / "rows.dat")


def test_load_data_says_that_a_parquet_file_is_cut_off(tmp_path: Path) -> None:
    pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / "rows.parquet")
    (tmp_path / "rows.parquet").write_bytes((tmp_path / "rows.parquet").read_bytes()[:-6])
    with pytest.raises(DataFileError, match=r"rows\.parquet.*starts with the Parquet mark.*does not end with it"):
        load_data(tmp_path / "rows.parquet")


def test_load_data_says_that_a_file_is_empty(tmp_path: Path) -> None:
    (tmp_path / "rows.csv").write_bytes(b"")
    with pytest.raises(DataFileError, match=r"rows\.csv.*is empty"):
        load_data(tmp_path / "rows.csv")


@pytest.mark.parametrize(
    ("content", "reader"),
    [(b"PAR1 not a real body PAR1", "Parquet"), (b"a,b\n1,2,3\n", "CSV")],
    ids=["parquet", "csv"],
)
def test_load_data_names_the_file_when_the_reader_fails(content: bytes, reader: str, tmp_path: Path) -> None:
    (tmp_path / "rows.dat").write_bytes(content)
    with pytest.raises(DataFileError, match=rf"rows\.dat.*as {reader}") as caught:
        load_data(tmp_path / "rows.dat")
    assert isinstance(caught.value.__cause__, pl.exceptions.PolarsError)


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


def test_a_read_of_a_revision_gives_the_model_with_the_edits_of_that_revision(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    session = store.editing()
    session.apply(SetScores("a", Bins(0, 1), 5.0), "Raise the lowest bins")
    edit = session.revision
    before, after, same = store.read("original"), store.read(edit), store.read()
    assert before.info == RevisionInfo("original", edit, 1)
    assert after.info == same.info == RevisionInfo(edit, edit, 1)
    assert term_view(after.model, "a").scores[1] == pytest.approx(5.0)
    assert term_view(before.model, "a").scores[1] == term_view(ebm, "a").scores[1]
    # the store never changes the model that the user loaded, and a read gives a copy
    assert after.model is not session.model_at(edit)
    assert term_view(ebm, "a").scores[1] != pytest.approx(5.0)
    with pytest.raises(EditError, match="no revision 'nothing'"):
        store.read("nothing")


def test_a_model_that_the_edits_do_not_support_has_reads_and_no_session(
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    store = ModelStore(ebm=multiclass_ebm)
    assert store.session is None
    for revision in ("original", "current"):
        read = store.read(revision)
        assert read.model is multiclass_ebm
        assert read.info == RevisionInfo("original", "original", 0)
    with pytest.raises(EditError, match="only revisions are 'original' and 'current', not 'abc'"):
        store.read("abc")
    with pytest.raises(UnsupportedModelError, match="no edit session"):
        store.editing()
    store.keep_description("original", "text")
    assert store.described("original") == "text"


def test_the_kept_description_belongs_to_the_revision_that_it_describes(continuous_ebm: FittedClassifier) -> None:
    """Contract: the store never serves the description of a model that is no longer the head."""
    ebm, _ = continuous_ebm
    store = ModelStore(ebm=ebm)
    assert store.described("original") is None
    store.keep_description("original", "words about the original")
    assert store.described("original") == "words about the original"
    store.editing().apply(SetScores("a", Bins(0, 1), 5.0), "Raise the lowest bins")
    edit = store.editing().revision
    assert store.described(edit) is None
    assert store.described("original") == "words about the original"
    store.keep_description("original", "a late text of the old head")
    assert store.described("original") == "words about the original"
    store.keep_description(edit, "words about the edit")
    assert store.described(edit) == "words about the edit"
    assert store.described("original") is None


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
