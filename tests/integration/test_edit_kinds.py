"""The path of an edit for every kind of model: edit, read, explain and save the edited model.

The README table "Supported models" promises that editing works for each kind of model. These tests read the kinds
from the row of the table, so a cell that the table marks as supported is a cell that a test runs.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import skops.io

from ebaml import (
    DescribeEBMRequest,
    EBMGraph,
    EditSession,
    ExplainInstanceRequest,
    ModelOptions,
    Monotonize,
    SetScores,
    UnseenBin,
    ZeroTerm,
    describe_ebm,
    edit_effect,
    explain_instance,
)
from ebaml.core.ebm_classes import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from tests.conftest import FakeLLM
from tests.docs.conftest import training_data
from tests.unit.ebaml.conftest import PandasFrame, as_frame

README = Path(__file__).parents[2] / "README.md"
EDITING_ROW = "| Editing (`EditSession`) |"
INSTANCE = {"age": 20.0, "fare": 50.0, "siblings": 1.0}
# the class that the edit of a multiclass model changes
EDITED_CLASS = "first"


def _supported_kinds() -> list[str]:
    """Read the kinds of model that the README table marks as supported for editing.

    Returns
    -------
    list[str]
        The names of the columns of kinds whose cell in the row of editing is a check mark.

    """
    lines = README.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("| Feature |"))
    row = next(line for line in lines if line.startswith(EDITING_ROW))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return [column for column, cell in zip(columns[1:], cells[1:], strict=True) if cell == "✅"]


def _fitted(kind: str) -> tuple[ExplainableBoostingClassifier | ExplainableBoostingRegressor, PandasFrame, np.ndarray]:
    features, target = training_data()
    random_generator = np.random.default_rng(3)
    noisy_fare = features["fare"].to_numpy() + random_generator.normal(0, 100, len(features))
    targets = {
        "Binary classifier": target.to_numpy(),
        "Multiclass classifier": np.where(noisy_fare > 350, "first", np.where(noisy_fare > 150, "second", "third")),
        "Regressor": noisy_fare,
    }
    model_class = ExplainableBoostingRegressor if kind == "Regressor" else ExplainableBoostingClassifier
    model = model_class(interactions=0, outer_bags=2, random_state=0).fit(features, targets[kind])
    return model, features, targets[kind]


def test_the_table_marks_every_kind_without_interactions_as_supported_for_editing() -> None:
    assert _supported_kinds() == ["Binary classifier", "Regressor", "Multiclass classifier"]


@pytest.fixture(scope="module", params=_supported_kinds())
def edited(request: pytest.FixtureRequest) -> tuple[EditSession, PandasFrame, np.ndarray]:
    """Fit a model of one kind of the table, and apply three edits to it.

    Returns
    -------
    tuple[EditSession, PandasFrame, np.ndarray]
        The session, the training features and the targets.

    """
    model, features, targets = _fitted(request.param)
    class_label = EDITED_CLASS if request.param == "Multiclass classifier" else None
    session = EditSession(model)
    session.apply(Monotonize("fare", "increasing", class_label=class_label), "The fare must not lower the outcome")
    session.apply(SetScores("age", UnseenBin(), 0.0), "An unseen age scores 0")
    session.apply(ZeroTerm("siblings"), "The outcome must not depend on the siblings")
    return session, features, targets


def test_the_edited_model_gives_the_graph_text_of_each_feature(
    edited: tuple[EditSession, PandasFrame, np.ndarray],
) -> None:
    session, _, _ = edited
    text = EBMGraph.from_ebm(session.model, 1).to_text()
    assert "fare" in text


def test_the_edited_model_is_described_and_explained_through_the_llm(
    edited: tuple[EditSession, PandasFrame, np.ndarray],
    llm_model: ModelOptions,
    fake_llm: FakeLLM,
) -> None:
    session, _, _ = edited
    summary = describe_ebm(DescribeEBMRequest(session.model, model=llm_model, num_sentences=3))
    explanation = explain_instance(ExplainInstanceRequest(session.model, INSTANCE, summary.text, model=llm_model))
    assert explanation.counterfactual
    assert len(fake_llm.requests) > 0


def test_the_effect_of_the_edits_has_the_metrics_of_the_kind(
    edited: tuple[EditSession, PandasFrame, np.ndarray],
) -> None:
    session, features, targets = edited
    effect = edit_effect(session, as_frame(features), targets)
    every_row = effect.scopes[0]
    assert every_row.row_count == len(features)
    assert every_row.current is not None
    assert every_row.current != every_row.original


def test_the_edited_model_survives_a_save_and_a_load(
    edited: tuple[EditSession, PandasFrame, np.ndarray],
    tmp_path: Path,
) -> None:
    session, features, _ = edited
    saved = session.save(tmp_path / "model.skops", "Integration test", allow_unreviewed=True)
    loaded = skops.io.load(
        tmp_path / "model.skops",
        trusted=skops.io.get_untrusted_types(file=tmp_path / "model.skops"),
    )
    assert np.array_equal(loaded.predict(features), session.model.predict(features))
    assert re.fullmatch(r"[0-9a-f]+", saved.model_hash)
