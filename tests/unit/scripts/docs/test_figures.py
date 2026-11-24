"""Tests for the script that makes the figures of the documentation from a fitted EBM."""

from pathlib import Path

import orjson
import pytest
from scripts.docs import figures

from ebaml import EBMGraph, compute_local_facts


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Write the figures once for the module, because the fit of the EBM takes a few seconds.

    Returns
    -------
    dict[str, Path]
        Each written file, by its name.

    """
    paths = figures.write_figures(tmp_path_factory.mktemp("generated"))
    return {path.name: path for path in paths}


def test_the_figures_are_the_files_that_the_pages_include(written: dict[str, Path]) -> None:
    assert sorted(written) == [
        "applicant.json",
        "counterfactual.json",
        "income-graph.svg",
        "income-graph.txt",
        "sufficient-reason.json",
    ]


def test_the_graph_image_is_a_valid_svg_with_a_title_that_says_the_data_is_synthetic(
    written: dict[str, Path],
) -> None:
    svg = written["income-graph.svg"].read_text(encoding="utf-8")
    assert svg.startswith('<?xml version="1.0"')
    assert svg.rstrip().endswith("</svg>")
    assert ">How income moves the loan decision (synthetic data)</text>" in svg


def test_the_graph_image_names_the_score_of_the_granted_class(written: dict[str, Path]) -> None:
    svg = written["income-graph.svg"].read_text(encoding="utf-8")
    assert f">change of the log-odds of '{figures.GRANTED}'</text>" in svg


def test_the_graph_text_reads_back_as_the_simplified_graph(written: dict[str, Path]) -> None:
    graph = EBMGraph.from_text(written["income-graph.txt"].read_text(encoding="utf-8"))
    assert graph.feature_name == figures.GRAPH_FEATURE
    assert 1 < len(graph.x_vals) < 40


def test_the_applicant_is_denied_with_a_counterfactual_of_one_change_and_a_sufficient_reason() -> None:
    data = figures.loan_data()
    applicant = data.features.row(figures.APPLICANT_ROW, named=True)
    facts = compute_local_facts(figures.fit_loan_model(data), applicant)
    assert facts.prediction.decision == figures.DENIED
    assert facts.counterfactuals
    assert len(facts.counterfactuals[0].changes) == 1
    assert facts.sufficient_reason is not None


def test_the_facts_files_are_the_json_that_the_llm_reads(written: dict[str, Path]) -> None:
    counterfactual = orjson.loads(written["counterfactual.json"].read_bytes())
    sufficient_reason = orjson.loads(written["sufficient-reason.json"].read_bytes())
    assert counterfactual["flip_possible"] is True
    assert sufficient_reason["exists"] is True


def test_the_data_has_both_decisions_and_stays_the_same_for_one_seed() -> None:
    first = figures.loan_data(row_count=200)
    assert set(first.target) == {figures.DENIED, figures.GRANTED}
    second = figures.loan_data(row_count=200)
    assert second.features.equals(first.features)
    assert second.target == first.target
