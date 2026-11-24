"""Make the figures of the documentation from a fitted EBM, so that a figure never drifts from the code.

``scripts/docs/build.py`` calls ``write_figures`` before each build. The files go to ``docs/generated/``, which is
gitignored. Zensical copies them to the site, and the pages include the text files with ``--8<--``. The README shows
the graph image through its address on the site, because PyPI shows the README without the repository.

The data is synthetic: a loan decision from income, debt ratio, age and late payments. Synthetic data needs no download,
and it gives an applicant whose local explanation is short. The figures say that the data is synthetic.
"""

from pathlib import Path

import attrs
import numpy as np
import orjson
import polars as pl

from ebaml import DataScientist, EBMGraph, GraphImageOptions, compute_local_facts, facts_to_texts, graph_svg, model_kind
from ebaml.core.ebm_classes import ExplainableBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "generated"
ROW_COUNT = 3000
SEED = 0
# the target labels sort with "granted" second, so a positive score of the binary EBM points toward a granted loan
DENIED = "denied"
GRANTED = "granted"
GRAPH_FEATURE = "income"
# a denied applicant whose counterfactual changes one feature and whose sufficient reason exists (checked by a test)
APPLICANT_ROW = 4
# the graph image says that the data is synthetic
GRAPH_TITLE = "How income moves the loan decision (synthetic data)"
# the graph text drops the bins that change the score by less than 5 % of its range, so the page shows a short text
SIMPLIFICATION = 0.05


@attrs.frozen
class LoanData:
    """The synthetic training data of the figures.

    Attributes
    ----------
    features: pl.DataFrame
        One column for each feature.
    target: list[str]
        ``denied`` or ``granted`` for each row.

    """

    features: pl.DataFrame
    target: list[str]


def loan_data(row_count: int = ROW_COUNT, seed: int = SEED) -> LoanData:
    """Make the synthetic loan data.

    Parameters
    ----------
    row_count: int
        The number of applicants.
    seed: int
        The seed of the random generator, so that the figures stay the same from build to build.

    Returns
    -------
    LoanData
        The features and the decisions.

    """
    random_generator = np.random.default_rng(seed)
    income = np.round(random_generator.lognormal(10.6, 0.45, row_count), -2)
    debt_ratio = np.round(random_generator.beta(2, 4, row_count), 2)
    age = random_generator.integers(18, 76, row_count).astype(float)
    late_payments = np.minimum(random_generator.poisson(0.7, row_count), 6).astype(float)
    # income helps up to a level, debt and late payments hurt, and age helps a little in the first years of credit
    logit = (
        1.6 * np.tanh((income - 38000) / 15000)
        - 5 * (debt_ratio - 0.35)
        - 0.8 * late_payments
        + 0.03 * np.minimum(age - 18, 25)
        - 0.5
    )
    granted = logit + random_generator.logistic(0, 1, row_count) > 0
    features = pl.DataFrame({
        "income": income,
        "debt_ratio": debt_ratio,
        "age": age,
        "late_payments": late_payments,
    })
    return LoanData(features=features, target=np.where(granted, GRANTED, DENIED).tolist())


def fit_loan_model(data: LoanData) -> ExplainableBoostingClassifier:
    """Fit the EBM of the figures, with no interactions, so that the local explanations accept it.

    Parameters
    ----------
    data: LoanData
        The training data.

    Returns
    -------
    ExplainableBoostingClassifier
        The fitted model.

    """
    # interpret ignores the column names of a polars frame
    model = ExplainableBoostingClassifier(interactions=0, random_state=SEED, feature_names=data.features.columns)
    return model.fit(data.features, data.target)


def write_figures(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Fit the loan model and write the figures of the documentation.

    Parameters
    ----------
    output_dir: Path
        The directory of the files. The function makes it if it does not exist.

    Returns
    -------
    list[Path]
        The files that the function wrote: the graph image, the graph text, the applicant and the facts of its local
        explanation.

    """
    data = loan_data()
    model = fit_loan_model(data)
    feature_index = data.features.columns.index(GRAPH_FEATURE)
    graph = EBMGraph.from_ebm(model, feature_index)
    applicant = data.features.row(APPLICANT_ROW, named=True)
    facts = facts_to_texts(compute_local_facts(model, applicant), DataScientist())
    contents = {
        "income-graph.svg": graph_svg(
            graph,
            GraphImageOptions(title=GRAPH_TITLE, score_axis=model_kind(model).score_axis),
        ),
        "income-graph.txt": graph.simplify(SIMPLIFICATION).to_text(),
        "applicant.json": orjson.dumps(applicant, option=orjson.OPT_INDENT_2).decode(),
        "counterfactual.json": facts.counterfactual,
        "sufficient-reason.json": facts.sufficient_reason,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in contents.items():
        path = output_dir / name
        path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        written.append(path)
    return written
