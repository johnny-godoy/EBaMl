"""The workloads of the performance benchmark: what one round of each measurement runs.

A workload names a model, a call of EBaMl and its options. Its ``prepare`` function does the work that the rounds
must not measure (load the model, set the replies of the fake LLM) and returns the round. The LLM calls go to the local
fake LLM server of the integration tests, so the benchmark never sends a real request.

The table ``WORKLOADS`` lists every workload. A new workload is a new row, or a new module of rows.
``benchmark/README.md`` lists the groups.
"""

import sys
from collections.abc import Callable, Mapping, Sequence

import narwhals as nw
from scripts.fixed_command import run_fixed

from benchmark.performance.llm_workloads import llm_workloads
from benchmark.performance.mcp_workloads import mcp_workloads
from benchmark.performance.workload_types import Context, Round, Workload
from ebaml import (
    EBMGraph,
    EBMGraphSimplificationError,
    EBMGraphTokenLimitError,
    EditEffect,
    EditSession,
    Monotonize,
    compute_local_facts,
    edit_effect,
)
from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.graphs.token_budget import check_token_budget

# the models of each group of workloads
GRAPH_MODELS: Sequence[str] = (
    "titanic",
    "diabetes",
    "multiclass",
    "categorical",
    "features-10-bins-32",
    "features-10-bins-1024",
    "features-100-bins-32",
    "features-100-bins-1024",
    "features-500-bins-32",
    "features-500-bins-1024",
)
LOCAL_MODELS: Sequence[str] = ("titanic", "diabetes", "multiclass", "categorical", "features-100-bins-1024")
# each model of the edit workloads, with the continuous feature that the edit makes increasing
EDIT_MODELS: Mapping[str, str] = {
    "titanic": "age",
    "diabetes": "bmi",
    "features-100-bins-1024": "x000",
    "features-500-bins-1024": "x000",
}


def _import(statement: str) -> Callable[[Context], Round]:
    """Measure a fresh interpreter that runs one statement, for the import time of a module.

    Parameters
    ----------
    statement: str
        The Python statement, for example ``import ebaml``.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload. The time includes the start of the interpreter.

    """

    def prepare(_context: Context) -> Round:
        return Round(lambda: run_fixed([sys.executable, "-c", statement], check=True))

    return prepare


def _render_or_fail(ebm: ExplainableBoostingModel, feature_index: int) -> str | None:
    """Render the graph text of one feature, or fail as ``describe_ebm`` does for a graph over the token budget.

    The workload calls the public ``EBMGraph`` with the budget check of ``describe_ebm``, so it runs on any version of
    ``ebaml``: the base of a pull request runs the workloads of its head. The defaults of ``EBMGraph.to_text`` are the
    defaults of ``GraphOptions``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.
    feature_index: int
        The index of the feature.

    Returns
    -------
    str | None
        The text, or ``None`` for a graph that no simplification brings under the token budget. ``describe_ebm``
        reports such a graph as a failure of its feature, after the same work.

    """
    try:
        return EBMGraph.from_ebm(ebm, feature_index).to_text(check_budget=check_token_budget)
    except (EBMGraphSimplificationError, EBMGraphTokenLimitError):
        return None


def _graphs(model: str) -> Callable[[Context], Round]:
    """Measure the graph text of every feature of a model, as ``describe_ebm`` renders it before the requests.

    Parameters
    ----------
    model: str
        The name of the model.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload.

    """

    def prepare(context: Context) -> Round:
        ebm = context.model(model).ebm
        feature_count = len(ebm.feature_names_in_)
        return Round(lambda: [_render_or_fail(ebm, feature_index) for feature_index in range(feature_count)])

    return prepare


def _local_facts(model: str) -> Callable[[Context], Round]:
    """Measure the facts of the local explanation of the first row of a model.

    Parameters
    ----------
    model: str
        The name of the model.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload, with the default options: the MIP search and 10,000 samples.

    """

    def prepare(context: Context) -> Round:
        fitted = context.model(model)
        instance = fitted.instance()
        return Round(lambda: compute_local_facts(fitted.ebm, instance))

    return prepare


def _edit(model: str, feature: str) -> Callable[[Context], Round]:
    """Measure one edit and its effect on the training data.

    Parameters
    ----------
    model: str
        The name of the model.
    feature: str
        The continuous feature that the edit makes increasing.

    Returns
    -------
    Callable[[Context], Round]
        The ``prepare`` function of the workload. A round starts a session, applies the edit and measures its effect.

    """

    def prepare(context: Context) -> Round:
        fitted = context.model(model)
        data = nw.from_native(fitted.data.features)

        def run() -> EditEffect:
            session = EditSession(fitted.ebm)
            session.apply(Monotonize(feature, "increasing"), "The edit of the benchmark")
            return edit_effect(session, data, fitted.data.target)

        return Round(run)

    return prepare


def _workloads() -> list[Workload]:
    """Make every workload of the benchmark.

    Returns
    -------
    list[Workload]
        The workloads, in the order of the run.

    """
    return [
        Workload("import.ebaml", "import", _import("import ebaml")),
        Workload("import.ebaml_llm", "import", _import("import ebaml; ebaml.describe_ebm")),
        *(Workload(f"graphs.render.{model}", "graphs", _graphs(model)) for model in GRAPH_MODELS),
        *(Workload(f"local.facts.{model}", "local", _local_facts(model)) for model in LOCAL_MODELS),
        *(
            Workload(f"edit.monotonize.{model}", "edit", _edit(model, feature))
            for model, feature in EDIT_MODELS.items()
        ),
        *llm_workloads(),
        *mcp_workloads(),
    ]


WORKLOADS: Sequence[Workload] = _workloads()
