"""Fail when the CI report leaves out a job that runs on a pull request.

``.github/scripts/ci_report.py`` writes the progress comment and the final report of a pull request from a list of
gates (``GATES``). A job of any workflow that runs on ``pull_request`` is either a gate of that list, or it is on the
list ``NOT_GATES`` below with the reason. This check fails when:

- a job of a workflow that runs on ``pull_request`` is neither a gate nor in ``NOT_GATES``,
- a gate or an entry of ``NOT_GATES`` is not the name of a job that runs on ``pull_request``,
- a gate waits for a job that runs only for some paths (``runs`` is not ``some``), or a path-filtered job is a gate
  that the report waits for,
- a required status check of the ruleset of ``main`` is not a gate.

Run it from the repo root: ``uv run python -m scripts.ci.report_gates``.
"""

import importlib.util
import sys
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson
import yaml

from scripts.ci.ruleset import PATH_FILTERS, job_names, pull_request_trigger, required_checks

ROOT = Path(__file__).resolve().parents[2]
RULESET = ROOT / ".github" / "rulesets" / "main.json"
WORKFLOWS = ROOT / ".github" / "workflows"
REPORT_SCRIPT = ROOT / ".github" / "scripts" / "ci_report.py"

# Jobs of the workflows that run on a pull request and that the report does not show, with the reason.
NOT_GATES = {
    "changes": "decides which gates of ci.yml run, and has no check of its own to pass",
    "docs changes": "decides whether the docs build runs, and has no check of its own to pass",
    "mutant plan": "splits the mutation run into parts, and the gate `mutation` sums up the result",
    "mutant part ${{ matrix.part }}": "one part of the mutation run, and the gate `mutation` sums up the parts",
    "progress": "writes the progress comment, so it is not a result",
    "report": "writes the final report, so it is not a result",
    "deploy": "publishes the site after a merge, so it never runs on a pull request",
}


def load_report_script() -> ModuleType:
    """Load ``ci_report.py`` by its path, because ``.github/scripts`` is not a package.

    Returns
    -------
    ModuleType
        The loaded module.

    Raises
    ------
    ImportError
        If the file has no loader.

    """
    spec = importlib.util.spec_from_file_location("ci_report", REPORT_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover -- a readable .py file always has a loader
        msg = f"cannot load {REPORT_SCRIPT}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_problems(
    gates: Iterable[tuple[str, str, str]],
    required: Iterable[str],
    workflows: Mapping[str, Mapping[Any, Any]],
    not_gates: Collection[str] = tuple(NOT_GATES),
) -> list[str]:
    """List the jobs that the CI report does not account for.

    Parameters
    ----------
    gates: Iterable[tuple[str, str, str]]
        The gates of the report, each as ``(id, check, runs)``.
    required: Iterable[str]
        The names of the required status checks of the ruleset.
    workflows: Mapping[str, Mapping[Any, Any]]
        The parsed workflows, by file name.
    not_gates: Collection[str]
        The names of the jobs that are no gate.

    Returns
    -------
    list[str]
        One message for each problem.

    """
    gate_list = list(gates)
    runs_of = {check: runs for _, check, runs in gate_list}
    pull_request_jobs: dict[str, bool] = {}  # job name -> whether its workflow filters paths
    for workflow in workflows.values():
        trigger = pull_request_trigger(workflow)
        if trigger is not None:
            filtered = any(key in trigger for key in PATH_FILTERS)
            pull_request_jobs.update(dict.fromkeys(job_names(workflow), filtered))
    problems = []
    for name, filtered in pull_request_jobs.items():
        if name not in runs_of and name not in not_gates:
            problems.append(f"job '{name}' runs on pull_request but is no gate of ci_report.py and not in NOT_GATES")
        elif name in runs_of and name in not_gates:
            problems.append(f"job '{name}' is a gate and is also in NOT_GATES")
        elif name in runs_of and filtered != (runs_of[name] == "some"):
            problems.append(f"gate '{name}' must have runs='some' exactly when its workflow filters paths")
    problems += [
        f"gate '{check}' is not the name of a job that runs on pull_request"
        for _, check, _ in gate_list
        if check not in pull_request_jobs
    ]
    problems += [
        f"NOT_GATES entry '{name}' is not the name of a job that runs on pull_request"
        for name in not_gates
        if name not in pull_request_jobs
    ]
    problems += [f"required check '{name}' is not a gate of ci_report.py" for name in required if name not in runs_of]
    return problems


def main() -> int:
    """Report every job that the CI report does not account for.

    Returns
    -------
    int
        The process exit code: 0 if the report covers every job, 1 otherwise.

    """
    ruleset = orjson.loads(RULESET.read_bytes())
    workflows = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(WORKFLOWS.glob("*.yml"))
    }
    gates = [(gate.id, gate.check, gate.runs) for gate in load_report_script().GATES]
    problems = report_problems(gates, required_checks(ruleset), workflows)
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
