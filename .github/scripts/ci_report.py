#!/usr/bin/env python3
"""Turn raw CI tool output into short, actionable failure reports.

Four subcommands, all stdlib only:

``snippet <gate>``
    Run inside a failed job. Reads that gate's logs from ``$CI_LOGS`` and writes ``$CI_REPORT/<gate>.md`` holding
    exactly what failed and how to fix it. Also emits ``::error`` annotations so findings show on the diff.

``assemble``
    Run in the final job. Waits until every gate has finished, then combines the snippets with the check runs of the
    commit into one markdown report: written to ``$CI_REPORT/comment.md`` and the step summary.

``watch``
    Run in the ``progress`` job, next to the gates. Every ``POLL_SECONDS`` seconds it reads the check runs of the commit
    from the GitHub API and updates the PR comment ("CI in progress"), until all gates have finished. When a gate
    fails, the comment shows the failure report of that gate at once. The report is the artifact that the failed job
    uploads. A gate of another workflow has no such artifact, so the comment links to its job log.

The gates cover every workflow that runs on a pull request (``GATES``). ``scripts/check_report_gates.py`` fails when a
job of such a workflow is neither a gate nor on its list of jobs that are no gate.

``post <file>``
    Create the PR comment from a file, or replace it if it exists. The ``report`` job uses it for the final report.
    ``watch`` and ``post`` find the comment by ``MARKER``, so one comment changes during the whole run. Without write
    access (a fork PR) they do nothing.
"""

import io
import json  # ruff: ignore[banned-api] -- this script runs under the bare python3 of the runner, with no project environment
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

LOGS: Path = Path(os.environ.get("CI_LOGS", Path(tempfile.gettempdir()) / "ci-logs"))
REPORT: Path = Path(os.environ.get("CI_REPORT", Path(tempfile.gettempdir()) / "ci-report"))
MAX_CHARS = 6000
MAX_LOG_TAIL = 3000
ERROR_CONTEXT_BEFORE = 25
ERROR_CONTEXT_AFTER = 10
LOG_NOISE: re.Pattern[str] = re.compile(
    r"^\d{4}-\d\d-\d\dT[\d:.]+Z |\x1b\[[\d;]*m",
)  # the timestamp that Actions adds, and ANSI colors
PERCENT = 100
SNIPPET_ARGC = 3
POST_ARGC = 3
POLL_SECONDS = 20
WATCH_LIMIT_SECONDS = 70 * 60
PAGE_SIZE = 100
FORBIDDEN = (401, 403)
REDIRECT = 302
MARKER = "<!-- ci-report -->"
SETTLE_SECONDS = 15 * 60
FINISHED = frozenset({"success", "failure", "cancelled", "skipped", "absent"})
CONCLUSIONS = {
    "success": "success",
    "failure": "failure",
    "timed_out": "failure",
    "action_required": "failure",
    "startup_failure": "failure",
    "cancelled": "cancelled",
    "stale": "cancelled",
    "skipped": "skipped",
    "neutral": "skipped",
}


class Gate(NamedTuple):
    """One check that the report covers.

    ``check`` is the name of the job, as GitHub names its check run. ``runs`` tells when the job exists:
    ``always`` (every pull request and every push), ``pull_request`` (every pull request) or ``some`` (only when the
    changed paths match the ``paths:`` filter of its workflow, so the report cannot wait for it).
    """

    id: str
    check: str
    title: str
    runs: str


# In fail-fast order. The gates of ci.yml come first: each has a builder in ``BUILDERS`` and a failure report. The
# gates of the other workflows have none, so the report links to their job logs.
GATES = [
    Gate("prek", "prek", "prek hooks", "always"),
    Gate("trufflehog", "trufflehog", "trufflehog (secrets)", "always"),
    Gate("semgrep", "semgrep", "semgrep (base rules)", "always"),
    Gate("trivy", "trivy", "trivy (vulnerabilities, secrets, misconfig)", "always"),
    Gate("baml", "baml check + test", "BAML fmt, check, test", "always"),
    Gate("pytest", "pytest (100% coverage)", "pytest + 100% coverage", "always"),
    Gate("mutation", "mutation (changed modules, 100% score)", "mutation score 100%", "always"),
    Gate("docs", "docs build", "docs build", "always"),
    Gate("fragment", "fragment", "changelog fragment", "pull_request"),
    Gate("title", "title", "pull request title", "pull_request"),
    Gate("commits", "commits", "commit messages", "pull_request"),
    Gate("prompts", "live prompt tests", "live prompt tests", "some"),
    Gate("proofs", "lake build", "Lean proofs", "some"),
]

PREK_HINTS = {
    "ruff-format": "Formatter rewrote files. Commit the changes: `uvx prek run ruff-format --all-files`.",
    "ruff-check": "Fix the lint errors (autofixes were applied, commit them): `uvx prek run ruff-check --all-files`.",
    "vulture": "Dead code. Delete it, or use it. Check locally: `uv --directory src run vulture`.",
    "pyrefly": "Type errors. Fix them. Check locally: `baml generate && uv --directory src run pyrefly check src`.",
    "pyrefly-coverage": (
        "A name has no type annotation. Annotate it. List them: "
        "`uv run pyrefly coverage check src/ebaml tests scripts .github/scripts`."
    ),
    "actionlint": (
        "Workflow syntax error. Fix the file named above. Check locally: `uvx prek run actionlint --all-files`."
    ),
}
PREK_DEFAULT_HINT = (
    "Reproduce locally with `uvx prek run --all-files`. Hooks that modify files: commit what they changed."
)


def read(path: Path) -> str:
    """Read a log file, tolerating a missing file and undecodable bytes.

    Parameters
    ----------
    path: Path
        The file to read.

    Returns
    -------
    str
        The contents of the file, or an empty string if it does not exist.

    """
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def clip(text: str, limit: int = MAX_CHARS) -> str:
    """Truncate a text to a number of characters.

    Parameters
    ----------
    text: str
        The text to truncate.
    limit: int
        The maximum number of characters kept.

    Returns
    -------
    str
        The text, with a note on the dropped characters if it was truncated.

    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... [{len(text) - limit} more characters, see the job log]"


def block(text: str) -> str:
    """Wrap a text in a markdown code fence.

    Parameters
    ----------
    text: str
        The text to wrap.

    Returns
    -------
    str
        The fenced, clipped text.

    """
    return f"```\n{clip(text)}\n```"


def annotate(file: str, line: int | str, message: str) -> None:
    """Emit a GitHub ``::error`` annotation, so that a finding shows on the diff.

    Parameters
    ----------
    file: str
        The file of the finding.
    line: int | str
        The line of the finding.
    message: str
        The text of the annotation.

    """
    message = message.replace("\n", " ").replace("%", "%25")
    safe_file = file if re.fullmatch(r"[\w./-]+", file) else "?"
    safe_line = int(line) if str(line).isdigit() else 1
    sys.stdout.write(f"::error file={safe_file},line={safe_line}::{message}\n")


def prek() -> str:
    """Report the prek hooks that failed.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    log = read(LOGS / "prek.log")
    failed: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in log.splitlines():
        status = re.match(r"^(.+?)\.{3,}\s*(Passed|Failed|Skipped)", line)
        if status:
            current = [] if status.group(2) == "Failed" else None
            if current is not None:
                failed.append((status.group(1).strip(), current))
        elif current is not None:
            current.append(line)
    if not failed:
        return f"No hook result was parsed. Last output:\n{block(log[-MAX_LOG_TAIL:])}\n\n**Fix:** {PREK_DEFAULT_HINT}"
    parts = []
    for name, output in failed:
        hook_id = next((m.group(1) for ln in output if (m := re.match(r"^- hook id: (\S+)", ln))), name)
        hint = PREK_HINTS.get(hook_id, PREK_DEFAULT_HINT)
        parts.append(f"**Hook `{name}` failed**\n{block(chr(10).join(output))}\n**Fix:** {hint}")
    return "\n\n".join(parts)


def trufflehog() -> str:
    """Report the secrets that trufflehog found.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    findings = []
    for line in read(LOGS / "trufflehog.json").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        git = item.get("SourceMetadata", {}).get("Data", {}).get("Git", {})
        if "DetectorName" not in item:
            continue
        findings.append((
            item["DetectorName"],
            item.get("Verified"),
            git.get("file", "?"),
            git.get("line", 0),
            git.get("commit", "?")[:10],
        ))
    if not findings:
        log = read(LOGS / "trufflehog.err")[-MAX_LOG_TAIL:]
        return f"trufflehog exited non-zero but reported no findings. Its log:\n{block(log)}"
    rows = ["| Detector | Verified | File | Line | Commit |", "|---|---|---|---|---|"]
    for detector, verified, file, line, commit in findings:
        rows.append(f"| {detector} | {'yes (live!)' if verified else 'unverified'} | `{file}` | {line} | `{commit}` |")
        annotate(file, line or 1, "trufflehog: possible secret, see the job summary")
    fix = (
        "**Fix:**\n"
        "1. Treat every verified hit as leaked: **rotate the credential now**."
        " Removing it from git does not un-leak it.\n"
        "2. Remove the secret from the code and load it from the environment or a secret store.\n"
        "3. The secret lives in git history, so a new commit is not enough. Rewrite the branch "
        "(`git rebase -i` / `git filter-repo`) before it is merged.\n"
        "4. False positive? Add `trufflehog:ignore` on the line."
    )
    return "\n".join(rows) + "\n\n" + fix


def semgrep() -> str:
    """Report the findings of semgrep.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    try:
        data = json.loads(read(LOGS / "semgrep.json"))
    except json.JSONDecodeError:
        return f"semgrep produced no report. Its log:\n{block(read(LOGS / 'semgrep.log')[-MAX_LOG_TAIL:])}"
    rows = ["| Rule | Location | Message |", "|---|---|---|"]
    for result in data.get("results", []):
        path, line = result["path"], result["start"]["line"]
        message = result["extra"]["message"].strip().splitlines()[0]
        rule = result["check_id"].rsplit(".", 1)[-1]
        rows.append(f"| `{rule}` | `{path}:{line}` | {message} |")
        annotate(path, line, f"semgrep {result['check_id']}: {message}")
    rows.extend(f"| error | | {err.get('message', err)} |" for err in data.get("errors", []))
    fix = (
        "**Fix:** change the flagged code as the message says."
        " Full rule text: `uvx semgrep scan --config p/default`.\n"
        "False positive? Add `# nosemgrep: <rule id>` on the line with a short reason."
    )
    return "\n".join(rows) + "\n\n" + fix


def trivy() -> str:
    """Report the vulnerabilities, secrets and misconfigurations that trivy found.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    try:
        data = json.loads(read(LOGS / "trivy.json"))
    except json.JSONDecodeError:
        return f"trivy produced no report. Its log:\n{block(read(LOGS / 'trivy.log')[-MAX_LOG_TAIL:])}"
    rows = ["| Kind | Target | Finding | Fix |", "|---|---|---|---|"]
    bumps: set[str] = set()
    for result in data.get("Results") or []:
        target = result["Target"]
        for vuln in result.get("Vulnerabilities") or []:
            fixed = vuln.get("FixedVersion", "")
            rows.append(
                f"| {vuln['Severity']} vuln | `{target}` | {vuln['VulnerabilityID']} in "
                f"`{vuln['PkgName']} {vuln['InstalledVersion']}` | upgrade to {fixed or 'n/a'} |",
            )
            if fixed:
                bumps.add(vuln["PkgName"])
        for secret in result.get("Secrets") or []:
            rows.append(
                f"| secret | `{target}:{secret.get('StartLine', '?')}` | {secret['Title']} | remove and rotate |",
            )
            annotate(target, secret.get("StartLine", 1), "trivy: possible secret, see the job summary")
        rows.extend(
            f"| {misconf['Severity']} misconfig | `{target}` | {misconf['ID']}: {misconf['Title']} |"
            f" {misconf.get('Resolution', '')} |"
            for misconf in result.get("Misconfigurations") or []
        )
    fix = ["**Fix:**"]
    if bumps:
        upgrades = " ".join(f"--upgrade-package {name}" for name in sorted(bumps))
        fix.append(
            f"- Vulnerable dependencies: `uv lock {upgrades}`, then `uv sync` and rerun the tests. Commit `uv.lock`.",
        )
    fix.extend((
        "- Secrets: remove, rotate the credential, load it from the environment.",
        "- Misconfigurations: apply the resolution in the table.",
        "- Not exploitable here? Add the ID to `.trivyignore` with a comment saying why.",
    ))
    return clip("\n".join(rows), MAX_CHARS) + "\n\n" + "\n".join(fix)


def baml() -> str:
    """Report the BAML steps (fmt, check, test) that failed.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    failed = read(LOGS / "baml.failed").split()
    if not failed:
        setup_log = read(LOGS / "baml-setup.log")[-MAX_LOG_TAIL:]
        return f"BAML job failed outside fmt/check/test (toolchain install?). Last output:\n{block(setup_log)}"
    hints = {
        "fmt": "The `.baml` files are not formatted. Run `baml fmt` and commit the result.",
        "check": "Compile error. Fix the file and line named above. Check locally: `baml check`.",
        "test": (
            "A BAML test failed. Reproduce with `OPENROUTER_API_KEY=unused baml test -x root::Live`; run one with "
            "`baml test -i <name>`. Update the code or the test's expectation."
        ),
    }
    parts = [
        f"**`baml {step}` failed**\n{block(read(LOGS / f'baml-{step}.log'))}\n**Fix:** {hints[step]}" for step in failed
    ]
    return "\n\n".join(parts)


def pytest() -> str:
    """Report the failing tests and the files that lack coverage.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    log = read(LOGS / "pytest.log")
    parts = []
    failures: list[str] = re.findall(r"^(?:FAILED|ERROR) .+$", log, re.MULTILINE)
    if failures:
        parts.append(
            "**Failing tests**\n"
            + block("\n".join(failures))
            + "\n**Fix:** reproduce with `uv run pytest -x <test id>`. "
            "The traceback is in the job log under the `FAILURES` heading.",
        )
        for failure in failures:
            if m := re.match(r"^(?:FAILED|ERROR) (\S+?\.py)::", failure):
                annotate(m.group(1), 1, failure)
    short = []
    for m in re.finditer(r"^(\S+\.py)\s+\d+\s+(\d+)\s+\d+\s+(\d+)\s+(\d+)%\s*(.*)$", log, re.MULTILINE):
        path, missed, partial, cover, missing = m.groups()
        if int(cover) < PERCENT:
            short.append(
                f"{path}: {cover}% | missed statements: {missed}, partial branches: {partial}"
                f" | lines: {missing or '-'}",
            )
    if short or "Required test coverage" in log:
        total = re.search(r"^TOTAL\s.*?(\d+)%", log, re.MULTILINE)
        head = f"**Coverage {total.group(1)}%, 100% required**\n" if total else "**Coverage below 100%**\n"
        parts.append(
            head
            + block("\n".join(short) or "see the coverage table in the job log")
            + "\n**Fix:** write tests that execute "
            "the listed lines and branches (`a->b` means the branch from line a to b is never taken). Check locally: "
            "`uv run pytest` and read the `Missing` column. Do not add `# pragma: no cover` to dodge this.",
        )
    if not parts:
        parts.append(f"pytest failed without a recognized summary. Last output:\n{block(log[-MAX_LOG_TAIL:])}")
    return "\n\n".join(parts)


def mutation() -> str:
    """Report the mutants that survived.

    Returns
    -------
    str
        The markdown with the failure and how to fix it.

    """
    log = read(LOGS / "mutation.log")
    survivors = read(LOGS / "mutation-survivors.txt")
    if "Baseline failed" in log:
        return (
            f"**Baseline failed: the test suite fails or times out with no mutation applied**\n{block(log[-2500:])}\n"
            "**Fix:** make `uv run pytest -x -o addopts='' tests/unit` pass on its own first. If it only times out, "
            "raise `timeout` in `cosmic-ray.toml`; CI runners are slower than laptops."
        )
    unfinished = re.search(r"mutation result\s+survived=\d+ total=(\d+) unfinished=([1-9]\d*)", log)
    if unfinished and not survivors:
        return (
            f"**The time limit stopped the mutation run: {unfinished.group(2)} of {unfinished.group(1)} mutants of "
            f"this part did not run**\n{block(log[-MAX_LOG_TAIL:])}\n"
            "**Fix:** the run has no survivor so far. Make the changed lines smaller, or split the pull request: "
            "each part of the run has 50 minutes."
        )
    head = "**Surviving mutants**\n"
    if unfinished:
        head += f"The time limit also stopped the run: {unfinished.group(2)} mutants of this part did not run.\n"
    return (
        head
        + block(survivors or log[-MAX_LOG_TAIL:])
        + "\n**Fix:** each diff above is a bug the tests did not notice: the `-` line is the real code, "
        "the `+` line the mutant. Add a test that fails when the code is changed that way "
        "(assert on the exact value or boundary). "
        "Equivalent mutant (no test can tell the difference)? Simplify the code so it does not exist. "
        "Reproduce locally: `uv run python scripts/mutation.py`."
    )


BUILDERS = {fn.__name__: fn for fn in (prek, trufflehog, semgrep, trivy, baml, pytest, mutation)}


def snippet(gate: str) -> None:
    """Write the report of a failed gate to ``$CI_REPORT/<gate>.md``.

    Parameters
    ----------
    gate: str
        The job id of the gate.

    """
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f"{gate}.md").write_text(BUILDERS[gate]() + "\n")


def run_url() -> str:
    """Build the URL of the workflow run.

    Returns
    -------
    str
        The URL of the run, from the variables of the runner.

    """
    server, repository, run_id = (os.environ[k] for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repository}/actions/runs/{run_id}"


def api(method: str, path: str, payload: Mapping[str, object] | None = None) -> Any:  # ruff: ignore[any-type] -- the GitHub API returns JSON of many shapes
    """Call the GitHub REST API with the token of the job.

    Parameters
    ----------
    method: str
        The HTTP method.
    path: str
        The path of the endpoint, with its query.
    payload: Mapping[str, object], optional
        The JSON body.

    Returns
    -------
    Any
        The decoded JSON of the response.

    Raises
    ------
    ValueError
        If ``GITHUB_API_URL`` is not an http or https URL.

    """
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    if not base.startswith(("https://", "http://")):  # urllib also opens file: URLs, so allow only web schemes
        msg = f"GITHUB_API_URL must be an http(s) URL: {base}"
        raise ValueError(msg)
    request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage] -- the scheme of the URL is checked above
        f"{base}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    # The scheme of the URL is checked above, and the host comes from the runner.
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # ruff: ignore[commented-out-code] -- semgrep marker, not code
    with urllib.request.urlopen(request, timeout=30) as response:  # ruff: ignore[suspicious-url-open-usage] -- the scheme of the URL is checked above
        return json.load(response)


def fetch_archive(url: str) -> bytes:
    """Download an artifact archive. The store behind the redirect rejects the token, so follow the redirect by hand.

    Parameters
    ----------
    url : str
        The ``archive_download_url`` of the artifact.

    Returns
    -------
    bytes
        The zip archive.

    Raises
    ------
    urllib.error.HTTPError
        If the API answers with an error other than the redirect to the store.

    """
    request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage] -- the URL comes from the GitHub API
        url,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "X-GitHub-Api-Version": "2022-11-28"},
    )
    opener = urllib.request.OpenerDirector()  # no redirect handler: the 302 then raises, and holds the final URL
    for handler in (urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler()):
        opener.add_handler(handler)
    try:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # ruff: ignore[commented-out-code] -- semgrep marker, not code
        return bytes(opener.open(request, timeout=30).read())
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location")
        if error.code != REDIRECT or not location or not location.startswith("https://"):
            raise
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # ruff: ignore[commented-out-code] -- semgrep marker, not code
    with urllib.request.urlopen(location, timeout=30) as response:  # ruff: ignore[suspicious-url-open-usage] -- the scheme is checked above
        return bytes(response.read())


def download_report(gate: str) -> str:
    """Read the failure report that the failed job of a gate uploaded as an artifact.

    Parameters
    ----------
    gate: str
        The job id of the gate.

    Returns
    -------
    str
        The markdown of the report, or an empty string if the artifact does not exist yet.

    """
    repository, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    listing = api("GET", f"/repos/{repository}/actions/runs/{run_id}/artifacts?per_page={PAGE_SIZE}")
    # a gate that runs in parallel parts uploads one artifact for each failed part: `report-<gate>-<part>`
    names = (f"report-{gate}", f"report-{gate}-")
    artifacts = [
        artifact
        for artifact in listing["artifacts"]
        if not artifact["expired"] and (artifact["name"] == names[0] or artifact["name"].startswith(names[1]))
    ]
    texts: list[str] = []
    for artifact in artifacts:
        with zipfile.ZipFile(io.BytesIO(fetch_archive(artifact["archive_download_url"]))) as archive:
            texts += [archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist()]
    return "\n".join(texts)


def pull_request_number() -> int:
    """Read the number of the pull request from the event of the run.

    Returns
    -------
    int
        The number of the pull request.

    """
    return int(json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))["pull_request"]["number"])


def upsert_comment(text: str) -> None:
    """Create the PR comment, or replace the text of the comment that has ``MARKER``.

    Parameters
    ----------
    text: str
        The markdown of the comment, without the marker.

    """
    repository = os.environ["GITHUB_REPOSITORY"]
    number = pull_request_number()
    body = f"{MARKER}\n{text}"
    page = 1
    while True:
        comments = api("GET", f"/repos/{repository}/issues/{number}/comments?per_page={PAGE_SIZE}&page={page}")
        for comment in comments:
            if MARKER in comment["body"]:
                api("PATCH", f"/repos/{repository}/issues/comments/{comment['id']}", {"body": body})
                return
        if len(comments) < PAGE_SIZE:
            break
        page += 1
    api("POST", f"/repos/{repository}/issues/{number}/comments", {"body": body})


def head_sha() -> str:
    """Read the commit that the checks of the run belong to.

    Returns
    -------
    str
        The head commit of the pull request, or the commit of the push.

    """
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    return str(event["pull_request"]["head"]["sha"]) if "pull_request" in event else os.environ["GITHUB_SHA"]


def check_runs() -> list[Mapping[str, Any]]:
    """Read the latest check run of each job of every workflow that ran on the commit.

    Returns
    -------
    list[Mapping[str, Any]]
        The check runs, as the GitHub API returns them.

    """
    repository = os.environ["GITHUB_REPOSITORY"]
    path = f"/repos/{repository}/commits/{head_sha()}/check-runs?filter=latest&per_page={PAGE_SIZE}"
    return list(api("GET", path)["check_runs"])


def gate_states(runs: Sequence[Mapping[str, str | None]], event_name: str) -> dict[str, str]:
    """Find the state of each gate in the check runs of the commit.

    Parameters
    ----------
    runs: Sequence[Mapping[str, str | None]]
        The check runs, from ``check_runs``.
    event_name: str
        The event of the run (``pull_request`` or ``push``). A gate that the event never starts has the state
        ``absent``, so the report does not wait for it.

    Returns
    -------
    dict[str, str]
        The state of each gate: ``absent``, ``waiting``, ``running``, or the conclusion of the check run.

    """
    by_name = {run["name"]: run for run in runs}
    states: dict[str, str] = {}
    for gate in GATES:
        expected = gate.runs == "always" or (gate.runs == "pull_request" and event_name == "pull_request")
        run = by_name.get(gate.check)
        if run is None:
            states[gate.id] = "waiting" if expected else "absent"
        elif run["status"] == "completed":
            states[gate.id] = CONCLUSIONS.get(run["conclusion"] or "", "cancelled")
        else:
            states[gate.id] = "running" if run["status"] == "in_progress" else "waiting"
    return states


def log_links(runs: Sequence[Mapping[str, str | None]]) -> dict[str, str]:
    """Find the job log of each gate that has a check run.

    Parameters
    ----------
    runs: Sequence[Mapping[str, str | None]]
        The check runs, from ``check_runs``.

    Returns
    -------
    dict[str, str]
        The URL of the check run of each gate, by gate id.

    """
    by_name = {run["name"]: run for run in runs}
    return {gate.id: by_name[gate.check]["html_url"] or "" for gate in GATES if gate.check in by_name}


def job_log_excerpt(url: str) -> str:
    """Read the lines around the first error of a job log.

    Parameters
    ----------
    url : str
        The URL of the check run of an Actions job (``.../job/<id>``).

    Returns
    -------
    str
        The lines before and after the first ``##[error]`` line, or an empty string if the log is not available.

    """
    match = re.search(r"/job/(\d+)$", url)
    if match is None:
        return ""
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    try:
        raw = fetch_archive(f"{base}/repos/{os.environ['GITHUB_REPOSITORY']}/actions/jobs/{match[1]}/logs")
    except urllib.error.URLError:  # HTTPError is a URLError. The pointer to the log is then the report.
        return ""
    lines = [LOG_NOISE.sub("", line) for line in raw.decode("utf-8", errors="replace").splitlines()]
    first = next((position for position, line in enumerate(lines) if "##[error]" in line), len(lines))
    return clip("\n".join(lines[max(first - ERROR_CONTEXT_BEFORE, 0) : first + ERROR_CONTEXT_AFTER]), MAX_LOG_TAIL)


def failure_text(url: str) -> str:
    """Write the report of a failed gate that uploads no failure report.

    Parameters
    ----------
    url : str
        The URL of the job log, or an empty string if the job has none.

    Returns
    -------
    str
        The excerpt of the log around the first error, then the pointer to the whole log.

    """
    excerpt = job_log_excerpt(url)
    return f"{block(excerpt)}\n\n{log_pointer(url)}" if excerpt.strip() else log_pointer(url)


def log_pointer(url: str) -> str:
    """Write the text for a failed gate that has no failure report.

    Parameters
    ----------
    url: str
        The URL of the job log, or an empty string if the job has none.

    Returns
    -------
    str
        The markdown that tells the reader where to find the error.

    """
    if url:
        return f"Open the [job log]({url}) for the whole output."
    return "The job failed before the tool produced a report (setup or runner error). Open the job log."


def render_progress(states: Mapping[str, str], reports: Mapping[str, str] | None = None) -> str:
    """Write the comment for a run that has not finished.

    Parameters
    ----------
    states: Mapping[str, str]
        The state of each gate, from ``gate_states``.
    reports: Mapping[str, str], optional
        The failure report of each failed gate whose report is available.

    Returns
    -------
    str
        The markdown of the comment.

    """
    icons = {
        "absent": "⚪ not started (no relevant change)",
        "waiting": "⏳ waiting",
        "running": "🔄 running",
        "success": "✅ passed",
        "failure": "❌ **failed**, details below",
        "cancelled": "🚫 cancelled",
        "skipped": "⏭️ skipped (no relevant change, or blocked by an earlier gate)",
    }
    tracked = [gate for gate in GATES if states[gate.id] != "absent"]
    finished = sum(states[gate.id] in FINISHED for gate in tracked)
    failed = sum(state == "failure" for state in states.values())
    heading = f"## 🔄 CI in progress: {finished} of {len(tracked)} gates finished"
    if failed:
        heading += f", {failed} failed"
    lines = [heading, "", "This comment updates by itself when a gate changes. The final report replaces it.", ""]
    lines += ["| Gate | State |", "|---|---|"]
    lines += [f"| {gate.title} | {icons.get(states[gate.id], states[gate.id])} |" for gate in GATES]
    lines.append("")
    for gate in GATES:
        if states[gate.id] == "failure" and (report := (reports or {}).get(gate.id)):
            lines += [f"### ❌ {gate.title}", "", report.strip(), ""]
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    lines += [f"<sub>Commit `{sha}` · [live run logs]({run_url()})</sub>", ""]
    return "\n".join(lines)


def update_progress(posted: str, reports: dict[str, str]) -> tuple[bool, str]:
    """Read the state of the gates and update the PR comment if the state changed.

    Parameters
    ----------
    posted: str
        The text that the last update wrote.
    reports: dict[str, str]
        The failure reports found so far. The function adds the report of each newly failed gate. A dict is
        needed because the function stores into it.

    Returns
    -------
    tuple[bool, str]
        Whether all gates have finished, and the text that the PR comment now holds.

    """
    runs = check_runs()
    states = gate_states(runs, os.environ["GITHUB_EVENT_NAME"])
    if all(state in FINISHED for state in states.values()):
        return True, posted
    links = log_links(runs)
    for gate, state in states.items():
        if state != "failure" or gate in reports:
            continue
        if gate not in BUILDERS:
            reports[gate] = failure_text(links.get(gate, ""))
        elif report := download_report(gate):
            reports[gate] = report  # an empty result means the job has not uploaded yet, so ask again next poll
    text = render_progress(states, reports)
    if text != posted:
        upsert_comment(text)
    return False, text


def watch() -> None:
    """Update the PR comment with the state of the gates until all gates have finished."""
    deadline = time.monotonic() + WATCH_LIMIT_SECONDS
    posted = ""
    reports: dict[str, str] = {}
    while time.monotonic() < deadline:
        try:
            done, posted = update_progress(posted, reports)
        except urllib.error.HTTPError as error:
            if error.code in FORBIDDEN:  # a fork PR has a read-only token, so there is nothing to update
                return
            sys.stderr.write(f"progress update failed: {error}\n")
        except urllib.error.URLError as error:
            sys.stderr.write(f"progress update failed: {error}\n")
        else:
            if done:
                return
        time.sleep(POLL_SECONDS)


def post(path: str) -> None:
    """Create the PR comment from a file, or replace the existing comment.

    Parameters
    ----------
    path: str
        The markdown file.

    Raises
    ------
    urllib.error.HTTPError
        If the API refuses the request for a reason other than missing write access.

    """
    try:
        upsert_comment(Path(path).read_text(encoding="utf-8"))
    except urllib.error.HTTPError as error:
        if error.code not in FORBIDDEN:
            raise
        sys.stderr.write("no write access to the PR (a fork PR?); the report stays in the run summary\n")


def settle() -> tuple[dict[str, str], dict[str, str]]:
    """Wait until every gate has finished, or until ``SETTLE_SECONDS`` have passed.

    The gates of ci.yml have finished when this runs. The gates of the other workflows run in parallel, so they may
    still run.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        The state of each gate, and the URL of its job log.

    """
    deadline = time.monotonic() + SETTLE_SECONDS
    states = {gate.id: "waiting" for gate in GATES}
    links: dict[str, str] = {}
    while True:
        try:
            runs = check_runs()
        except urllib.error.URLError as error:  # HTTPError is a URLError
            sys.stderr.write(f"check run read failed: {error}\n")
        else:
            states, links = gate_states(runs, os.environ["GITHUB_EVENT_NAME"]), log_links(runs)
        if all(state in FINISHED for state in states.values()) or time.monotonic() >= deadline:
            return states, links
        time.sleep(POLL_SECONDS)


def assemble() -> None:
    """Combine the snippets and the check runs into the PR comment and the step summary."""
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    icons = {
        "absent": "⚪ not started (no relevant change)",
        "success": "✅ passed",
        "failure": "❌ **failed**",
        "cancelled": "🚫 cancelled",
        "skipped": "⏭️ skipped (no relevant change, or blocked by an earlier gate)",
    }
    results, links = settle()
    tracked = [gate for gate in GATES if results[gate.id] != "absent"]
    failed = [gate for gate in GATES if results[gate.id] == "failure"]

    if failed:
        lines = [f"## ❌ CI failed: {len(failed)} of {len(tracked)} gates", ""]
    elif all(r in {"success", "skipped", "absent"} for r in results.values()):
        # With no failure, a skipped gate is one that the changed files cannot affect (see `changes` in ci.yml).
        passed = sum(r == "success" for r in results.values())
        lines = ["## ✅ CI passed", "", f"{passed} of {len(tracked)} gates green on `{sha}`; the others need no run."]
    else:
        lines = ["## ⚠️ CI did not finish", ""]
    lines += ["| Gate | Result |", "|---|---|"]
    lines += [f"| {gate.title} | {icons.get(results[gate.id], results[gate.id])} |" for gate in GATES]
    lines.append("")
    for gate in failed:
        body = read(REPORT / f"{gate.id}.md") or failure_text(links.get(gate.id, ""))
        lines += [f"### ❌ {gate.title}", "", body.strip(), ""]
    lines.append(f"<sub>Commit `{sha}` · [full run logs]({run_url()})</sub>")
    text = "\n".join(lines) + "\n"
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "comment.md").write_text(text)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(text)


def main() -> int:
    """Run the subcommand given on the command line.

    Returns
    -------
    int
        The exit code: 0 on success, 2 on a usage error.

    """
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "snippet" and len(sys.argv) == SNIPPET_ARGC and sys.argv[2] in BUILDERS:
        snippet(sys.argv[2])
    elif command == "assemble":
        assemble()
    elif command == "watch":
        watch()
    elif command == "post" and len(sys.argv) == POST_ARGC:
        post(sys.argv[2])
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
