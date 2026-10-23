"""Run cosmic-ray mutation testing in parallel and report the mutation score.

cosmic-ray mutates the source tree in place, so every shard works on its own copy of the project. The work items of
each module in the main session are split over the shards, executed concurrently, and the results are merged back into
the main session for reporting. A shard runs the tests of its own module first, because they kill most mutants of that
module, and all shards share one cache of fitted models. A progress line appears every minute.

Usage: ``uv run python scripts/mutation.py [shards] [--base REF | --module PATH ...]``. With ``--base``, only the
modules that changed since ``REF`` are mutated. With ``--module``, only the given modules are mutated. Exits non-zero
if any mutant survives.
"""

import argparse
import os
import shutil
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- must launch cosmic-ray and the interpreter as subprocesses
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import attrs
import structlog

logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "cosmic-ray.toml"
SESSION = ROOT / ".mutation" / "session.sqlite"
COPIED = ("src", "tests", "benchmark", "scripts", "pyproject.toml", "cosmic-ray.toml")
POLL_SECONDS = 60


@attrs.frozen
class Shard:
    """A copy of the project that runs a share of the work items of one module.

    Attributes
    ----------
    path: Path
        The directory of the shard.
    module: str
        The path of the module, relative to the project root.

    """

    path: Path
    module: str


def run(*args: str, cwd: Path = ROOT, env: dict | None = None) -> str:
    """Run a command and collect its output, whatever its exit code.

    Parameters
    ----------
    *args: str
        The command and its arguments.
    cwd: Path
        The working directory.
    env: dict, optional
        The environment of the command. Defaults to the current environment.

    Returns
    -------
    str
        The standard output followed by the standard error.

    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def changed_modules(base: str) -> set[str]:
    """List the source modules that changed between a base revision and HEAD.

    Parameters
    ----------
    base: str
        The base revision, for example ``origin/main``. The comparison starts at the merge base.

    Returns
    -------
    set[str]
        The paths of the changed modules, relative to the project root.

    """
    paths = run("git", "diff", "--name-only", f"{base}...HEAD", "--", "src/ebaml").split()
    return {path for path in paths if path.endswith(".py") and (ROOT / path).exists()}


def session_modules(session: Path) -> set[str]:
    """List the modules that have work items in a session.

    Parameters
    ----------
    session: Path
        The cosmic-ray session.

    Returns
    -------
    set[str]
        The paths of the modules, relative to the project root.

    """
    with sqlite3.connect(session) as db:
        return {row[0] for row in db.execute("SELECT DISTINCT module_path FROM mutation_specs")}


def keep_modules(session: Path, modules: set[str]) -> None:
    """Drop the work items of all modules except the given ones.

    Parameters
    ----------
    session: Path
        The cosmic-ray session.
    modules: set[str]
        The paths of the modules to keep.

    """
    paths = session_modules(session)
    with sqlite3.connect(session) as db:
        dropped = [(path,) for path in paths - modules]
        db.executemany(
            "DELETE FROM work_items WHERE job_id IN (SELECT job_id FROM mutation_specs WHERE module_path = ?)",
            dropped,
        )
        db.executemany("DELETE FROM mutation_specs WHERE module_path = ?", dropped)
        db.commit()


def make_shard(workdir: Path, module: str, index: int, shards: int) -> Shard:
    """Copy the project into its own directory, together with the share of the work items of the shard.

    Parameters
    ----------
    workdir: Path
        The directory that holds the shards.
    module: str
        The path of the module whose work items the shard runs.
    index: int
        The index of the shard among the shards of the module.
    shards: int
        The number of shards of the module.

    Returns
    -------
    Shard
        The shard.

    """
    path = workdir / f"{Path(module).stem}-{index}"
    path.mkdir()
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, path / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, path / name)
    session = path / "session.sqlite"
    shutil.copy(SESSION, session)
    with sqlite3.connect(session) as db:
        # keep every ``shards``-th job of the module; results of earlier runs are dropped so they are recomputed
        # jobs the pragma filter skipped keep their result in the main session and are not executed again
        skipped = {row[0] for row in db.execute("SELECT job_id FROM work_results WHERE worker_outcome = 'SKIPPED'")}
        db.execute("DELETE FROM work_results")
        job_ids = [row[0] for row in db.execute("SELECT job_id FROM work_items ORDER BY job_id")]
        of_module = {row[0] for row in db.execute("SELECT job_id FROM mutation_specs WHERE module_path = ?", (module,))}
        candidates = [job for job in job_ids if job in of_module and job not in skipped]
        mine = set(candidates[index::shards])
        dropped = [(job,) for job in job_ids if job not in mine]
        db.executemany("DELETE FROM mutation_specs WHERE job_id = ?", dropped)
        db.executemany("DELETE FROM work_items WHERE job_id = ?", dropped)
    return Shard(path, module)


def exec_shard(shard: Shard, fits: Path) -> None:
    """Execute the work items of a shard against the copy of the project of that shard.

    Parameters
    ----------
    shard: Shard
        The shard.
    fits: Path
        The directory of the cache of fitted models that all shards share.

    """
    env = {
        **os.environ,
        "PYTHONPATH": str(shard.path / "src"),
        # tests/unit/conftest.py runs the tests of this module first, so a killed mutant stops the run early
        "MUTATION_FIRST": Path(shard.module).stem,
        # tests/unit/ebaml/conftest.py reads fitted models from this directory instead of fitting them again
        "EBAML_FIT_CACHE": str(fits),
    }
    started = time.monotonic()
    run(sys.executable, "-m", "cosmic_ray.cli", "exec", "cosmic-ray.toml", "session.sqlite", cwd=shard.path, env=env)
    logger.info("shard done", module=shard.module, shard=shard.path.name, seconds=round(time.monotonic() - started))


def count_results(sessions: list[Path], seen: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    """Count the finished and the surviving mutants of the sessions of the shards.

    Parameters
    ----------
    sessions: list[Path]
        The sessions of the shards.
    seen: dict[Path, tuple[int, int]]
        The last count of each session. A session that is locked for writing keeps its last count.

    Returns
    -------
    tuple[int, int]
        The number of finished mutants and the number of surviving mutants.

    """
    for session in sessions:
        try:
            with sqlite3.connect(session, timeout=5) as db:
                query = "SELECT COUNT(*), COALESCE(SUM(test_outcome = 'SURVIVED'), 0) FROM work_results"
                seen[session] = db.execute(query).fetchone()
        except sqlite3.OperationalError:
            logger.warning("session locked", session=str(session))
    return sum(done for done, _ in seen.values()), sum(survived for _, survived in seen.values())


def report_progress(sessions: list[Path], total: int, stop: threading.Event) -> None:
    """Log the progress of the shards every ``POLL_SECONDS`` seconds, until the event is set.

    Parameters
    ----------
    sessions: list[Path]
        The sessions of the shards.
    total: int
        The number of mutants that the shards run.
    stop: threading.Event
        The event that ends the loop.

    """
    started = time.monotonic()
    seen: dict[Path, tuple[int, int]] = {}
    while not stop.wait(POLL_SECONDS):
        done, survived = count_results(sessions, seen)
        elapsed = time.monotonic() - started
        eta = round(elapsed / done * (total - done)) if done else None
        logger.info("progress", done=done, total=total, survived=survived, elapsed_s=round(elapsed), eta_s=eta)


def merge_results(sessions: list[Path]) -> None:
    """Copy the results of the shards into the main session.

    Parameters
    ----------
    sessions: list[Path]
        The sessions of the shards.

    """
    with sqlite3.connect(SESSION) as db:
        for session in sessions:
            db.execute("ATTACH DATABASE ? AS shard", (str(session),))
            db.execute("INSERT OR REPLACE INTO work_results SELECT * FROM shard.work_results")
            db.commit()
            db.execute("DETACH DATABASE shard")


def count_mutants() -> int:
    """Count the mutants that the main session still has to run.

    Returns
    -------
    int
        The number of work items, without the items that the pragma filter skipped.

    """
    with sqlite3.connect(SESSION) as db:
        return db.execute(
            "SELECT COUNT(*) FROM work_items WHERE job_id NOT IN "
            "(SELECT job_id FROM work_results WHERE worker_outcome = 'SKIPPED')",
        ).fetchone()[0]


def main() -> int:
    """Run the mutation testing in shards.

    Returns
    -------
    int
        The exit code: non-zero if the baseline fails or any mutant survives.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="?", type=int, default=os.cpu_count() or 1)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--base", help="mutate only the modules that changed since this revision")
    which.add_argument("--module", action="append", help="mutate only this module; repeat for more modules")
    args = parser.parse_args()
    shards = args.shards
    SESSION.parent.mkdir(exist_ok=True)
    SESSION.unlink(missing_ok=True)
    run(sys.executable, "-m", "cosmic_ray.cli", "init", str(CONFIG), str(SESSION))
    # lines marked ``# pragma: no mutate`` hold mutants that are equivalent to the original code
    run(sys.executable, "-m", "cosmic_ray.tools.filters.pragma_no_mutate", str(SESSION))
    if args.base or args.module:
        if args.base:
            modules = changed_modules(args.base)
        else:
            modules = set(args.module)
            missing = sorted(path for path in modules if not (ROOT / path).is_file())
            if missing:
                parser.error(f"no such module: {', '.join(missing)}")
        keep_modules(SESSION, modules)
        logger.info("partial run", modules=sorted(modules), base=args.base)
        if not modules:
            return 0
    modules = sorted(session_modules(SESSION))
    if not modules:
        logger.info("no mutants to run")
        return 0
    total = count_mutants()
    logger.info("mutants", total=total, modules=modules, shards=shards)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        fits = workdir / "fits"
        fits.mkdir()
        # the baseline runs every test once, so it also fills the cache of fitted models for the shards
        baseline = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
            [sys.executable, "-m", "cosmic_ray.cli", "baseline", str(CONFIG)],
            cwd=ROOT,
            env={**os.environ, "EBAML_FIT_CACHE": str(fits)},
            capture_output=True,
            text=True,
            check=False,
        )
        if baseline.returncode:
            return baseline.returncode

        prepared = [make_shard(workdir, module, i, shards) for module in modules for i in range(shards)]
        sessions = [shard.path / "session.sqlite" for shard in prepared]
        stop = threading.Event()
        monitor = threading.Thread(target=report_progress, args=(sessions, total, stop))
        monitor.start()
        try:
            with ThreadPoolExecutor(max_workers=shards) as pool:
                list(pool.map(lambda shard: exec_shard(shard, fits), prepared))
        finally:
            stop.set()
            monitor.join()
        merge_results(sessions)

    rate = float(run(sys.executable, "-m", "cosmic_ray.tools.survival_rate", str(SESSION)).strip())
    logger.info("mutation score", survival_rate=rate)
    return int(rate > 0)


if __name__ == "__main__":
    sys.exit(main())
