"""Run cosmic-ray mutation testing in parallel and report the mutation score.

cosmic-ray mutates the source tree in place, so every shard works on its own copy of the project. The work items of
the main session are split over the shards, executed concurrently, and the results are merged back into the main
session for reporting.

Usage: ``uv run python scripts/mutation.py [shards]``. Exits non-zero if any mutant survives.
"""

import os
import shutil
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- must launch cosmic-ray and the interpreter as subprocesses
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "cosmic-ray.toml"
SESSION = ROOT / ".mutation" / "session.sqlite"
COPIED = ("src", "tests", "benchmark", "scripts", "pyproject.toml", "cosmic-ray.toml")


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


def make_shard(workdir: Path, index: int, shards: int) -> Path:
    """Copy the project into its own directory, together with the share of the work items of the shard.

    Parameters
    ----------
    workdir: Path
        The directory that holds the shards.
    index: int
        The index of the shard.
    shards: int
        The total number of shards.

    Returns
    -------
    Path
        The directory of the shard.

    """
    shard = workdir / f"shard{index}"
    shard.mkdir()
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, shard / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, shard / name)
    session = shard / "session.sqlite"
    shutil.copy(SESSION, session)
    with sqlite3.connect(session) as db:
        # keep every ``shards``-th job; results of earlier runs are dropped so they are recomputed
        # jobs the pragma filter skipped keep their result in the main session and are not executed again
        skipped = {row[0] for row in db.execute("SELECT job_id FROM work_results WHERE worker_outcome = 'SKIPPED'")}
        db.execute("DELETE FROM work_results")
        job_ids = [
            row[0] for row in db.execute("SELECT job_id FROM work_items ORDER BY job_id") if row[0] not in skipped
        ]
        mine = set(job_ids[index::shards])
        db.executemany("DELETE FROM mutation_specs WHERE job_id = ?", [(j,) for j in job_ids if j not in mine])
        db.executemany("DELETE FROM work_items WHERE job_id = ?", [(j,) for j in job_ids if j not in mine])
        db.executemany("DELETE FROM work_items WHERE job_id = ?", [(j,) for j in skipped])
        db.executemany("DELETE FROM mutation_specs WHERE job_id = ?", [(j,) for j in skipped])
    return shard


def exec_shard(shard: Path) -> None:
    """Execute the work items of a shard against the copy of the project of that shard.

    Parameters
    ----------
    shard: Path
        The directory of the shard.

    """
    env = {**os.environ, "PYTHONPATH": str(shard / "src")}
    run(sys.executable, "-m", "cosmic_ray.cli", "exec", "cosmic-ray.toml", "session.sqlite", cwd=shard, env=env)


def main() -> int:
    """Run the mutation testing in shards.

    Returns
    -------
    int
        The exit code: non-zero if the baseline fails or any mutant survives.

    """
    shards = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() or 1
    SESSION.parent.mkdir(exist_ok=True)
    SESSION.unlink(missing_ok=True)
    run(sys.executable, "-m", "cosmic_ray.cli", "init", str(CONFIG), str(SESSION))
    # lines marked ``# pragma: no mutate`` hold mutants that are equivalent to the original code
    run(sys.executable, "-m", "cosmic_ray.tools.filters.pragma_no_mutate", str(SESSION))
    baseline = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable, never untrusted input
        [sys.executable, "-m", "cosmic_ray.cli", "baseline", str(CONFIG)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if baseline.returncode:
        return baseline.returncode

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        prepared = [make_shard(workdir, i, shards) for i in range(shards)]
        with ThreadPoolExecutor(max_workers=shards) as pool:
            list(pool.map(exec_shard, prepared))
        with sqlite3.connect(SESSION) as db:
            for shard in prepared:
                db.execute("ATTACH DATABASE ? AS shard", (str(shard / "session.sqlite"),))
                db.execute("INSERT OR REPLACE INTO work_results SELECT * FROM shard.work_results")
                db.commit()
                db.execute("DETACH DATABASE shard")

    rate = float(run(sys.executable, "-m", "cosmic_ray.tools.survival_rate", str(SESSION)).strip())
    return int(rate > 0)


if __name__ == "__main__":
    sys.exit(main())
