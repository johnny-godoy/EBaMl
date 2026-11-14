"""Tests for the choice of the mutants that the mutation run executes."""

import signal
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- starts a real child process to test the deadline kill
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
from coverage import CoverageData
from scripts.fixed_command import run_fixed
from scripts.mutation import run

DIFF = """\
diff --git a/src/ebaml/a.py b/src/ebaml/a.py
index 1111111..2222222 100644
--- a/src/ebaml/a.py
+++ b/src/ebaml/a.py
@@ -3 +3 @@ def f():
-    return 1
+    return 2
@@ -10,0 +11,3 @@ def g():
+    x = 1
+    y = 2
+    return x + y
diff --git a/src/ebaml/b.py b/src/ebaml/b.py
--- a/src/ebaml/b.py
+++ b/src/ebaml/b.py
@@ -5,2 +4,0 @@ def h():
-    a = 1
-    b = 2
"""


def make_session(path: Path, specs: list[tuple[str, str, int, int]]) -> Path:
    """Write a session with the tables of cosmic-ray that the mutation run reads.

    Parameters
    ----------
    path: Path
        The directory of the session.
    specs: list[tuple[str, str, int, int]]
        The job id, the module, the first row and the last row of each mutant.

    Returns
    -------
    Path
        The session.

    """
    session = path / "session.sqlite"
    with closing(sqlite3.connect(session)) as db, db:  # closing releases the file, db commits
        db.execute("CREATE TABLE work_items (job_id VARCHAR PRIMARY KEY)")
        db.execute(
            "CREATE TABLE mutation_specs (module_path VARCHAR, start_pos_row INTEGER, end_pos_row INTEGER, "
            "start_pos_col INTEGER DEFAULT 0, operator_name VARCHAR DEFAULT '', occurrence INTEGER DEFAULT 0, "
            "job_id VARCHAR PRIMARY KEY)",
        )
        db.execute(
            "CREATE TABLE work_results (job_id VARCHAR PRIMARY KEY, worker_outcome VARCHAR, test_outcome VARCHAR)",
        )
        db.executemany("INSERT INTO work_items VALUES (?)", [(job,) for job, *_ in specs])
        db.executemany(
            "INSERT INTO mutation_specs (module_path, start_pos_row, end_pos_row, job_id) VALUES (?, ?, ?, ?)",
            [(module, start, end, job) for job, module, start, end in specs],
        )
    return session


def jobs(session: Path) -> set[str]:
    with closing(sqlite3.connect(session)) as db, db:  # closing releases the file, db commits
        items = {row[0] for row in db.execute("SELECT job_id FROM work_items")}
        assert items == {row[0] for row in db.execute("SELECT job_id FROM mutation_specs")}
        return items


def test_parse_changed_lines_reads_the_new_side_of_each_hunk() -> None:
    assert run.parse_changed_lines(DIFF) == {"src/ebaml/a.py": {3, 11, 12, 13}, "src/ebaml/b.py": set()}


def test_parse_changed_lines_of_an_empty_diff_is_empty() -> None:
    assert run.parse_changed_lines("") == {}


def test_keep_lines_keeps_only_the_mutants_that_touch_a_changed_line(tmp_path: Path) -> None:
    session = make_session(
        tmp_path,
        [
            ("on", "src/ebaml/a.py", 3, 3),
            ("spans", "src/ebaml/a.py", 9, 11),
            ("before", "src/ebaml/a.py", 2, 2),
            ("after", "src/ebaml/a.py", 14, 14),
            ("other", "src/ebaml/b.py", 3, 3),
        ],
    )
    run.keep_lines(session, {"src/ebaml/a.py": {3, 11, 12, 13}})
    assert jobs(session) == {"on", "spans"}


def test_keep_modules_keeps_every_mutant_of_the_given_modules(tmp_path: Path) -> None:
    session = make_session(tmp_path, [("a", "src/ebaml/a.py", 1, 1), ("b", "src/ebaml/b.py", 900, 901)])
    run.keep_modules(session, ["src/ebaml/b.py"])
    assert jobs(session) == {"b"}


def test_build_coverage_map_lists_the_tests_of_each_line_of_the_given_modules(tmp_path: Path) -> None:
    module = tmp_path / "src" / "pkg" / "m.py"
    other = tmp_path / "src" / "pkg" / "other.py"
    data = CoverageData(str(tmp_path / ".coverage"))
    data.set_context("")
    data.add_lines({str(module): [1]})
    for context, lines in (("t.py::a|setup", [2]), ("t.py::a|run", [2, 3]), ("t.py::b|run", [3])):
        data.set_context(context)
        data.add_lines({str(module): lines, str(other): lines})
    data.write()
    coverage = run.build_coverage_map(tmp_path / ".coverage", tmp_path, {"src/pkg/m.py"})
    assert coverage == {"src/pkg/m.py": {"1": [], "2": ["t.py::a"], "3": ["t.py::a", "t.py::b"]}}


def test_the_script_starts_when_it_is_run_by_path() -> None:
    # CI and the docs run `python scripts/mutation/run.py`. Then only the scripts directory is on the import path, so an
    # import of `scripts.<module>` raised ModuleNotFoundError before the first line of main ran.
    result = run_fixed([sys.executable, str(Path(run.__file__)), "--help"], cwd=Path(run.__file__).parents[2])
    assert result.returncode == 0, result.stderr
    assert "mutation testing" in result.stdout


def test_keep_part_splits_the_mutants_by_position_into_parts_that_cover_all_of_them(tmp_path: Path) -> None:
    specs = [(f"job{row}", "src/ebaml/a.py", row, row) for row in (7, 3, 5, 1, 9)]
    parts = []
    for index in (1, 2):
        directory = tmp_path / str(index)
        directory.mkdir()
        session = make_session(directory, specs)
        run.keep_part(session, index, 2)
        parts.append(jobs(session))
    # the order is by row, not by job id: rows 1, 3, 5, 7, 9 alternate between the parts
    assert parts == [{"job1", "job5", "job9"}, {"job3", "job7"}]


def test_parse_part_reads_the_index_and_the_count() -> None:
    assert run.parse_part("2/4") == (2, 4)


@pytest.mark.parametrize("text", ["0/4", "5/4", "2", "a/b", "1/2/3", ""])
def test_parse_part_rejects_an_index_outside_the_count_or_a_bad_form(text: str) -> None:
    with pytest.raises(ValueError, match="INDEX/COUNT"):
        run.parse_part(text)


def test_the_counts_name_the_survivors_and_the_mutants_without_a_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(tmp_path, [(job, "src/ebaml/a.py", 1, 1) for job in ("killed", "survived", "open")])
    with closing(sqlite3.connect(session)) as db, db:  # closing releases the file, db commits
        db.executemany(
            "INSERT INTO work_results VALUES (?, 'NORMAL', ?)",
            [("killed", "KILLED"), ("survived", "SURVIVED")],
        )
    monkeypatch.setattr(run, "SESSION", session)
    assert run.count_survivors() == 1
    assert run.count_unfinished() == 1


def start_sleeper(seconds: int) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv from sys.executable
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        start_new_session=True,
    )


def test_wait_until_kills_a_process_that_runs_past_the_deadline() -> None:
    process = start_sleeper(60)
    started = time.monotonic()
    assert run.wait_until(process, time.monotonic() + 0.5) is False
    assert time.monotonic() - started < 30
    assert process.returncode == -signal.SIGKILL


def test_wait_until_lets_a_process_that_ends_in_time_finish() -> None:
    process = start_sleeper(0)
    assert run.wait_until(process, time.monotonic() + 60) is True
    assert run.wait_until(start_sleeper(0), None) is True
