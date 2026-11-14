"""Run the tests of one mutant: the tests that cover the mutated statements first, then all tests.

cosmic-ray starts the test command once for each mutant, and pytest then collects every test of ``tests/unit`` before
it runs the first one. The collection takes about as long as the run of a killed mutant. This script first runs only the
tests that execute the mutated statements, so pytest imports only their files. A failure there kills the mutant at once.
If they pass, the script runs the full command, so a mutant survives only when every test passes. The result is the
same as the result of the full command alone, and a killed mutant is much faster.

``scripts/mutation/run.py`` sets this script as the test command of each shard, and gives the full command in
``MUTATION_TEST_COMMAND``.
"""

import os
import shlex
import sys
from pathlib import Path

from scripts.fixed_command import run_fixed
from scripts.mutation import covering_tests

PYTEST_NO_TESTS = 5  # the exit code of pytest when it collects no test
PYTEST_PASSED = 0
TEST_PATH = "tests/unit"


def stage_one_command(command: list[str], covering: set[str]) -> list[str]:
    """Build the command that runs only the covering tests.

    Parameters
    ----------
    command: list[str]
        The full test command. It ends with the test directory.
    covering: set[str]
        The node ids of the tests that execute the mutated statements.

    Returns
    -------
    list[str]
        The command with the node ids in place of the test directory.

    """
    return [*(word for word in command if word != TEST_PATH), *sorted(covering)]


def run_tests(command: list[str], covering: set[str]) -> int:
    """Run the covering tests, and then the full command if the covering tests pass.

    Parameters
    ----------
    command: list[str]
        The full test command.
    covering: set[str]
        The node ids of the tests that execute the mutated statements. With no node id, only the full command runs.

    Returns
    -------
    int
        The exit code: zero if every test passes, which means that the mutant survives.

    """
    if covering:
        first = run_fixed(stage_one_command(command, covering), capture_output=False).returncode
        if first not in {PYTEST_PASSED, PYTEST_NO_TESTS}:
            return first
    return run_fixed(command, capture_output=False).returncode


def main() -> int:
    """Run the tests of the mutant that the environment describes.

    Returns
    -------
    int
        The exit code of the tests.

    """
    command = shlex.split(os.environ["MUTATION_TEST_COMMAND"])
    if command[0] == "python":
        command[0] = sys.executable
    return run_tests(command, covering_tests.covering_from_environment(Path.cwd(), os.environ))


if __name__ == "__main__":
    sys.exit(main())
