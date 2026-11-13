"""Tests for the check that a file, a function or a class stays under its size limit."""

from pathlib import Path

from scripts.check_size import (
    FUNCTION_LINES,
    PUBLIC_METHODS_PER_CLASS,
    SOURCE_FILE_LINES,
    TEST_FILE_LINES,
    Measure,
    measure_file,
    size_problems,
)


def _measures(tmp_path: Path, source: str, *, is_test: bool = False) -> dict[str, Measure]:
    path = tmp_path / "module.py"
    path.write_text(source)
    return {measure.key: measure for measure in measure_file(path, "module.py", is_test=is_test)}


def test_a_function_is_measured_from_its_def_line_to_its_last_line(tmp_path: Path) -> None:
    body = "\n".join("    value = 1" for _ in range(FUNCTION_LINES))
    measures = _measures(tmp_path, f"def long():\n{body}\n")
    assert measures["module.py::long"].size == FUNCTION_LINES + 1
    assert measures["module.py::long"].is_over


def test_the_docstring_of_a_function_does_not_count(tmp_path: Path) -> None:
    # a limit that counted documentation would push authors to write less of it
    docstring = '    """Describe.\n\n' + "\n".join("    More." for _ in range(FUNCTION_LINES)) + '\n    """\n'
    measures = _measures(tmp_path, f"def documented():\n{docstring}    return 1\n")
    assert measures["module.py::documented"].size == 2


def test_a_test_file_has_a_higher_limit_than_a_source_file(tmp_path: Path) -> None:
    assert _measures(tmp_path, "x = 1\n")["module.py"].limit == SOURCE_FILE_LINES
    assert _measures(tmp_path, "x = 1\n", is_test=True)["module.py"].limit == TEST_FILE_LINES


def test_a_class_counts_public_and_private_methods_apart_and_skips_dunder_methods(tmp_path: Path) -> None:
    lines = [
        "class A:",
        "    def __init__(self): ...",
        "    def run(self): ...",
        "    def _help(self): ...",
        "    def _more(self): ...",
    ]
    source = "\n".join(lines) + "\n"
    measures = _measures(tmp_path, source)
    assert measures["module.py::A:public"].size == 1
    assert measures["module.py::A:private"].size == 2


def test_two_methods_with_one_name_in_two_classes_have_two_keys(tmp_path: Path) -> None:
    measures = _measures(tmp_path, "class A:\n    def run(self): ...\nclass B:\n    def run(self): ...\n")
    assert {"module.py::A.run", "module.py::B.run"} <= measures.keys()


def test_a_class_over_the_method_limit_is_over(tmp_path: Path) -> None:
    methods = "".join(f"    def method_{number}(self): ...\n" for number in range(PUBLIC_METHODS_PER_CLASS + 1))
    assert _measures(tmp_path, f"class A:\n{methods}")["module.py::A:public"].is_over


def _measure(size: int, limit: int = 10) -> Measure:
    return Measure("module.py", size, limit, "lines")


def test_a_unit_under_its_limit_has_no_problem() -> None:
    assert size_problems([_measure(10)], {}) == []


def test_a_new_unit_over_the_limit_fails() -> None:
    [problem] = size_problems([_measure(11)], {})
    assert "module.py: 11 lines" in problem


def test_a_unit_in_the_baseline_may_not_grow() -> None:
    assert size_problems([_measure(15)], {"module.py": 15}) == []
    [problem] = size_problems([_measure(16)], {"module.py": 15})
    assert "limit is 15" in problem


def test_a_shrunk_unit_must_lower_its_entry_and_a_unit_under_the_limit_must_leave_the_baseline() -> None:
    [lower] = size_problems([_measure(12)], {"module.py": 15})
    assert "Lower the entry" in lower
    [delete] = size_problems([_measure(10)], {"module.py": 15})
    assert "Delete the baseline entry" in delete


def test_an_entry_for_a_missing_unit_fails() -> None:
    [problem] = size_problems([], {"gone.py": 99})
    assert "does not exist" in problem
