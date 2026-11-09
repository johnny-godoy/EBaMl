"""Tests for the check that one ignore rule does not repeat beyond the limit."""

import pytest
from scripts import check_ignore_repeats
from scripts.check_ignore_repeats import LIMIT, SHARED_HOMES, find_repeats
from scripts.ignore_sites import Site

RULE = "ruff: ignore[banned-api]"
REASON = "the test builds the pandas frame that interpret fits on"


def _sites(
    count: int,
    *,
    file: str = "tests/unit/test_{index}.py",
    what: str = RULE,
    program: str = "ruff",
) -> list[Site]:
    return [
        Site(program=program, file=file.format(index=index), line=1, what=what, reason=REASON) for index in range(count)
    ]


def test_a_rule_at_the_limit_passes() -> None:
    assert find_repeats(_sites(LIMIT)) == []


def test_a_rule_over_the_limit_is_a_repeat_with_every_site() -> None:
    # six copies of one cause were the pattern that the audit found, and each copy hid the cause
    [repeat] = find_repeats(_sites(LIMIT + 1))
    assert (repeat.program, repeat.rule, len(repeat.sites)) == ("ruff", RULE, LIMIT + 1)


def test_sites_in_a_shared_home_do_not_count() -> None:
    home = min(SHARED_HOMES)
    assert find_repeats(_sites(LIMIT * 2, file=home)) == []
    assert find_repeats([*_sites(LIMIT, file=home), *_sites(LIMIT)]) == []


def test_a_tool_without_a_shared_fix_has_no_limit() -> None:
    assert find_repeats(_sites(LIMIT * 2, what="pragma: no mutate", program="cosmic-ray")) == []


def test_two_rules_do_not_add_up() -> None:
    assert find_repeats([*_sites(LIMIT), *_sites(LIMIT, what="ruff: ignore[too-many-arguments]")]) == []


def test_the_same_rule_of_two_tools_does_not_add_up() -> None:
    assert find_repeats([*_sites(LIMIT), *_sites(LIMIT, program="pyrefly")]) == []


def test_spaces_inside_a_rule_do_not_split_the_group() -> None:
    sites = [*_sites(LIMIT), *_sites(1, what="ruff:  ignore[banned-api] ", file="a.py")]
    assert len(find_repeats(sites)) == 1


def test_the_largest_repeat_comes_first() -> None:
    sites = [*_sites(LIMIT + 1), *_sites(LIMIT + 3, what="ruff: ignore[too-many-arguments]")]
    assert [len(repeat.sites) for repeat in find_repeats(sites)] == [LIMIT + 3, LIMIT + 1]


def test_the_message_names_the_places_and_the_rule_file() -> None:
    [repeat] = find_repeats(_sites(LIMIT + 1))
    assert "tests/unit/test_0.py:1" in repeat.message
    assert RULE in repeat.message
    assert "prek-and-ignores.md" in repeat.message


def test_main_fails_and_lists_a_repeat(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(check_ignore_repeats, "find_sites", lambda: _sites(LIMIT + 1))
    assert check_ignore_repeats.main() == 1
    assert RULE in capsys.readouterr().err


def test_main_passes_without_a_repeat(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(check_ignore_repeats, "find_sites", lambda: _sites(LIMIT))
    assert check_ignore_repeats.main() == 0
    assert not capsys.readouterr().err


def test_the_repo_has_no_repeat() -> None:
    assert find_repeats(check_ignore_repeats.find_sites()) == []
