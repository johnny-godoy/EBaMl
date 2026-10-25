"""Tests for the check that each user-visible change is released with a new version."""

from pathlib import Path

import pytest
from scripts import check_changelog

HEADER = "# Changelog\n\nIntro text.\n\n"
RELEASED = HEADER + "## [0.0.3] - 2026-10-25\n\n### Added\n\n- New.\n\n## [0.0.2] - 2026-10-23\n\n- Old.\n"


def test_a_changelog_whose_newest_heading_is_the_version_passes() -> None:
    assert check_changelog.changelog_problems(RELEASED, "0.0.3") == []


def test_an_unreleased_section_fails_because_the_change_was_not_released() -> None:
    changelog = HEADER + "## [Unreleased]\n\n### Added\n\n- Waiting.\n\n## [0.0.3] - 2026-10-25\n"
    [problem] = check_changelog.changelog_problems(changelog, "0.0.3")
    assert problem.startswith("CHANGELOG.md has an [Unreleased] section")


def test_an_empty_unreleased_heading_also_fails() -> None:
    changelog = HEADER + "## [unreleased]\n\n## [0.0.3] - 2026-10-25\n"
    assert check_changelog.changelog_problems(changelog, "0.0.3")


def test_a_version_that_was_not_raised_fails() -> None:
    [problem] = check_changelog.changelog_problems(RELEASED, "0.0.2")
    assert problem == "pyproject.toml has version 0.0.2, but the newest CHANGELOG.md heading is [0.0.3]"


def test_a_version_that_is_not_semver_fails_the_comparison() -> None:
    assert check_changelog.changelog_problems(RELEASED, "0.0.3rc1")


@pytest.mark.parametrize("heading", ["## [0.0.4]", "## [0.0.4] - 2026-13-01", "## [0.0.4] - 25/10/2026"])
def test_a_heading_without_a_valid_date_fails(heading: str) -> None:
    changelog = HEADER + heading + "\n\n" + RELEASED.removeprefix(HEADER)
    [problem] = check_changelog.changelog_problems(changelog, "0.0.4")
    assert problem == "heading [0.0.4] needs a release date in the form ' - YYYY-MM-DD'"


@pytest.mark.parametrize("name", ["1.0", "v1.0.0", "01.0.0"])
def test_a_heading_that_is_not_a_version_fails(name: str) -> None:
    changelog = HEADER + f"## [{name}] - 2026-10-26\n\n" + RELEASED.removeprefix(HEADER)
    problems = check_changelog.changelog_problems(changelog, "0.0.3")
    assert problems == [f"heading [{name}] is not a MAJOR.MINOR.PATCH version"]


@pytest.mark.parametrize("below", ["0.0.3", "0.0.4", "1.0.0"])
def test_headings_must_decrease(below: str) -> None:
    changelog = HEADER + f"## [0.0.3] - 2026-10-25\n\n## [{below}] - 2026-10-24\n"
    [problem] = check_changelog.changelog_problems(changelog, "0.0.3")
    assert problem == f"heading [{below}] must be lower than the heading above it"


def test_versions_compare_as_numbers_not_as_text() -> None:
    changelog = HEADER + "## [0.10.0] - 2026-10-25\n\n## [0.9.0] - 2026-10-24\n"
    assert check_changelog.changelog_problems(changelog, "0.10.0") == []


def test_a_changelog_without_releases_fails() -> None:
    assert check_changelog.changelog_problems(HEADER, "0.0.1") == ["CHANGELOG.md has no release heading"]


def test_main_reports_problems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.0.2"\n')
    (tmp_path / "CHANGELOG.md").write_text(RELEASED)
    monkeypatch.setattr(check_changelog, "ROOT", tmp_path)
    assert check_changelog.main() == 1
    assert "pyproject.toml has version 0.0.2" in capsys.readouterr().err


def test_main_passes_for_a_released_changelog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.0.3"\n')
    (tmp_path / "CHANGELOG.md").write_text(RELEASED)
    monkeypatch.setattr(check_changelog, "ROOT", tmp_path)
    assert check_changelog.main() == 0
