"""Parse the registry of shared abstractions (``architecture/abstractions.yml``) into typed classes."""

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import attrs
import yaml

ROOT = Path(__file__).resolve().parents[2]


REGISTRY_PATH = Path("architecture") / "abstractions.yml"


SOURCE_PATH = Path("src") / "ebaml"


SCHEMA_VERSION = 1


ERROR = "error"


WARNING = "warning"


ACTIVE = "active"


DEPRECATED = "deprecated"


STATUSES = (ACTIVE, DEPRECATED)


DEFAULT_CONTEXT = "production"


ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


MIN_REASON_WORDS = 4


MIN_REASON_CHARACTERS = 20


CLOSE_MATCH_COUNT = 3


@attrs.frozen
class Diagnostic:
    """One finding of a check, in a form that a person and a tool can read.

    Parameters
    ----------
    rule_id: str
        The stable name of the rule that failed.
    path: str
        The file of the finding, relative to the repo root.
    line: int
        The line of the finding, starting at 1.
    capability_id: str
        The ID of the capability, or an empty string if the finding has none.
    candidate: str
        The existing symbol or capability that the finding refers to, or an empty string.
    suggestion: str
        What to do to fix the finding.
    severity: str
        ``error`` fails the check. ``warning`` is a review cue.

    """

    rule_id: str
    path: str
    line: int
    capability_id: str
    candidate: str
    suggestion: str
    severity: str = ERROR

    def render(self) -> str:
        """Format the finding as one text line.

        Returns
        -------
        str
            ``path:line: rule_id [capability_id] (candidate) - suggestion``. A warning starts with ``warning:``.

        """
        parts = [f"{self.path}:{self.line}: {self.rule_id}"]
        if self.severity != ERROR:
            parts.insert(0, f"{self.severity}:")
        if self.capability_id:
            parts.append(f"[{self.capability_id}]")
        if self.candidate:
            parts.append(f"({self.candidate})")
        return f"{' '.join(parts)} - {self.suggestion}"


@attrs.frozen
class Capability:
    """One entry of the registry.

    Parameters
    ----------
    capability_id: str
        The stable ID in lower-case kebab-case.
    responsibility: str
        One sentence that says what the capability does.
    owner: str
        The module or package that owns the capability, for example ``ebaml.graphs``.
    symbol: str
        The canonical symbol as ``module:Name``.
    consumers: tuple[str, ...]
        The modules that use the capability. The list describes, and it grants no import right.
    keywords: tuple[str, ...]
        Words for the search.
    status: str
        ``active`` or ``deprecated``.
    context: str
        The bounded context, ``production`` unless the entry says otherwise.
    replaced_by: str
        The ID of the replacement of a deprecated capability, or an empty string.
    line: int
        The line of the entry in the registry file.

    """

    capability_id: str
    responsibility: str
    owner: str
    symbol: str
    consumers: tuple[str, ...]
    keywords: tuple[str, ...]
    status: str
    context: str
    replaced_by: str
    line: int


@attrs.frozen
class OverlapException:
    """An approved overlap of two contracts, with the reason.

    Parameters
    ----------
    symbols: tuple[str, str]
        The two contracts as ``module:Name``, in any order.
    reason: str
        Why the two contracts stay apart.
    line: int
        The line of the entry in the registry file.

    """

    symbols: tuple[str, str]
    reason: str
    line: int


@attrs.frozen
class Registry:
    """The parsed registry.

    Parameters
    ----------
    capabilities: tuple[Capability, ...]
        The entries, in file order.
    exceptions: tuple[OverlapException, ...]
        The approved contract overlaps.

    """

    capabilities: tuple[Capability, ...]
    exceptions: tuple[OverlapException, ...]


def _text_list(value: object) -> list[str] | None:
    """Read a list of strings from a YAML value.

    Parameters
    ----------
    value: object
        The parsed YAML value. ``None`` stands for an absent key.

    Returns
    -------
    list[str] | None
        The strings, an empty list for ``None``, or ``None`` if the value is not a list of strings.

    """
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in cast(list[object], value)):
        return cast(list[str], list(value))
    return None


def _entry_lines(document: yaml.Node | None, key: str) -> list[int]:
    """Find the line of each item under a top-level key of the registry.

    Parameters
    ----------
    document: yaml.Node | None
        The composed YAML document.
    key: str
        The top-level key whose value is a list.

    Returns
    -------
    list[int]
        The first line of each item, starting at 1. The list is empty if the key is absent.

    """
    if not isinstance(document, yaml.MappingNode):
        return []
    for key_node, value_node in document.value:
        if key_node.value == key and isinstance(value_node, yaml.SequenceNode):
            return [item.start_mark.line + 1 for item in value_node.value]
    return []


def _capability(entry: Mapping[str, Any], line: int) -> Capability | str:
    """Build a capability from one registry entry.

    Parameters
    ----------
    entry: Mapping[str, Any]
        The parsed YAML mapping.
    line: int
        The line of the entry.

    Returns
    -------
    Capability | str
        The capability, or the message that says what is wrong with the entry.

    """
    required = ("id", "responsibility", "owner", "symbol", "status")
    missing = [key for key in required if not isinstance(entry.get(key), str) or not entry[key].strip()]
    consumers = _text_list(entry.get("consumers"))
    keywords = _text_list(entry.get("keywords"))
    if missing:
        return f"the entry needs a text value for {', '.join(missing)}"
    if consumers is None or keywords is None:
        return "consumers and keywords must be lists of text"
    return Capability(
        capability_id=entry["id"],
        responsibility=entry["responsibility"],
        owner=entry["owner"],
        symbol=entry["symbol"],
        consumers=tuple(consumers),
        keywords=tuple(keywords),
        status=entry["status"],
        context=str(entry.get("context", DEFAULT_CONTEXT)),
        replaced_by=str(entry.get("replaced_by", "")),
        line=line,
    )


def _overlap_exception(entry: Mapping[str, Any], line: int) -> OverlapException | str:
    """Build an overlap exception from one registry entry.

    Parameters
    ----------
    entry: Mapping[str, Any]
        The parsed YAML mapping.
    line: int
        The line of the entry.

    Returns
    -------
    OverlapException | str
        The exception, or the message that says what is wrong with the entry.

    """
    symbols = _text_list(entry.get("symbols"))
    reason = entry.get("reason")
    if symbols is None or len(symbols) != 2 or not isinstance(reason, str):  # ruff: ignore[magic-value-comparison] -- an exception names a pair of contracts
        return "an exception needs `symbols` (two module:Name values) and a `reason`"
    return OverlapException(symbols=(symbols[0], symbols[1]), reason=reason, line=line)


def _schema_problem(path: str, line: int, message: str) -> Diagnostic:
    """Build the diagnostic for a malformed registry.

    Parameters
    ----------
    path: str
        The registry file.
    line: int
        The line of the problem.
    message: str
        What is wrong.

    Returns
    -------
    Diagnostic
        A ``registry-schema`` finding.

    """
    return Diagnostic("registry-schema", path, line, "", "", message)


def parse_registry(text: str, path: str = REGISTRY_PATH.as_posix()) -> tuple[Registry, list[Diagnostic]]:
    """Parse the registry text and check its shape.

    Parameters
    ----------
    text: str
        The content of ``architecture/abstractions.yml``.
    path: str
        The file name for the diagnostics.

    Returns
    -------
    tuple[Registry, list[Diagnostic]]
        The valid entries and the findings about the others. A malformed file gives an empty registry.

    """
    empty = Registry(capabilities=(), exceptions=())
    try:
        document: yaml.Node | None = yaml.compose(text)
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        return empty, [_schema_problem(path, 1, f"the YAML is malformed: {error}")]
    if not isinstance(data, dict):
        return empty, [_schema_problem(path, 1, "the file must be a mapping with `schema_version` and `capabilities`")]
    if data.get("schema_version") != SCHEMA_VERSION:
        return empty, [_schema_problem(path, 1, f"`schema_version` must be {SCHEMA_VERSION}")]
    capabilities, capability_problems = _parse_items(data, document, "capabilities", _capability, path)
    exceptions, exception_problems = _parse_items(data, document, "exceptions", _overlap_exception, path)
    problems = capability_problems + exception_problems
    return Registry(capabilities=tuple(capabilities), exceptions=tuple(exceptions)), problems


def _parse_items[T](
    data: Mapping[str, Any],
    document: yaml.Node | None,
    key: str,
    build: Callable[[Mapping[str, Any], int], T | str],
    path: str,
) -> tuple[list[T], list[Diagnostic]]:
    """Build the entries under one top-level key of the registry.

    Parameters
    ----------
    data: Mapping[str, Any]
        The parsed registry.
    document: yaml.Node | None
        The composed registry, for the line numbers.
    key: str
        The top-level key, ``capabilities`` or ``exceptions``.
    build: Callable[[Mapping[str, Any], int], T | str]
        Builds one entry from its mapping and its line. It returns a message if the entry is wrong.
    path: str
        The registry file for the diagnostics.

    Returns
    -------
    tuple[list[T], list[Diagnostic]]
        The valid entries and the findings about the others.

    """
    items = data.get(key) or []
    if not isinstance(items, list):
        return [], [_schema_problem(path, 1, f"`{key}` must be a list")]
    lines = _entry_lines(document, key)
    built: list[T] = []
    problems: list[Diagnostic] = []
    for number, item in enumerate(cast(list[object], items)):
        line = lines[number] if number < len(lines) else 1
        result = build(item, line) if isinstance(item, dict) else "the entry must be a mapping"
        if isinstance(result, str):
            problems.append(_schema_problem(path, line, result))
        else:
            built.append(result)
    return built, problems


def load_registry(root: Path) -> tuple[Registry, list[Diagnostic]]:
    """Read and parse the registry file of a repository.

    Parameters
    ----------
    root: Path
        The repo root.

    Returns
    -------
    tuple[Registry, list[Diagnostic]]
        The registry and the findings about its shape. A missing file is a finding too.

    """
    path = root / REGISTRY_PATH
    if not path.is_file():
        empty = Registry(capabilities=(), exceptions=())
        return empty, [_schema_problem(REGISTRY_PATH.as_posix(), 1, "the registry file is missing")]
    return parse_registry(path.read_text(encoding="utf-8"), REGISTRY_PATH.as_posix())
