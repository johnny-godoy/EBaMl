"""Check the registry of shared abstractions against the source, and search it.

``architecture/abstractions.yml`` lists the shared capabilities of ``src/ebaml``: the contracts and entry points that
more than one module uses. Each capability has one stable ID, one owner and one canonical symbol. An agent that wants a
new shared contract searches the registry first, so that it reuses an existing one (``.claude/rules/reuse.md``).

The check reads the source with ``ast`` and never imports it. It fails when:

- the registry is malformed, or an ID is not lower-case kebab-case or is not unique,
- an owner or a consumer is not a module of ``src/ebaml``,
- a symbol does not exist, or its module is outside the owner,
- two active capabilities claim the same symbol, or the same responsibility in the same owner and context,
- a deprecated capability does not name its replacement,
- an exception for a contract overlap has no reason or names a symbol that does not exist.

A private helper needs no entry. The check fails on no class only because the registry lacks it.

Run it from the repo root:

- ``uv run python -m scripts.check_abstractions --check`` validates the registry (the default).
- ``uv run python -m scripts.check_abstractions --list`` prints the capabilities.
- ``uv run python -m scripts.check_abstractions --search graph text`` finds capabilities and symbols by word.

``--format json`` prints the diagnostics or the hits as JSON. ``--search`` is a lexical search. A search without a hit
does not prove that no equivalent exists, so read the owner modules too.
"""

import difflib
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import attrs
import cyclopts
import orjson

from scripts.abstractions_index import SourceIndex, build_index
from scripts.abstractions_registry import (
    ACTIVE,
    CLOSE_MATCH_COUNT,
    DEPRECATED,
    ID_PATTERN,
    MIN_REASON_CHARACTERS,
    MIN_REASON_WORDS,
    REGISTRY_PATH,
    ROOT,
    STATUSES,
    Capability,
    Diagnostic,
    OverlapException,
    Registry,
    load_registry,
)
from scripts.check_layout import layer_names


@attrs.frozen
class SearchHit:
    """One result of ``--search``.

    Parameters
    ----------
    kind: str
        ``capability`` for a registered entry and ``symbol`` for another public symbol.
    name: str
        The ID of the capability, or the ``module:Name`` of the symbol.
    owner: str
        The owner of the capability, or the module of the symbol.
    path: str
        The file of the canonical symbol.
    line: int
        The line of the definition.
    score: int
        The number of query words that matched.
    imported_by: str
        The layers that may import the hit.

    """

    kind: str
    name: str
    owner: str
    path: str
    line: int
    score: int
    imported_by: str

    def render(self) -> str:
        """Format the hit as one text line.

        Returns
        -------
        str
            The kind, the name, the place, and the layers that may import it.

        """
        return f"{self.kind}: {self.name} ({self.path}:{self.line}) owner {self.owner}; {self.imported_by}"


def normalize_responsibility(responsibility: str) -> str:
    """Reduce a responsibility to lower-case words, so that a copy with other punctuation matches.

    The reduction is syntactic. It does not prove that two sentences mean the same.

    Parameters
    ----------
    responsibility: str
        The sentence of the entry.

    Returns
    -------
    str
        The words in lower case, separated by single spaces.

    """
    words: list[str] = re.findall(r"[a-z0-9]+", responsibility.lower())
    return " ".join(words)


def _close_names(index: SourceIndex, module: str, name: str) -> str:
    """Name the public symbols of a module that look like a missing one.

    Parameters
    ----------
    index: SourceIndex
        The source index.
    module: str
        The module of the missing symbol.
    name: str
        The missing name.

    Returns
    -------
    str
        Up to three close ``module:Name`` values, comma-separated, or an empty string.

    """
    names = [symbol.name for symbol in index.symbols.values() if symbol.module == module]
    return ", ".join(f"{module}:{close}" for close in difflib.get_close_matches(name, names, n=CLOSE_MATCH_COUNT))


def _inside(module: str, owner: str) -> bool:
    """Tell whether a module is the owner or lies below it.

    Parameters
    ----------
    module: str
        The dotted module.
    owner: str
        The dotted owner.

    Returns
    -------
    bool
        True if ``module`` equals ``owner`` or starts with ``owner.``.

    """
    return module == owner or module.startswith(f"{owner}.")


def _capability_problems(capability: Capability, index: SourceIndex, path: str) -> list[Diagnostic]:
    """Check one capability against the source, without looking at the other capabilities.

    Parameters
    ----------
    capability: Capability
        The entry.
    index: SourceIndex
        The source index.
    path: str
        The registry file.

    Returns
    -------
    list[Diagnostic]
        The findings. The list is empty if the entry is right.

    """
    identity = capability.capability_id
    found: list[Diagnostic] = []
    if not ID_PATTERN.fullmatch(identity):
        found.append(
            Diagnostic("invalid-id", path, capability.line, identity, "", "write the ID in lower-case kebab-case"),
        )
    if capability.status not in STATUSES:
        found.append(
            Diagnostic("invalid-status", path, capability.line, identity, capability.status, f"use one of {STATUSES}"),
        )
    if capability.owner not in index.modules:
        found.append(
            Diagnostic(
                "unknown-owner",
                path,
                capability.line,
                identity,
                capability.owner,
                "name a module or package of src/ebaml",
            ),
        )
    found.extend(
        Diagnostic(
            "unknown-consumer",
            path,
            capability.line,
            identity,
            consumer,
            "name a module or package of src/ebaml",
        )
        for consumer in capability.consumers
        if consumer not in index.modules
    )
    module, _, name = capability.symbol.partition(":")
    if not name:
        found.append(
            Diagnostic(
                "invalid-symbol",
                path,
                capability.line,
                identity,
                capability.symbol,
                "write the symbol as module:Name",
            ),
        )
    elif capability.symbol not in index.symbols:
        close = _close_names(index, module, name)
        hint = f"did you mean {close}?" if close else "the symbol was renamed or deleted: fix or remove the entry"
        found.append(Diagnostic("missing-symbol", path, capability.line, identity, capability.symbol, hint))
    elif not _inside(module, capability.owner):
        found.append(
            Diagnostic(
                "symbol-owner-mismatch",
                path,
                capability.line,
                identity,
                module,
                f"move the symbol into {capability.owner} or change the owner",
            ),
        )
    if capability.status == DEPRECATED and not capability.replaced_by:
        found.append(
            Diagnostic(
                "missing-replacement",
                path,
                capability.line,
                identity,
                "",
                "add `replaced_by` with the ID of the replacement",
            ),
        )
    return found


def _first_claim[K](seen: dict[K, Capability], key: K, capability: Capability) -> Capability | None:
    """Record a claim, and return the earlier entry that made the same claim.

    Parameters
    ----------
    seen: dict[K, Capability]
        The first entry for each key.
    key: K
        What the capability claims.
    capability: Capability
        The entry that makes the claim.

    Returns
    -------
    Capability | None
        The earlier entry, or ``None`` if this entry is the first.

    """
    first = seen.setdefault(key, capability)
    return None if first is capability else first


def _collision_problems(capabilities: Sequence[Capability], path: str) -> list[Diagnostic]:
    """Find the claims that two entries make on the same thing.

    Parameters
    ----------
    capabilities: Sequence[Capability]
        Every entry of the registry.
    path: str
        The registry file.

    Returns
    -------
    list[Diagnostic]
        One finding for each later entry that repeats an ID, a symbol or a responsibility, and for each unknown
        replacement.

    """
    found: list[Diagnostic] = []
    by_id: dict[str, Capability] = {}
    by_symbol: dict[str, Capability] = {}
    by_duty: dict[tuple[str, str, str], Capability] = {}
    for capability in capabilities:
        identity = capability.capability_id
        if identity in by_id:
            found.append(
                Diagnostic(
                    "duplicate-id",
                    path,
                    capability.line,
                    identity,
                    f"{path}:{by_id[identity].line}",
                    "keep one entry for the ID",
                ),
            )
        by_id.setdefault(identity, capability)
        if capability.status != ACTIVE:
            continue
        duty = (capability.owner, normalize_responsibility(capability.responsibility), capability.context)
        for rule, earlier in (
            ("duplicate-symbol", _first_claim(by_symbol, capability.symbol, capability)),
            ("duplicate-responsibility", _first_claim(by_duty, duty, capability)),
        ):
            if earlier:
                found.append(
                    Diagnostic(
                        rule,
                        path,
                        capability.line,
                        identity,
                        earlier.capability_id,
                        f"reuse {earlier.capability_id} or mark one entry deprecated",
                    ),
                )
    found.extend(
        Diagnostic(
            "unknown-replacement",
            path,
            capability.line,
            capability.capability_id,
            capability.replaced_by,
            "name an existing ID",
        )
        for capability in capabilities
        if capability.replaced_by and capability.replaced_by not in by_id
    )
    return found


def _exception_problems(exceptions: Sequence[OverlapException], index: SourceIndex, path: str) -> list[Diagnostic]:
    """Check the approved overlaps.

    Parameters
    ----------
    exceptions: Sequence[OverlapException]
        The exceptions of the registry.
    index: SourceIndex
        The source index.
    path: str
        The registry file.

    Returns
    -------
    list[Diagnostic]
        A finding for each exception with a short reason or with a symbol that does not exist.

    """
    found: list[Diagnostic] = []
    for exception in exceptions:
        words = exception.reason.split()
        if len(words) < MIN_REASON_WORDS or len(exception.reason.strip()) < MIN_REASON_CHARACTERS:
            found.append(
                Diagnostic(
                    "exception-reason",
                    path,
                    exception.line,
                    "",
                    " + ".join(exception.symbols),
                    "say why the contracts stay apart, in at least four words",
                ),
            )
        found.extend(
            Diagnostic(
                "exception-missing-symbol",
                path,
                exception.line,
                "",
                symbol,
                "the contract was renamed or deleted: remove the exception",
            )
            for symbol in exception.symbols
            if symbol not in index.symbols
        )
    return found


def registry_diagnostics(
    registry: Registry,
    index: SourceIndex,
    path: str = REGISTRY_PATH.as_posix(),
) -> list[Diagnostic]:
    """Check a parsed registry against the source.

    Parameters
    ----------
    registry: Registry
        The parsed registry.
    index: SourceIndex
        The source index.
    path: str
        The registry file for the diagnostics.

    Returns
    -------
    list[Diagnostic]
        Every finding, in registry order.

    """
    found: list[Diagnostic] = []
    for capability in registry.capabilities:
        found.extend(_capability_problems(capability, index, path))
    found.extend(_collision_problems(registry.capabilities, path))
    found.extend(_exception_problems(registry.exceptions, index, path))
    return found


def check(root: Path) -> list[Diagnostic]:
    """Run every check of the registry.

    Parameters
    ----------
    root: Path
        The repo root.

    Returns
    -------
    list[Diagnostic]
        Every finding. The list is empty if the registry is right.

    """
    registry, problems = load_registry(root)
    return problems + registry_diagnostics(registry, build_index(root))


def importable_by(owner: str, layers: Sequence[str]) -> str:
    """Say which layers may import a module, from the order of the layers contract.

    Parameters
    ----------
    owner: str
        The dotted module, for example ``ebaml.graphs.terms``.
    layers: Sequence[str]
        The layer names from the highest to the lowest, as ``scripts.check_layout.layer_names`` returns them.

    Returns
    -------
    str
        A sentence about the layers that may import the module.

    """
    parts = owner.split(".")
    layer = parts[1] if len(parts) > 1 else ""
    if layer not in layers:
        return "imported by no other layer (it stands alone)"
    allowed = layers[: layers.index(layer) + 1]
    return f"importable from {', '.join(allowed)}"


def _words(text: str) -> set[str]:
    """Split a text into lower-case words, with snake_case and CamelCase names cut apart.

    Parameters
    ----------
    text: str
        Any text or name.

    Returns
    -------
    set[str]
        The lower-case words.

    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return set(re.findall(r"[a-z0-9]+", spaced.lower()))


def search(registry: Registry, index: SourceIndex, query: Sequence[str], layers: Sequence[str]) -> list[SearchHit]:
    """Find capabilities and public symbols by words.

    Parameters
    ----------
    registry: Registry
        The parsed registry.
    index: SourceIndex
        The source index.
    query: Sequence[str]
        The words to look for.
    layers: Sequence[str]
        The layer names from the highest to the lowest.

    Returns
    -------
    list[SearchHit]
        The hits with at least one matching word. Capabilities come first, then the best score, then the name.

    """
    wanted = {word for item in query for word in _words(item)}
    hits: list[SearchHit] = []
    registered = set()
    for capability in registry.capabilities:
        registered.add(capability.symbol)
        text = " ".join([capability.capability_id, capability.responsibility, capability.symbol, *capability.keywords])
        score = len(wanted & _words(text))
        symbol = index.symbols.get(capability.symbol)
        if score:
            where = (symbol.path, symbol.line) if symbol else (REGISTRY_PATH.as_posix(), capability.line)
            hits.append(
                SearchHit(
                    "capability",
                    capability.capability_id,
                    capability.owner,
                    where[0],
                    where[1],
                    score,
                    importable_by(capability.owner, layers),
                ),
            )
    for reference, symbol in index.symbols.items():
        score = len(wanted & _words(symbol.name))
        if score and reference not in registered:
            hits.append(
                SearchHit(
                    "symbol",
                    reference,
                    symbol.module,
                    symbol.path,
                    symbol.line,
                    score,
                    importable_by(symbol.module, layers),
                ),
            )
    return sorted(hits, key=lambda hit: (hit.kind != "capability", -hit.score, hit.name))


def _read_layers(root: Path) -> list[str]:
    """Read the layer names from ``pyproject.toml``.

    Parameters
    ----------
    root: Path
        The repo root.

    Returns
    -------
    list[str]
        The layers from the highest to the lowest.

    """
    return layer_names(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")))


_MODE = cyclopts.Group("Mode", validator=cyclopts.validators.MutuallyExclusive())


app: cyclopts.App = cyclopts.App(
    name="check_abstractions",
    help=__doc__.splitlines()[0],
    result_action="return_value",
)


def emit(rows: Sequence[Any], output_format: str, empty_text: str) -> None:
    """Write findings or hits to standard output.

    Parameters
    ----------
    rows: Sequence[Any]
        Attrs objects. Each has a ``render`` method.
    output_format: str
        ``text`` or ``json``.
    empty_text: str
        The line to print for no rows in the text format.

    """
    if output_format == "json":
        sys.stdout.write(orjson.dumps([attrs.asdict(row) for row in rows], option=orjson.OPT_INDENT_2).decode() + "\n")
    else:
        sys.stdout.write("\n".join(row.render() for row in rows) + "\n" if rows else f"{empty_text}\n")


@app.default
def run(
    *,
    check: Annotated[bool, cyclopts.Parameter(group=_MODE)] = False,
    list_capabilities: Annotated[bool, cyclopts.Parameter(name="--list", group=_MODE)] = False,
    search_words: Annotated[
        list[str] | None,
        cyclopts.Parameter(name="--search", group=_MODE, consume_multiple=True, negative_iterable=()),
    ] = None,
    output_format: Annotated[Literal["text", "json"], cyclopts.Parameter(name="--format")] = "text",
    root: Path = ROOT,
) -> int:
    """Check the registry, list its capabilities or search it.

    Parameters
    ----------
    check: bool
        Validate the registry. This is the default mode.
    list_capabilities: bool
        Print the registered capabilities.
    search_words: list[str], optional
        Find capabilities and symbols by word.
    output_format: Literal["text", "json"]
        The output format.
    root: Path
        The repo root. The default is this repository.

    Returns
    -------
    int
        The process exit code: 0 if the registry is right or the command only lists, 1 if the check found a problem.

    """
    del check  # the flag only names the default mode
    registry, shape_problems = load_registry(root)
    index = build_index(root)
    if list_capabilities:
        listing: defaultdict[str, list[Capability]] = defaultdict(list)
        for capability in registry.capabilities:
            listing[capability.owner].append(capability)
        lines = [
            f"{owner}: {capability.capability_id} ({capability.status}) {capability.symbol} -"
            f" {capability.responsibility}"
            for owner, capabilities in sorted(listing.items())
            for capability in capabilities
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        return 0
    if search_words:
        emit(
            search(registry, index, search_words, _read_layers(root)),
            output_format,
            "no match: read the owner modules before you add a shared contract",
        )
        return 0
    problems = shape_problems + registry_diagnostics(registry, index)
    emit(problems, output_format, "the registry of shared abstractions is right")
    return 1 if problems else 0


def main(argv: Sequence[str] = ()) -> int:
    """Run the command line.

    Parameters
    ----------
    argv: Sequence[str]
        The arguments. The default is none, which runs ``--check`` on this repository.

    Returns
    -------
    int
        The process exit code: 0 if the registry is right or the command only lists, 1 if the check found a problem.

    """
    return int(app(list(argv)))
