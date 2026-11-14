"""Tests of the save tool: the confirmation of a person, the output directory and the races."""

from pathlib import Path

import anyio
import orjson
import pytest
import skops.io
from mcp import Client, types
from mcp.client import ClientRequestContext
from mcp.server.mcpserver import MCPServer
from mcp_types import TextContent

from ebaml.core.exceptions import EditError, StaleRevisionError
from ebaml.edit.edits import SetScores
from ebaml.edit.selections import Bins
from ebaml.mcp import save_tools
from ebaml.mcp.model_store import ModelStore
from ebaml.mcp.server import create_server
from tests.unit.ebaml.conftest import FittedClassifier, as_frame

LOW_BINS: dict[str, object] = {
    "kind": "Shift",
    "feature": "a",
    "selection": {"kind": "Bins", "first": 0, "last": 1},
    "delta": 0.5,
}
# the SDK has two ways to ask a person: a standalone request in the handshake protocols, and a retry round of the
# tool call in the modern protocol (2026-07-28), so each test of the elicitation runs in both
MODES = ["legacy", "auto"]


def _store(continuous_ebm: FittedClassifier, output: Path) -> ModelStore:
    ebm, data = continuous_ebm
    return ModelStore(ebm=ebm, data=as_frame(data), output_directory=output)


def _edit(store: ModelStore, reason: str, delta: float = 0.5) -> str:
    store.editing().apply(SetScores("a", Bins(0, 1), delta), reason)
    return store.editing().revision


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _exists(path: Path) -> bool:
    return path.exists()


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _save(store: ModelStore, message: str, *, confirmed: bool | None) -> save_tools.SaveResult:
    return save_tools.save_model(store, message, save_tools.current_head(store), confirmed=confirmed)


def test_the_question_lists_the_message_the_head_and_each_edit_with_its_reason(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    head = _edit(store, "Low values are rare")
    _edit(store, "The model must not drop for small a", -0.25)
    text = save_tools.question(store, "Cap the lowest bins", save_tools.current_head(store), None)
    assert text is not None
    assert "Cap the lowest bins" in text
    assert "Edits since the last save (2)" in text
    assert "1. Set 2 bins of a" in text
    assert "Reason: Low values are rare" in text
    assert "2. Set 2 bins of a" in text
    assert "Reason: The model must not drop for small a" in text
    assert head not in text
    assert f"revision {save_tools.current_head(store)}" in text


def test_the_question_is_empty_when_no_edit_came_after_the_last_save(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    assert save_tools.question(store, "Keep it", "original", None) is None
    _edit(store, "Low values are rare")
    _save(store, "Cap", confirmed=True)
    assert save_tools.question(store, "Save again", save_tools.current_head(store), None) is None


def test_the_question_fails_before_the_person_is_asked_when_the_call_cannot_go_on(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    ebm, _ = continuous_ebm
    store = _store(continuous_ebm, tmp_path)
    head = _edit(store, "Low values are rare")
    with pytest.raises(EditError, match="A message is needed"):
        save_tools.question(store, "  ", head, None)
    with pytest.raises(EditError, match="no output directory"):
        save_tools.question(ModelStore(ebm=ebm), "Save", "original", None)
    with pytest.raises(StaleRevisionError, match="expected 'original'"):
        save_tools.question(store, "Save", head, "original")
    with pytest.raises(StaleRevisionError, match="The head of the session is"):
        save_tools.question(store, "Save", "original", None)
    assert save_tools.question(store, "Save", head, head) is not None
    assert save_tools.question(store, "Save", head, "current") is not None


def test_a_confirmed_save_writes_the_model_and_the_history_and_marks_the_edits_reviewed(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    _edit(store, "Low values are rare")
    result = _save(store, "Cap the lowest bins", confirmed=True)
    assert result.confirmation == "person_confirmed"
    assert result.model_file == "edited-model-001.skops"
    assert result.history_file == "edited-model-001.skops.history.json"
    assert _names(tmp_path) == [result.model_file, result.history_file]
    assert (result.commit.kind, result.commit.reason, result.commit.reviewed) == ("save", "Cap the lowest bins", True)
    assert [(row.reason, row.reviewed) for row in result.edits_saved] == [("Low values are rare", True)]
    history = orjson.loads((tmp_path / result.history_file).read_bytes())
    assert [entry["reviewed"] for entry in history["entries"]] == [True, True]
    file = tmp_path / result.model_file
    saved = skops.io.load(file, trusted=skops.io.get_untrusted_types(file=file))
    assert saved.term_scores_[0].tolist() == store.read().model.term_scores_[0].tolist()


def test_a_refusal_saves_nothing_and_leaves_the_edits_unreviewed(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    _edit(store, "Low values are rare")
    with pytest.raises(EditError, match="did not confirm the save, so nothing was saved"):
        _save(store, "Cap the lowest bins", confirmed=False)
    assert _names(tmp_path) == []
    assert [(row.kind, row.reviewed) for row in store.editing().history_table()] == [("edit", False)]


def test_a_client_that_cannot_ask_gets_a_save_and_an_unreviewed_history(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    _edit(store, "Low values are rare")
    result = _save(store, "Cap the lowest bins", confirmed=None)
    assert result.confirmation == "client_cannot_ask"
    assert not result.commit.reviewed
    history = orjson.loads((tmp_path / result.history_file).read_bytes())
    assert [entry["reviewed"] for entry in history["entries"]] == [False, False]


def test_a_save_with_no_edit_since_the_last_save_needs_no_confirmation(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    store = _store(continuous_ebm, tmp_path)
    first = _save(store, "Keep the model as it is", confirmed=None)
    assert (first.confirmation, first.commit.reviewed, first.edits_saved) == ("no_edits_to_confirm", True, [])


def test_each_save_gets_a_new_name_and_never_overwrites_a_file(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    (tmp_path / "edited-model-002.skops").write_bytes(b"a file that the user keeps")
    store = _store(continuous_ebm, tmp_path)
    names = [_save(store, f"Save {number}", confirmed=None).model_file for number in range(3)]
    assert names == ["edited-model-003.skops", "edited-model-004.skops", "edited-model-005.skops"]
    assert (tmp_path / "edited-model-002.skops").read_bytes() == b"a file that the user keeps"


def test_a_save_takes_the_next_name_when_a_history_file_blocks_the_name(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    (tmp_path / "edited-model-001.skops.history.json").write_bytes(b"a history that the user keeps")
    store = _store(continuous_ebm, tmp_path)
    result = _save(store, "Save", confirmed=None)
    assert result.model_file == "edited-model-002.skops"
    assert (tmp_path / "edited-model-001.skops.history.json").read_bytes() == b"a history that the user keeps"
    assert not (tmp_path / "edited-model-001.skops").exists()


def test_the_save_needs_a_message_and_an_output_directory(continuous_ebm: FittedClassifier, tmp_path: Path) -> None:
    ebm, _ = continuous_ebm
    store = _store(continuous_ebm, tmp_path)
    with pytest.raises(EditError, match="A message is needed"):
        _save(store, "  ", confirmed=None)
    with pytest.raises(EditError, match="no output directory"):
        save_tools.save_model(ModelStore(ebm=ebm), "Save", "original", confirmed=None)
    assert _names(tmp_path) == []


def test_a_confirmation_of_an_older_revision_never_saves_a_newer_one(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    """Contract: the person confirmed the edits of the question, so a save of a head that moved must fail."""
    store = _store(continuous_ebm, tmp_path)
    asked_head = _edit(store, "Low values are rare")
    _edit(store, "An edit while the person decides", 0.25)
    with pytest.raises(StaleRevisionError, match="The head of the session is"):
        save_tools.save_model(store, "Save", asked_head, confirmed=True)
    assert _names(tmp_path) == []
    assert not any(row.reviewed for row in store.editing().history_table())


async def _text(server: MCPServer, name: str, arguments: dict[str, object], **client: object) -> tuple[bool, str]:
    async with Client(server, **client) as session:  # pyrefly: ignore[bad-argument-type] -- the test passes the client options of the SDK
        result = await session.call_tool(name, arguments)
    (content,) = result.content
    assert isinstance(content, TextContent)
    return result.is_error, content.text


class Person:
    """The elicitation callback of a client. It answers in order, and it may run a step before it answers."""

    def __init__(self, *answers: types.ElicitResult) -> None:
        """Keep the answers, in the order that the person gives them."""
        self.answers: list[types.ElicitResult] = list(answers)
        self.questions: list[str] = []

    async def __call__(
        self,
        _context: ClientRequestContext,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Keep the question and give the next answer.

        Returns
        -------
        types.ElicitResult | types.ErrorData
            The next answer.

        """
        assert isinstance(params, types.ElicitRequestFormParams)
        assert params.requested_schema["required"] == ["confirm"]
        self.questions.append(params.message)
        return self.answers.pop(0)


YES: types.ElicitResult = types.ElicitResult(action="accept", content={"confirm": True})


@pytest.mark.anyio
@pytest.mark.parametrize("mode", MODES)
async def test_the_save_tool_asks_the_person_through_the_elicitation_of_the_client(
    mode: str,
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})
    person = Person(YES)
    is_error, text = await _text(
        server,
        "save_model",
        {"message": "Cap the lowest bins"},
        elicitation_callback=person,
        mode=mode,
    )
    result = orjson.loads(text)
    assert not is_error
    assert result["confirmation"] == "person_confirmed"
    assert result["commit"]["reviewed"]
    assert len(person.questions) == 1
    assert "Low values are rare" in person.questions[0]
    assert _exists(tmp_path / result["model_file"])
    listing = orjson.loads((await _text(server, "list_edits", {}))[1])
    assert [commit["reviewed"] for commit in listing["commits"]] == [True, True]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "answer",
    [
        types.ElicitResult(action="accept", content={"confirm": False}),
        types.ElicitResult(action="decline"),
        types.ElicitResult(action="cancel"),
    ],
)
async def test_the_save_tool_saves_nothing_unless_the_person_says_yes(
    mode: str,
    answer: types.ElicitResult,
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})
    is_error, text = await _text(
        server,
        "save_model",
        {"message": "Cap the lowest bins"},
        elicitation_callback=Person(answer),
        mode=mode,
    )
    assert is_error
    assert _names(tmp_path) == []
    assert "did not confirm the save" in text or "elicitation was" in text


@pytest.mark.anyio
@pytest.mark.parametrize("mode", MODES)
async def test_the_save_tool_of_a_client_without_elicitation_marks_the_edits_unreviewed(
    mode: str,
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})
    is_error, text = await _text(server, "save_model", {"message": "Cap the lowest bins"}, mode=mode)
    result = orjson.loads(text)
    assert not is_error
    assert result["confirmation"] == "client_cannot_ask"
    history = orjson.loads(_read(tmp_path / result["history_file"]))
    assert [entry["reviewed"] for entry in history["entries"]] == [False, False]
    listing = orjson.loads((await _text(server, "list_edits", {}))[1])
    assert [commit["reviewed"] for commit in listing["commits"]] == [False, False]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", MODES)
async def test_a_save_with_no_edit_to_confirm_asks_nobody(
    mode: str,
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    person = Person()
    is_error, text = await _text(server, "save_model", {"message": "Keep"}, elicitation_callback=person, mode=mode)
    assert not is_error
    assert orjson.loads(text)["confirmation"] == "no_edits_to_confirm"
    assert person.questions == []


@pytest.mark.anyio
async def test_an_edit_while_the_person_decides_fails_the_save_in_the_handshake_protocols(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    """The standalone request waits inside the call, so the save checks the head again after the answer."""
    store = _store(continuous_ebm, tmp_path)
    server = create_server(store)
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})

    class SlowPerson(Person):
        async def __call__(
            self,
            context: ClientRequestContext,
            params: types.ElicitRequestParams,
        ) -> types.ElicitResult | types.ErrorData:
            _edit(store, "An edit while the person decides", 0.25)
            return await super().__call__(context, params)

    is_error, text = await _text(
        server,
        "save_model",
        {"message": "Cap"},
        elicitation_callback=SlowPerson(YES),
        mode="legacy",
    )
    assert is_error
    assert "The head of the session is" in text
    assert _names(tmp_path) == []
    assert not any(row.reviewed for row in store.editing().history_table())


@pytest.mark.anyio
async def test_an_edit_between_the_rounds_makes_the_person_answer_the_new_question(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    """In the modern protocol an answer counts only for the question that was shown, so a moved head asks again."""
    store = _store(continuous_ebm, tmp_path)
    server = create_server(store)
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})

    class SlowPerson(Person):
        async def __call__(
            self,
            context: ClientRequestContext,
            params: types.ElicitRequestParams,
        ) -> types.ElicitResult | types.ErrorData:
            if not self.questions:
                _edit(store, "An edit while the person decides", 0.25)
            return await super().__call__(context, params)

    person = SlowPerson(YES, YES)
    is_error, text = await _text(server, "save_model", {"message": "Cap"}, elicitation_callback=person, mode="auto")
    assert not is_error
    assert len(person.questions) == 2
    assert "An edit while the person decides" not in person.questions[0]
    assert "An edit while the person decides" in person.questions[1]
    assert [row["reviewed"] for row in orjson.loads(text)["edits_saved"]] == [True, True]


@pytest.mark.anyio
async def test_the_save_tool_exists_only_with_an_output_directory_and_takes_no_path(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    ebm, _ = continuous_ebm
    async with Client(create_server(ModelStore(ebm=ebm))) as session:
        without = {tool.name for tool in (await session.list_tools()).tools}
    server = create_server(_store(continuous_ebm, tmp_path))
    async with Client(server) as session:
        save = next(tool for tool in (await session.list_tools()).tools if tool.name == "save_model")
    assert "save_model" not in without
    assert set(save.input_schema["properties"]) == {"message", "expected_revision"}
    outside = tmp_path.parent / "outside.skops"
    escape = await _text(server, "save_model", {"message": "x", "path": str(outside)})
    assert not escape[0]
    assert _names(tmp_path) == [
        "edited-model-001.skops",
        "edited-model-001.skops.history.json",
    ]
    assert not _exists(outside)


@pytest.mark.anyio
async def test_the_save_tool_reports_a_stale_head_and_a_bad_message_without_asking(
    continuous_ebm: FittedClassifier,
    tmp_path: Path,
) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    await _text(server, "apply_edit", {"edit": LOW_BINS, "reason": "Low values are rare"})
    person = Person()
    stale = await _text(
        server,
        "save_model",
        {"message": "Cap", "expected_revision": "original"},
        elicitation_callback=person,
    )
    empty = await _text(server, "save_model", {"message": " "}, elicitation_callback=person)
    assert stale[0]
    assert "expected 'original'" in stale[1]
    assert empty[0]
    assert "A message is needed" in empty[1]
    assert person.questions == []
    assert _names(tmp_path) == []


@pytest.mark.anyio
async def test_two_saves_at_once_get_two_names(continuous_ebm: FittedClassifier, tmp_path: Path) -> None:
    server = create_server(_store(continuous_ebm, tmp_path))
    names: list[str] = []

    async def save(number: int) -> None:
        _, text = await _text(server, "save_model", {"message": f"Save {number}"})
        names.append(orjson.loads(text)["model_file"])

    async with anyio.create_task_group() as group:
        group.start_soon(save, 1)
        group.start_soon(save, 2)
    assert sorted(names) == ["edited-model-001.skops", "edited-model-002.skops"]
    assert len([name for name in _names(tmp_path) if name.endswith(".history.json")]) == 2
