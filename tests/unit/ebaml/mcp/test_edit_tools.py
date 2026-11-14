"""Tests of the edit tools, called as functions on a store."""

import orjson
import pytest

from ebaml.core.custom_types import JSONMapping
from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import EditError, HistoryError, SelectionError, StaleRevisionError, UnsupportedModelError
from ebaml.edit.edit_session import RevisionInfo
from ebaml.edit.edits import Monotonize, RemoveTerms, SetScores, Shift
from ebaml.edit.model_view import term_view
from ebaml.edit.selections import Bins, Categories, ValueInterval
from ebaml.mcp import edit_tools
from ebaml.mcp.model_store import ModelStore
from tests.unit.ebaml.conftest import FittedClassifier, as_frame

RAISE_LOW_BINS: JSONMapping = {
    "kind": "Shift",
    "feature": "a",
    "selection": {"kind": "Bins", "first": 0, "last": 1},
    "delta": 0.5,
}


@pytest.fixture
def store(continuous_ebm: FittedClassifier) -> ModelStore:
    ebm, data = continuous_ebm
    return ModelStore(ebm=ebm, data=as_frame(data))


def test_list_bins_gives_the_number_the_edges_the_weight_and_the_score_of_each_real_bin(store: ModelStore) -> None:
    listing = edit_tools.list_bins(store, "a")
    view = term_view(store.read().model, "a")
    assert listing.feature_type == "continuous"
    assert [info.number for info in listing.bins] == list(range(view.real_bin_count))
    assert [info.score for info in listing.bins] == view.scores[1:-1].tolist()
    assert [info.weight for info in listing.bins] == view.weights[1:-1].tolist()
    assert sum(
        info.weight_share for info in listing.bins
    ) + listing.missing.weight_share + listing.unseen.weight_share == (pytest.approx(1.0))
    assert [(info.left, info.right) for info in listing.bins] == view.bin_edges()
    assert all(info.label == view.label(info.number + 1) for info in listing.bins if info.number is not None)
    assert listing.missing.label == "missing"
    assert listing.unseen.label == "unseen"
    assert listing.missing.number is None
    assert listing.unseen.number is None
    assert listing.missing.left is None
    assert listing.pair_terms_using_feature == []
    assert listing.session == RevisionInfo("original", "original", 0)


def test_list_bins_of_a_nominal_feature_names_the_categories_and_has_no_edges(store: ModelStore) -> None:
    listing = edit_tools.list_bins(store, "c")
    assert [info.label for info in listing.bins] == ["x", "y"]
    assert [info.number for info in listing.bins] == [0, 1]
    assert all(info.left is None and info.right is None for info in listing.bins)


def test_list_bins_reads_the_revision_that_the_caller_names(store: ModelStore) -> None:
    before = edit_tools.list_bins(store, "a")
    edit_tools.apply_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "Raise the lowest bins")
    after = edit_tools.list_bins(store, "a")
    assert after.bins[0].score == pytest.approx(before.bins[0].score + 0.5)
    earlier = edit_tools.list_bins(store, "a", "original")
    assert earlier.bins == before.bins
    assert earlier.session == RevisionInfo("original", after.session.head, 1)


def test_list_bins_names_the_pair_terms_that_use_the_feature(interaction_ebm: ExplainableBoostingClassifier) -> None:
    listing = edit_tools.list_bins(ModelStore(ebm=interaction_ebm), "p")
    assert listing.pair_terms_using_feature == ["p & q"]


def test_list_bins_of_an_unknown_feature_lists_the_features(store: ModelStore) -> None:
    with pytest.raises(SelectionError, match=r"no feature 'zzz'. The features are \['a', 'b', 'c'\]"):
        edit_tools.list_bins(store, "zzz")


def test_the_edit_tools_refuse_a_model_that_the_edits_do_not_support(
    multiclass_ebm: ExplainableBoostingClassifier,
) -> None:
    store = ModelStore(ebm=multiclass_ebm)
    calls = [
        lambda: edit_tools.list_bins(store, "a"),
        lambda: edit_tools.preview_edit(store, Shift("a", Bins(0, 0), 1.0)),
        lambda: edit_tools.apply_edit(store, Shift("a", Bins(0, 0), 1.0), "Reason"),
        lambda: edit_tools.undo(store),
        lambda: edit_tools.redo(store),
        lambda: edit_tools.checkout(store, "original"),
        lambda: edit_tools.list_edits(store),
    ]
    for call in calls:
        with pytest.raises(UnsupportedModelError, match="binary classifiers and regressors"):
            call()


def test_parse_edit_builds_the_edit_and_its_selection_from_the_kind_fields() -> None:
    assert edit_tools.parse_edit(RAISE_LOW_BINS) == Shift("a", Bins(0, 1), 0.5)
    assert edit_tools.parse_edit({"kind": "Monotonize", "feature": "a", "direction": "increasing"}) == Monotonize(
        "a",
        "increasing",
    )
    assert edit_tools.parse_edit({"kind": "RemoveTerms", "terms": ["a", "b"]}) == RemoveTerms(("a", "b"))
    assert edit_tools.parse_edit({
        "kind": "SetScores",
        "feature": "c",
        "selection": {"kind": "Categories", "names": ["x"]},
        "value": 0,
    }) == SetScores("c", Categories(("x",)), 0.0)
    assert edit_tools.parse_edit({
        "kind": "SetScores",
        "feature": "a",
        "selection": {"kind": "ValueInterval", "low": 0.5, "high": None},
        "value": 1.5,
    }) == SetScores("a", ValueInterval(0.5, None), 1.5)


def test_parse_edit_turns_the_whole_numbers_that_the_sdk_sends_as_floats_into_ints() -> None:
    """Bug: the SDK reads each JSON number as a float, so ``Bins(first=0)`` arrived as ``first=0.0`` and failed."""
    sent: JSONMapping = {
        "kind": "Shift",
        "feature": "a",
        "selection": {"kind": "Bins", "first": 0.0, "last": 3.0},
        "delta": 2.0,
    }
    edit = edit_tools.parse_edit(sent)
    assert edit == Shift("a", Bins(0, 3), 2.0)
    assert isinstance(edit, Shift)
    assert isinstance(edit.selection, Bins)
    assert type(edit.selection.first) is int
    with pytest.raises(EditError, match="violates type hint"):
        edit_tools.parse_edit({**sent, "selection": {"kind": "Bins", "first": 0.5, "last": 3.0}})


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"feature": "a"}, r"edit needs a field 'kind' with one of \[.*'Shift'.*\], but it has None"),
        ({"kind": "Explode"}, r"but it has 'Explode'"),
        (
            {"kind": "Shift", "feature": "a", "selection": {"kind": "Everything"}, "delta": 1},
            r"selection needs a field 'kind' with one of \[.*'Bins'.*\], but it has 'Everything'",
        ),
        ({"kind": "Shift", "feature": "a", "delta": 1}, "does not fit its class"),
        ({"kind": "Shift", "feature": "a", "selection": {"kind": "Bins", "first": 0}, "delta": 1}, "does not fit"),
        ({"kind": "Shift", "feature": "a", "selection": {"kind": "Bins", "first": 0, "last": 1}, "delta": "up"}, "fit"),
    ],
)
def test_parse_edit_names_what_is_known_when_the_object_does_not_fit(arguments: JSONMapping, message: str) -> None:
    with pytest.raises(EditError, match=message) as caught:
        edit_tools.parse_edit(arguments)
    assert isinstance(caught.value, EditError)
    assert not isinstance(caught.value, HistoryError) or "does not fit" in str(caught.value)


def test_the_help_lists_every_edit_and_every_selection_with_its_fields() -> None:
    text = edit_tools.edit_kinds_help()
    for name in (
        "Shift",
        "SetScores",
        "Align",
        "Monotonize",
        "Interpolate",
        "FitLine",
        "Scale",
        "ZeroTerm",
        "Recenter",
    ):
        assert f"kind {name}(" in text
    for name in ("RemoveTerms", "RemoveFeature", "Bins", "Categories", "MissingBin", "UnseenBin", "WholeGraph"):
        assert f"kind {name}(" in text
    assert "kind Shift(feature: str, selection: selection, delta: float): Add one value to each selected score" in text
    assert "kind Monotonize(feature: str, direction: Literal['increasing', 'decreasing', 'auto']," in text
    assert "selection: selection = whole graph)" in text
    assert "kind Bins(first: int, last: int)" in text
    assert "kind RemoveTerms(terms: tuple[str, ...])" in text
    assert 'Example: {"kind": "Shift"' in text


def test_preview_changes_nothing_and_counts_the_rows_that_the_edit_reaches(store: ModelStore) -> None:
    result = edit_tools.preview_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS))
    assert not result.applied
    assert result.session == RevisionInfo("original", "original", 0)
    assert result.report.revision is None
    assert result.report.bins_changed == 2
    assert result.report_text == result.report.to_text()
    assert result.effect_on_data is not None
    assert result.effect_on_data.row_count == 600
    assert result.effect_on_data.rows_with_changed_value > 0
    assert result.effect_on_data_note is None
    assert store.editing().history == ()


def test_the_result_says_in_one_field_that_it_did_not_check_the_effect_when_the_server_has_no_data(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, _ = continuous_ebm
    result = edit_tools.preview_edit(ModelStore(ebm=ebm), edit_tools.parse_edit(RAISE_LOW_BINS))
    assert result.effect_on_data is None
    assert (
        result.effect_on_data_note
        == "The server has no data (--data), so the effect of the edit on rows is not checked."
    )


def test_preview_starts_from_the_revision_that_the_caller_names(store: ModelStore) -> None:
    edit_tools.apply_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "Raise the lowest bins")
    from_original = edit_tools.preview_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "original")
    from_current = edit_tools.preview_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS))
    assert from_original.report.base_revision == "original"
    assert from_current.report.base_revision == store.editing().revision
    assert from_original.report.changes[0].before == pytest.approx(from_current.report.changes[0].before - 0.5)
    assert from_original.session == from_current.session


def test_apply_adds_the_edit_with_its_reason_and_the_effect_on_the_data(store: ModelStore) -> None:
    result = edit_tools.apply_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "Low values are rare", "original")
    session = store.editing()
    assert result.applied
    assert result.report.revision == session.revision
    assert result.session == RevisionInfo(session.revision, session.revision, 1)
    assert result.effect_on_data is not None
    assert result.effect_on_data.rows_with_changed_value > 0
    assert [row.reason for row in session.history_table()] == ["Low values are rare"]
    preview = edit_tools.preview_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "original")
    assert preview.effect_on_data == result.effect_on_data


def test_apply_keeps_the_edit_when_the_data_does_not_fit_the_model(
    continuous_ebm: FittedClassifier,
) -> None:
    ebm, data = continuous_ebm
    store = ModelStore(ebm=ebm, data=as_frame(data.drop(columns="b")))
    result = edit_tools.apply_edit(store, edit_tools.parse_edit(RAISE_LOW_BINS), "Low values are rare")
    assert result.applied
    assert result.effect_on_data is None
    assert result.effect_on_data_note is not None
    assert "no column for these features: ['b']" in result.effect_on_data_note
    assert store.editing().revision_info().edit_count == 1


def test_apply_needs_a_reason_and_a_current_head(store: ModelStore) -> None:
    edit = edit_tools.parse_edit(RAISE_LOW_BINS)
    with pytest.raises(EditError, match="A message is needed"):
        edit_tools.apply_edit(store, edit, "  ")
    edit_tools.apply_edit(store, edit, "First")
    with pytest.raises(StaleRevisionError, match=r"The head of the session is '.*', and the call expected 'original'"):
        edit_tools.apply_edit(store, edit, "Second", "original")
    assert store.editing().revision_info().edit_count == 1


def test_undo_redo_and_checkout_move_the_head_and_say_where_it_is(store: ModelStore) -> None:
    edit = edit_tools.parse_edit(RAISE_LOW_BINS)
    first = edit_tools.apply_edit(store, edit, "First").session.head
    second = edit_tools.apply_edit(store, edit, "Second").session.head
    undone = edit_tools.undo(store, second)
    assert (undone.action, undone.session) == ("undo", RevisionInfo(first, first, 1))
    redone = edit_tools.redo(store, first)
    assert (redone.action, redone.session) == ("redo", RevisionInfo(second, second, 2))
    moved = edit_tools.checkout(store, "original", second)
    assert (moved.action, moved.session) == ("checkout", RevisionInfo("original", "original", 0))
    assert edit_tools.checkout(store, first).session == RevisionInfo(first, first, 1)


def test_undo_redo_and_checkout_fail_with_a_stale_head_or_nothing_to_move(store: ModelStore) -> None:
    edit = edit_tools.parse_edit(RAISE_LOW_BINS)
    with pytest.raises(EditError, match="no edit to undo"):
        edit_tools.undo(store)
    with pytest.raises(EditError, match="no edit to redo"):
        edit_tools.redo(store)
    edit_tools.apply_edit(store, edit, "First")
    for call in (
        lambda: edit_tools.undo(store, "original"),
        lambda: edit_tools.redo(store, "original"),
        lambda: edit_tools.checkout(store, "original", "original"),
    ):
        with pytest.raises(StaleRevisionError, match="expected 'original'"):
            call()
    with pytest.raises(EditError, match="no revision 'abc'"):
        edit_tools.checkout(store, "abc")


def test_list_edits_gives_the_commits_the_undone_edits_and_on_request_the_replay_json(store: ModelStore) -> None:
    edit = edit_tools.parse_edit(RAISE_LOW_BINS)
    first = edit_tools.apply_edit(store, edit, "First").session.head
    second = edit_tools.apply_edit(store, edit, "Second").session.head
    edit_tools.undo(store)
    listing = edit_tools.list_edits(store)
    assert [(row.id, row.reason, row.kind) for row in listing.commits] == [
        (first, "First", "edit"),
        (second, "Second", "edit"),
    ]
    assert listing.undone == [second]
    assert listing.session == RevisionInfo(first, first, 1)
    assert listing.replay_json is None
    replay = edit_tools.list_edits(store, include_replay_json=True).replay_json
    assert replay is not None
    assert orjson.loads(replay) == orjson.loads(store.editing().to_json())
    assert edit_tools.list_edits(ModelStore(ebm=store.original)).commits == []
