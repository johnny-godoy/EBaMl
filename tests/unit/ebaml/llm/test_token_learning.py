"""Tests for the token learner: the table that ships, the estimator of each model, and the learned weights file."""

from pathlib import Path

import attrs
import orjson
import pytest
from structlog.testing import capture_logs

import baml_sdk
from ebaml.graphs.token_budget import DEFAULT_RELATIVE_SPREAD, DEFAULT_WEIGHTS
from ebaml.graphs.token_weights import ModelKey, TextFeatures, TokenSample, TokenWeights, WeightsEntry, WeightsTable
from ebaml.llm.token_learning import RequestSample, TokenLearner, default_table, default_weights_path
from tests.unit.ebaml.conftest import protocol_mismatches

LUNA: ModelKey = ModelKey("openrouter", "openai/gpt-6-luna")
OTHER: ModelKey = ModelKey("openrouter", "other/model")
START: TokenWeights = TokenWeights(words=1.0, bytes=0.0, digits=0.0)


def _table(*, update_allowed: bool = True) -> WeightsTable:
    entry = WeightsEntry(START, relative_spread=0.1, samples=5, update_allowed=update_allowed)
    return WeightsTable(default=WeightsEntry(START, 0.2, 1), models={LUNA: entry})


def _sample(
    key: ModelKey = LUNA,
    text: str = "alpha beta gamma 1 2 3",
    tokens: int = 40,
) -> baml_sdk.TokenSample:
    body = orjson.dumps({"model": key.model, "messages": [{"role": "user", "content": text}]}).decode()
    return baml_sdk.TokenSample(provider=key.provider, model=key.model, body=body, input_tokens=tokens)


def _learner(
    tmp_path: Path,
    *,
    learning: bool = False,
    persist: bool = True,
    prior_samples: float = 10.0,
    margin: float = 1.0,
) -> TokenLearner:
    return TokenLearner(
        learning=learning,
        persist=persist,
        path=tmp_path / "learned.json",
        table=_table(),
        prior_samples=prior_samples,
        margin=margin,
    )


def test_the_table_that_ships_has_the_default_weights_of_the_estimator() -> None:
    entry = default_table().default
    assert entry.weights == pytest.approx(DEFAULT_WEIGHTS)
    assert entry.relative_spread == pytest.approx(DEFAULT_RELATIVE_SPREAD)


def test_the_table_that_ships_lists_the_model_of_the_default_weights() -> None:
    assert default_table().entry_for(LUNA).weights == pytest.approx(DEFAULT_WEIGHTS)
    assert default_table().entry_for(LUNA).samples == 100


def test_a_request_sample_protocol_matches_the_sdk_class() -> None:
    assert protocol_mismatches(RequestSample, "TokenSample", {}) == []
    assert isinstance(_sample(), RequestSample)


def test_the_default_path_is_in_the_xdg_cache_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_weights_path() == tmp_path / "ebaml" / "token_weights.json"


def test_the_default_path_falls_back_to_the_cache_directory_of_the_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert default_weights_path() == tmp_path / ".cache" / "ebaml" / "token_weights.json"


def test_the_estimator_of_a_listed_model_starts_at_its_entry(tmp_path: Path) -> None:
    estimator = _learner(tmp_path).estimator_for(LUNA)
    assert (estimator.weights, estimator.relative_spread) == (START, 0.1)


def test_the_estimator_of_an_unlisted_or_unknown_model_starts_at_the_default_entry(tmp_path: Path) -> None:
    learner = _learner(tmp_path)
    assert learner.estimator_for(OTHER).relative_spread == pytest.approx(0.2)
    assert learner.estimator_for(None).relative_spread == pytest.approx(0.2)


def test_a_learner_gives_the_same_estimator_for_the_same_model(tmp_path: Path) -> None:
    learner = _learner(tmp_path)
    assert learner.estimator_for(LUNA) is learner.estimator_for(LUNA)
    assert learner.estimator_for(LUNA) is not learner.estimator_for(OTHER)


def test_an_estimator_for_an_unknown_model_is_not_kept(tmp_path: Path) -> None:
    learner = _learner(tmp_path, learning=True)
    assert learner.estimator_for(None) is not learner.estimator_for(None)


def test_the_estimator_gets_the_margin_and_the_prior_of_the_learner(tmp_path: Path) -> None:
    estimator = _learner(tmp_path, margin=2.5, prior_samples=3.0).estimator_for(LUNA)
    assert (estimator.margin, estimator.prior_samples) == (2.5, 3.0)


def test_learning_is_off_by_default_and_the_estimator_follows_the_learner(tmp_path: Path) -> None:
    assert _learner(tmp_path).estimator_for(LUNA).learning is False
    assert _learner(tmp_path, learning=True).estimator_for(LUNA).learning is True


def test_a_model_whose_entry_is_final_never_learns(tmp_path: Path) -> None:
    learner = TokenLearner(learning=True, path=tmp_path / "learned.json", table=_table(update_allowed=False))
    assert learner.estimator_for(LUNA).learning is False
    assert learner.observe([_sample()]) == 0
    assert not (tmp_path / "learned.json").exists()


def test_observe_does_nothing_when_learning_is_off(tmp_path: Path) -> None:
    learner = _learner(tmp_path)
    assert learner.observe([_sample()]) == 0
    assert learner.estimator_for(LUNA).weights == START
    assert not (tmp_path / "learned.json").exists()


def test_observe_moves_the_weights_of_the_model_that_answered(tmp_path: Path) -> None:
    learner = _learner(tmp_path, learning=True)
    assert learner.observe([_sample(LUNA, tokens=400)]) == 1
    assert learner.estimator_for(LUNA).samples_seen == 1
    assert learner.estimator_for(LUNA).current_weights.words > START.words
    assert learner.estimator_for(OTHER).samples_seen == 0


def test_observe_counts_the_words_of_the_message_and_not_the_json_around_it(tmp_path: Path) -> None:
    learner = _learner(tmp_path, learning=True)
    learner.observe([_sample(LUNA, text="one two", tokens=10)])
    reference = TokenLearner(table=_table(), learning=True, path=tmp_path / "other.json")
    reference.estimator_for(LUNA).observe(TokenSample(TextFeatures(words=2, bytes=7, digits=0), 10))
    assert learner.estimator_for(LUNA).current_weights == reference.estimator_for(LUNA).current_weights


def test_observe_skips_a_sample_whose_body_holds_no_text(tmp_path: Path) -> None:
    learner = _learner(tmp_path, learning=True)
    empty = baml_sdk.TokenSample(provider="openrouter", model=LUNA.model, body="{}", input_tokens=9)
    assert learner.observe([empty]) == 0
    assert not (tmp_path / "learned.json").exists()


def test_learned_weights_persist_in_the_file_and_the_next_learner_starts_from_them(tmp_path: Path) -> None:
    first = _learner(tmp_path, learning=True)
    first.observe([_sample(LUNA, tokens=400), _sample(LUNA, tokens=380)])
    second = _learner(tmp_path, learning=True)
    assert second.estimator_for(LUNA).current_weights == first.estimator_for(LUNA).current_weights
    assert second.estimator_for(LUNA).samples_seen == 2


def test_the_file_keeps_the_models_that_this_learner_did_not_touch(tmp_path: Path) -> None:
    _learner(tmp_path, learning=True).observe([_sample(OTHER, tokens=300)])
    _learner(tmp_path, learning=True).observe([_sample(LUNA, tokens=400)])
    models = {
        (record["provider"], record["model"])
        for record in orjson.loads((tmp_path / "learned.json").read_bytes())["models"]
    }
    assert models == {(OTHER.provider, OTHER.model), (LUNA.provider, LUNA.model)}


def test_a_learner_with_persistence_off_writes_no_file_and_reads_none(tmp_path: Path) -> None:
    _learner(tmp_path, learning=True).observe([_sample(LUNA, tokens=400)])
    quiet = _learner(tmp_path, learning=True, persist=False)
    assert quiet.estimator_for(LUNA).samples_seen == 0
    quiet.observe([_sample(LUNA, tokens=100)])
    saved = orjson.loads((tmp_path / "learned.json").read_bytes())["models"]
    assert saved[0]["samples_seen"] == 1


def test_a_learner_that_does_not_learn_ignores_a_file_of_learned_weights(tmp_path: Path) -> None:
    _learner(tmp_path, learning=True).observe([_sample(LUNA, tokens=400)])
    assert _learner(tmp_path).estimator_for(LUNA).current_weights == START


def test_a_file_that_cannot_be_read_gives_a_warning_and_a_fresh_start(tmp_path: Path) -> None:
    (tmp_path / "learned.json").write_bytes(b"not json")
    with capture_logs() as logs:
        learner = _learner(tmp_path, learning=True)
        assert learner.estimator_for(LUNA).current_weights == START
    assert [log["event"] for log in logs] == ["token_weights_unreadable"]


@pytest.mark.parametrize(
    "document",
    [
        {"models": [{"provider": "p", "model": "m", "weights": [1, 0, 0], "covariance": [1.0], "samples_seen": 1}]},
        {
            "models": [
                {"provider": "p", "model": "m", "weights": [-1, 0, 0], "covariance": [0.0] * 9, "samples_seen": 1},
            ],
        },
        {"models": [{"provider": "p"}]},
        {"other": []},
    ],
)
def test_a_file_with_a_wrong_record_gives_a_warning(tmp_path: Path, document: dict[str, object]) -> None:
    (tmp_path / "learned.json").write_bytes(orjson.dumps(document))
    with capture_logs() as logs:
        _learner(tmp_path, learning=True).estimator_for(LUNA)
    assert [log["event"] for log in logs] == ["token_weights_unreadable"]


def test_a_file_that_cannot_be_written_gives_a_warning_and_does_not_stop_the_call(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"a file where the directory should be")
    learner = TokenLearner(learning=True, path=blocker / "learned.json", table=_table())
    with capture_logs() as logs:
        assert learner.observe([_sample(LUNA, tokens=400)]) == 1
    assert "token_weights_not_saved" in [log["event"] for log in logs]


def test_the_learner_writes_to_the_default_path_when_it_gets_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    TokenLearner(learning=True, table=_table()).observe([_sample(LUNA, tokens=400)])
    assert (tmp_path / "ebaml" / "token_weights.json").exists()


def test_no_file_and_no_log_holds_the_text_of_a_message(tmp_path: Path) -> None:
    marker = "SECRET-MARKER-IN-THE-PROMPT"
    learner = _learner(tmp_path, learning=True)
    with capture_logs() as logs:
        learner.observe([_sample(LUNA, text=f"{marker} 1 2 3", tokens=50)])
    assert marker.encode() not in (tmp_path / "learned.json").read_bytes()
    assert marker not in repr(logs)


def test_save_with_learning_off_writes_nothing(tmp_path: Path) -> None:
    _learner(tmp_path).save()
    assert not (tmp_path / "learned.json").exists()


def test_a_learner_is_a_plain_attrs_class_with_identity_equality(tmp_path: Path) -> None:
    assert attrs.has(TokenLearner)
    assert _learner(tmp_path) != _learner(tmp_path)
