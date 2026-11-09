"""Tests for the script that collects token samples of a model and fits its token weights."""

import ast
import io
from pathlib import Path

import attrs
import numpy as np
import orjson
import pytest
from scripts import fit_token_weights as script

from ebaml.core.ebm_classes import ExplainableBoostingClassifier
from ebaml.core.exceptions import ProviderUnreachableError, RateLimitedError, TokenFitError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.token_weights import ModelKey, TextFeatures, TokenWeights, WeightsEntry, WeightsTable
from ebaml.llm.options import ModelOptions

LUNA: ModelKey = ModelKey("openrouter", "openai/gpt-6-luna")
OTHER: ModelKey = ModelKey("openrouter", "other/model")
BASELINE: int = 200


@attrs.frozen
class FakeSample:
    """A counted request that obeys the ``RequestSample`` protocol."""

    provider: str
    model: str
    body: str
    input_tokens: int


def tokens_of(text: str) -> int:
    """Count the graph tokens of a text as a model with the weights 4 for a word and 0.3 for a digit would.

    Returns
    -------
    int
        The tokens.

    """
    return round(4.0 * len(text.split()) + 0.3 * sum(char.isdigit() for char in text))


PROSE: str = "p" * 880  # 880 bytes of template, which the model counts as 200 tokens


def body_of(text: str) -> str:
    return orjson.dumps({"model": "m", "messages": [{"role": "user", "content": PROSE + text}]}).decode()


def fake_sampler(key: ModelKey = LUNA, *, calls: list[str] | None = None) -> script.Sampler:
    def sampler(text: str) -> list[FakeSample]:
        if calls is not None:
            calls.append(text)
        return [FakeSample(key.provider, key.model, body_of(text), BASELINE + tokens_of(text))]

    return sampler


def no_sleep(_seconds: float) -> None:
    """Skip the wait between two attempts."""


def texts(count: int) -> list[str]:
    return [
        " ".join(f"{(index * 7 + position) % 91}.{position}" for position in range(3 + index % 40))
        for index in range(count)
    ]


def records(count: int = 60, key: ModelKey = LUNA) -> list[script.GraphRecord]:
    return [
        script.GraphRecord(key.provider, key.model, text, BASELINE, len(PROSE), BASELINE + tokens_of(text))
        for text in texts(count)
    ]


def small_ebm() -> ExplainableBoostingClassifier:
    random_generator = np.random.default_rng(0)
    features = random_generator.normal(size=(200, 2))
    labels = (features[:, 0] + random_generator.normal(scale=0.5, size=200) > 0).astype(int)
    return ExplainableBoostingClassifier(
        feature_names=["a", "b"],
        interactions=0,
        max_rounds=3,
        random_state=0,
    ).fit(features, labels)


def continuous_graph(bins: int) -> EBMGraph:
    x_vals = [(float(bin_index), float(bin_index + 1)) for bin_index in range(bins)]
    return EBMGraph("x", "continuous", x_vals, [bin_index * 0.001 for bin_index in range(bins)], [0.01] * bins)


def test_a_graph_record_counts_the_tokens_of_the_graph_without_the_template() -> None:
    record = script.GraphRecord("p", "m", "g", baseline=200, baseline_bytes=880, total=260)
    assert record.tokens == 60
    assert record.key == ModelKey("p", "m")


def test_graph_texts_make_distinct_texts_of_several_sizes() -> None:
    result = script.graph_texts([continuous_graph(200)], budgets=(100, 400, 1500))
    assert len(result) == len(set(result)) > 1
    assert len(result[0]) < len(result[-1])


def test_graph_texts_skip_a_budget_that_a_graph_cannot_fit() -> None:
    # a categorical graph cannot be simplified, so a budget of 50 tokens is too small for 300 categories
    graph = EBMGraph(
        "c",
        "nominal",
        [f"cat{category_index}" for category_index in range(300)],
        [0.1] * 300,
        [0.1] * 300,
    )
    result = script.graph_texts([graph], budgets=(50, 100000))
    assert result
    assert result == script.graph_texts([graph], budgets=(100000,))


def test_build_ebms_fits_three_models_that_give_graphs() -> None:
    ebms = script.build_ebms(max_rounds=2)
    assert len(ebms) == 3
    assert all(EBMGraph.from_ebm(ebm, 0).to_text(max_tokens=300, overbudget_percent=200) for ebm in ebms)


def test_collect_sends_an_empty_graph_first_and_subtracts_its_tokens() -> None:
    calls: list[str] = []
    result = list(script.collect(texts(5), fake_sampler(calls=calls), count=3, sleep=no_sleep))
    assert not calls[0]
    assert len(result) == 3
    assert {record.baseline for record in result} == {BASELINE}
    assert [record.tokens for record in result] == [tokens_of(text) for text in texts(3)]


def test_collect_stops_after_count_records_and_sends_no_more_requests() -> None:
    calls: list[str] = []
    list(script.collect(texts(10), fake_sampler(calls=calls), count=2, sleep=no_sleep))
    assert len(calls) == 3  # the empty graph and two texts


def test_collect_with_no_count_sends_only_the_empty_graph() -> None:
    calls: list[str] = []
    assert list(script.collect(texts(10), fake_sampler(calls=calls), count=0, sleep=no_sleep)) == []
    assert calls == [""]


def test_collect_leaves_out_a_request_that_another_model_answered() -> None:
    def sampler(text: str) -> list[FakeSample]:
        key = LUNA if text in {"", "kept"} else OTHER
        return [FakeSample(key.provider, key.model, body_of(text), 300)]

    result = list(script.collect(["kept", "dropped"], sampler, count=5, sleep=no_sleep))
    assert [record.graph for record in result] == ["kept"]


def test_collect_keeps_the_first_sample_of_a_request() -> None:
    def sampler(text: str) -> list[FakeSample]:
        return (
            [FakeSample("p", "m", body_of(text), 10), FakeSample("p", "m", body_of(text), 99)]
            if text
            else [FakeSample("p", "m", "{}", 3)]
        )

    assert [record.total for record in list(script.collect(["a"], sampler, count=5, sleep=no_sleep))] == [10]


def test_collect_sends_a_request_again_after_a_rate_limit_and_waits_longer_each_time() -> None:
    attempts = {"count": 0}
    waits: list[float] = []

    def sampler(text: str) -> list[FakeSample]:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RateLimitedError
        return [FakeSample(LUNA.provider, LUNA.model, body_of(text), 300)]

    result = list(script.collect(["a"], sampler, count=1, sleep=waits.append))
    assert len(result) == 1
    assert waits == [5.0, 10.0]


def test_collect_sends_a_request_again_after_a_network_failure() -> None:
    attempts = {"count": 0}

    def sampler(text: str) -> list[FakeSample]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ProviderUnreachableError
        return [FakeSample(LUNA.provider, LUNA.model, body_of(text), 300)]

    assert len(list(script.collect(["a"], sampler, count=1, sleep=no_sleep))) == 1


def test_collect_raises_the_error_of_the_provider_after_the_last_attempt() -> None:
    waits: list[float] = []

    def sampler(_text: str) -> list[FakeSample]:
        raise RateLimitedError

    with pytest.raises(RateLimitedError):
        list(script.collect(["a"], sampler, count=1, sleep=waits.append))
    assert waits == [5.0, 10.0]


def test_collect_sends_a_request_again_when_the_provider_reports_no_count() -> None:
    calls: list[str] = []

    def sampler(text: str) -> list[FakeSample]:
        calls.append(text)
        return [] if len(calls) == 1 else [FakeSample(LUNA.provider, LUNA.model, body_of(text), 300)]

    assert len(list(script.collect(["a"], sampler, count=1, sleep=no_sleep))) == 1
    assert calls == ["", "", "a"]


def test_collect_gives_no_record_for_a_text_that_never_gets_a_count() -> None:
    def sampler(text: str) -> list[FakeSample]:
        return [FakeSample(LUNA.provider, LUNA.model, body_of(text), 300)] if text != "bad" else []

    result = list(script.collect(["bad", "good"], sampler, count=5, sleep=no_sleep))
    assert [record.graph for record in result] == ["good"]


def test_collect_yields_each_record_before_it_sends_the_next_request() -> None:
    calls: list[str] = []
    records_so_far = script.collect(texts(5), fake_sampler(calls=calls), count=5, sleep=no_sleep)
    next(records_so_far)
    assert calls == ["", texts(5)[0]]


def test_collect_fails_when_the_empty_graph_has_no_count() -> None:
    with pytest.raises(TokenFitError, match="empty graph"):
        list(script.collect(["a"], lambda _text: [], count=1, sleep=no_sleep))


def test_records_are_written_by_appending_and_read_back_by_model(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    script.write_records(records(3), path)
    script.write_records(records(2, key=OTHER), path)
    assert script.read_records(path, LUNA) == records(3)
    assert script.read_records(path, OTHER) == records(2, key=OTHER)
    assert len(path.read_bytes().splitlines()) == 5


def test_fit_model_recovers_the_weights_and_beats_the_rule_of_three_characters() -> None:
    report = script.fit_model(records())
    assert report.key == LUNA
    # words and digits grow together in the sample texts, so the test judges the estimates, not each weight alone
    sample = records()[7]
    estimate = report.entry.weights.estimate(TextFeatures.of(sample.graph))
    assert estimate == pytest.approx(sample.tokens, rel=0.03)
    assert report.entry.samples == 60
    assert report.entry.prose_bytes_per_token == pytest.approx(880 / 200)
    assert report.entry.update_allowed is True
    assert report.fit_error.mape < 2.0
    assert report.rule_error.mape > 10 * report.fit_error.mape
    assert report.entry.relative_spread == pytest.approx(report.fit_error.mape / 100, abs=0.05)


def test_fit_model_marks_a_final_entry() -> None:
    assert script.fit_model(records(), final=True).entry.update_allowed is False


def test_fit_model_leaves_out_a_record_whose_graph_has_no_tokens() -> None:
    record_without_tokens = script.GraphRecord(LUNA.provider, LUNA.model, "g", BASELINE, len(PROSE), BASELINE)
    assert script.fit_model([*records(), record_without_tokens]).entry.samples == 60


def test_fit_model_rejects_records_of_two_models() -> None:
    with pytest.raises(TokenFitError, match="one model"):
        script.fit_model([*records(30), *records(30, key=OTHER)])


def test_fit_model_rejects_records_without_a_measure_of_the_template() -> None:
    unmeasured = [attrs.evolve(record, baseline_bytes=0) for record in records()]
    with pytest.raises(TokenFitError, match="prompt template"):
        script.fit_model(unmeasured)


def test_collect_measures_the_bytes_of_the_template() -> None:
    result = list(script.collect(texts(2), fake_sampler(), count=2, sleep=no_sleep))
    assert {record.baseline_bytes for record in result} == {len(PROSE)}


def test_fit_model_rejects_too_few_records() -> None:
    with pytest.raises(TokenFitError, match="at least"):
        script.fit_model(records(4))


def test_set_default_constants_replaces_the_weights_and_the_spread_and_nothing_else() -> None:
    source = script.ESTIMATOR_PATH.read_text()
    entry = WeightsEntry(TokenWeights(words=1.5, bytes=0.25, digits=0.125), relative_spread=0.0625, samples=9)
    patched = script.set_default_constants(source, entry)
    assert "DEFAULT_WEIGHTS: TokenWeights = TokenWeights(words=1.5, bytes=0.25, digits=0.125)" in patched
    assert "DEFAULT_RELATIVE_SPREAD: float = 0.0625" in patched
    ast.parse(patched)
    assert len(patched.splitlines()) == len(source.splitlines())
    assert script.set_default_constants(patched, entry) == patched


def test_set_default_constants_fails_without_the_lines_to_replace() -> None:
    entry = WeightsEntry(TokenWeights(words=1.0, bytes=0.0, digits=0.0), relative_spread=0.1, samples=1)
    with pytest.raises(TokenFitError, match="no DEFAULT_WEIGHTS"):
        script.set_default_constants("x = 1\n", entry)
    with pytest.raises(TokenFitError, match="no DEFAULT_WEIGHTS"):
        script.set_default_constants("DEFAULT_WEIGHTS: TokenWeights = TokenWeights(words=1)\n", entry)


def test_the_shipped_default_constants_are_the_ones_that_the_script_writes() -> None:
    source = script.ESTIMATOR_PATH.read_text()
    default = WeightsTable.from_bytes(script.TABLE_PATH.read_bytes()).default
    assert script.set_default_constants(source, default) == source


def test_a_model_name_with_slashes_is_read_after_the_first_slash() -> None:
    assert script._parse_model("openrouter/openai/gpt-6-luna") == LUNA


@pytest.mark.parametrize("text", ["gpt", "/model", "provider/"])
def test_a_model_without_a_provider_or_a_name_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="provider/model"):
        script._parse_model(text)


def test_the_default_samples_file_is_named_after_the_model() -> None:
    assert script._default_samples_path(LUNA) == Path("token_samples/openrouter--openai--gpt-6-luna.jsonl")


def test_collect_without_yes_sends_nothing_and_says_how_many_requests_it_would_send(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    code = script.main(
        ["collect", "openrouter/openai/gpt-6-luna", "--count", "7", "--output", str(tmp_path / "s.jsonl")],
        sampler=fake_sampler(calls=calls),
    )
    assert code == 2
    assert calls == []
    assert "8 paid requests" in capsys.readouterr().err
    assert not (tmp_path / "s.jsonl").exists()


def test_collect_with_yes_writes_the_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(script, "build_ebms", lambda: [small_ebm()])
    code = script.main(
        [
            "collect",
            "openrouter/openai/gpt-6-luna",
            "--count",
            "4",
            "--yes",
            "--output",
            str(tmp_path / "out" / "s.jsonl"),
        ],
        sampler=fake_sampler(),
    )
    assert code == 0
    assert len(script.read_records(tmp_path / "out" / "s.jsonl", LUNA)) == 4


def _collect_arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "collect",
        "openrouter/openai/gpt-6-luna",
        "--count",
        "4",
        "--yes",
        "--output",
        str(tmp_path / "s.jsonl"),
        *extra,
    ]


def test_collect_shows_a_progress_bar_of_the_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_stream: io.StringIO,
) -> None:
    monkeypatch.setattr(script, "build_ebms", lambda: [small_ebm()])
    script.main(_collect_arguments(tmp_path), sampler=fake_sampler())
    lines = error_stream.getvalue().splitlines()
    assert all(line.startswith("collect ") for line in lines)
    assert "(4 of 4)" in lines[-1]


def test_collect_with_no_progress_shows_no_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_stream: io.StringIO,
) -> None:
    monkeypatch.setattr(script, "build_ebms", lambda: [small_ebm()])
    script.main(_collect_arguments(tmp_path, "--no-progress"), sampler=fake_sampler())
    assert not error_stream.getvalue()


def test_collect_of_zero_records_shows_no_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_stream: io.StringIO,
) -> None:
    # progressbar2 draws a spinner for a total of 0, which tells the reader nothing
    monkeypatch.setattr(script, "build_ebms", lambda: [small_ebm()])
    arguments = _collect_arguments(tmp_path)
    arguments[arguments.index("--count") + 1] = "0"
    script.main(arguments, sampler=fake_sampler())
    assert not error_stream.getvalue()


def test_the_real_sampler_asks_for_a_short_reply_from_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from ebaml.llm import descriptors  # ruff: ignore[import-outside-top-level] -- the test needs the generated baml_sdk only here

    graphs: list[str] = []
    models: list[ModelOptions] = []

    def fake(graph: str, model: ModelOptions) -> list[FakeSample]:
        graphs.append(graph)
        models.append(model)
        return []

    monkeypatch.setattr(descriptors, "sample_graph_tokens", fake)
    assert script._real_sampler(LUNA)("GRAPH") == []
    assert graphs == ["GRAPH"]
    assert (models[0].provider, models[0].model, models[0].max_output_tokens) == (LUNA.provider, LUNA.model, 16)


def write_samples(tmp_path: Path) -> Path:
    path = tmp_path / "samples.jsonl"
    script.write_records(records(), path)
    return path


def test_fit_dry_run_prints_the_report_and_writes_nothing(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    table = tmp_path / "table.json"
    original = script.TABLE_PATH.read_bytes()
    table.write_bytes(original)
    code = script.main(
        [
            "fit",
            "openrouter/openai/gpt-6-luna",
            "--samples",
            str(write_samples(tmp_path)),
            "--table",
            str(table),
            "--dry-run",
        ],
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "60 samples" in output
    assert "3 chars/token" in output
    assert table.read_bytes() == original


def test_fit_writes_the_entry_of_the_model_into_the_table(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    table = tmp_path / "table.json"
    table.write_bytes(script.TABLE_PATH.read_bytes())
    new_model = "openrouter/openai/gpt-6-luna"
    code = script.main(["fit", new_model, "--samples", str(write_samples(tmp_path)), "--table", str(table), "--final"])
    assert code == 0
    entry = WeightsTable.from_bytes(table.read_bytes()).entry_for(LUNA)
    assert entry.update_allowed is False
    assert entry.weights.estimate(TextFeatures.of(records()[7].graph)) == pytest.approx(records()[7].tokens, rel=0.03)
    assert f"wrote {table}" in capsys.readouterr().out
    assert orjson.loads(table.read_bytes())["default"] == orjson.loads(script.TABLE_PATH.read_bytes())["default"]


def test_fit_with_set_default_also_changes_the_default_entry_and_the_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    table = tmp_path / "table.json"
    table.write_bytes(script.TABLE_PATH.read_bytes())
    estimator = tmp_path / "token_budget.py"
    estimator.write_text(script.ESTIMATOR_PATH.read_text())
    monkeypatch.setattr(script, "ESTIMATOR_PATH", estimator)
    code = script.main(
        [
            "fit",
            "openrouter/openai/gpt-6-luna",
            "--samples",
            str(write_samples(tmp_path)),
            "--table",
            str(table),
            "--set-default",
        ],
    )
    assert code == 0
    default = WeightsTable.from_bytes(table.read_bytes()).default
    assert default.weights.estimate(TextFeatures.of(records()[7].graph)) == pytest.approx(records()[7].tokens, rel=0.03)
    assert script.set_default_constants(estimator.read_text(), default) == estimator.read_text()
