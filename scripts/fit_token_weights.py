"""Collect token samples of a model and fit its token weights.

Run it when a new model comes out, or when you want to refit the default weights. There are two steps. The first sends
requests, so it costs money and needs a key. The second sends nothing.

1. ``collect`` sends one request for each graph text, and one for an empty graph. The empty graph measures the tokens
   of the prompt template, which the fit takes away from every request. The result is a JSON lines file.
2. ``fit`` reads that file, fits ``tokens = words * a + bytes * b + digits * c`` with weights of zero or above, prints
   the error on held-out samples, and writes the entry of the model into ``src/ebaml/llm/token_weights.json``.
   With ``--set-default`` it also makes the entry the default, in the table and in ``DEFAULT_WEIGHTS`` of
   ``src/ebaml/graphs/token_budget.py``.

Example for a new model::

    OPENROUTER_API_KEY=... uv run python scripts/fit_token_weights.py collect openrouter/openai/gpt-6-luna --yes
    uv run python scripts/fit_token_weights.py fit openrouter/openai/gpt-6-luna
"""

import argparse
import math
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
import orjson
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # pyrefly: ignore[implicit-reexport] -- interpret.glassbox has no __all__, so pyrefly reads its public classes as private imports
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EBMGraphSimplificationError, EBMGraphTokenLimitError, TokenFitError
from ebaml.graphs.ebm_graph import EBMGraph
from ebaml.graphs.token_budget import check_token_budget
from ebaml.graphs.token_weights import (
    ModelKey,
    TextFeatures,
    TokenSample,
    WeightsEntry,
    WeightsTable,
    fit_weights,
    relative_spread,
)
from ebaml.llm.token_learning import RequestSample

ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = ROOT / "src" / "ebaml" / "llm" / "token_weights.json"
ESTIMATOR_PATH = ROOT / "src" / "ebaml" / "graphs" / "token_budget.py"

_BUDGETS = (120, 300, 700, 1500, 3500)


@attrs.frozen
class _Display:
    """The display options of ``EBMGraph.to_text`` that change the length of a text and its share of digits."""

    x_axis_precision: int | None = None
    confidence_bounds: bool = True
    include_description: bool = True


_DISPLAYS = (_Display(), _Display(x_axis_precision=1, confidence_bounds=False), _Display(3, include_description=False))
_FOLDS = 5
_SEED = 0
_REPLY_LIMIT = 16
# The rule that a rough estimate often follows. The report compares the fit with it.
_CHARS_PER_TOKEN = 3

Sampler = Callable[[str], Sequence[RequestSample]]


@attrs.frozen
class GraphRecord:
    """One measured graph: its text, and the input tokens of the request with it and of the request with no graph.

    Attributes
    ----------
    provider: str
        The provider that answered.
    model: str
        The model that answered.
    graph: str
        The text of the graph.
    baseline: int
        The input tokens of the request with an empty graph, which are the tokens of the prompt template.
    total: int
        The input tokens of the request with this graph.

    """

    provider: str
    model: str
    graph: str
    baseline: int
    total: int

    @property
    def tokens(self) -> int:
        """The tokens of the graph alone."""
        return self.total - self.baseline

    @property
    def key(self) -> ModelKey:
        """The model that answered."""
        return ModelKey(self.provider, self.model)


@attrs.frozen
class ErrorStats:
    """How far estimates are from the real counts, on samples that the fit did not see.

    Attributes
    ----------
    mape: float
        The mean absolute relative error, in percent.
    bias: float
        The mean signed relative error in percent. Above zero, the estimates are too high.
    under: float
        The share of estimates below the real count, in percent.

    """

    mape: float
    bias: float
    under: float


@attrs.frozen
class FitReport:
    """The result of a fit.

    Attributes
    ----------
    key: ModelKey
        The model.
    entry: WeightsEntry
        The weights, their spread and the number of samples.
    fit: ErrorStats
        The held-out error of the fit.
    rule: ErrorStats
        The held-out error of the rule of 3 characters per token on the same samples.

    """

    key: ModelKey
    entry: WeightsEntry
    fit: ErrorStats
    rule: ErrorStats


def build_ebms(max_rounds: int = 60) -> list[ExplainableBoostingModel]:
    """Fit small EBMs on datasets that ship with scikit-learn, so that the graphs differ in size and shape.

    Parameters
    ----------
    max_rounds: int, default=60
        The boosting rounds of each EBM.

    Returns
    -------
    list[ExplainableBoostingModel]
        Three EBMs: two classifiers and a regressor.

    """
    cancer_x, cancer_y = load_breast_cancer(return_X_y=True, as_frame=True)
    diabetes_x, diabetes_y = load_diabetes(return_X_y=True, as_frame=True)
    wine_x, wine_y = load_wine(return_X_y=True, as_frame=True)
    return [
        ExplainableBoostingClassifier(interactions=0, max_rounds=max_rounds, random_state=_SEED).fit(
            cancer_x,
            cancer_y,
        ),
        ExplainableBoostingRegressor(interactions=0, max_rounds=max_rounds, random_state=_SEED).fit(
            diabetes_x,
            diabetes_y,
        ),
        ExplainableBoostingClassifier(interactions=0, max_rounds=max_rounds, random_state=_SEED).fit(
            wine_x,
            (np.asarray(wine_y) == 0).astype(int),
        ),
    ]


def graph_texts(graphs: Sequence[EBMGraph], budgets: Sequence[int] = _BUDGETS) -> list[str]:
    """Make the text of each graph at each token budget and with each set of display options.

    ``EBMGraph.to_text`` simplifies a graph to a budget only when it gets a budget check, so the function passes the
    default one. The display options change the length and the share of digits of the text. The function keeps the
    distinct texts.

    Parameters
    ----------
    graphs: Sequence[EBMGraph]
        The graphs.
    budgets: Sequence[int]
        The token budgets.

    Returns
    -------
    list[str]
        The distinct texts, in order. A graph that cannot fit a budget gives no text for it.

    """
    texts: dict[str, None] = {}
    for graph in graphs:
        for budget in budgets:
            for display in _DISPLAYS:
                try:
                    text = graph.to_text(
                        max_tokens=budget,
                        overbudget_percent=0,
                        check_budget=check_token_budget,
                        **attrs.asdict(display),
                    )
                except (EBMGraphSimplificationError, EBMGraphTokenLimitError):
                    continue
                texts[text] = None
    return list(texts)


def collect(texts: Sequence[str], sampler: Sampler, count: int) -> list[GraphRecord]:
    """Measure the graph texts: send an empty graph first, and then the texts until there are ``count`` records.

    The tokens of the template are the same in every request, so the empty graph measures them. A request that a
    different model answered gives no record, because another tokenizer counted it.

    Parameters
    ----------
    texts: Sequence[str]
        The graph texts.
    sampler: Sampler
        A function that sends one request with a graph text and returns the token samples, such as
        ``ebaml.llm.descriptors.sample_graph_tokens``.
    count: int
        The most records to make.

    Returns
    -------
    list[GraphRecord]
        The records.

    Raises
    ------
    TokenFitError
        If the provider reports no input count for the empty graph.

    """
    baseline = sampler("")
    if not baseline:
        msg = "The provider reported no input count for the empty graph, so the template cannot be measured."
        raise TokenFitError(msg)
    key = ModelKey(baseline[0].provider, baseline[0].model)
    records: list[GraphRecord] = []
    for text in texts:
        if len(records) >= count:
            break
        records.extend(
            GraphRecord(sample.provider, sample.model, text, baseline[0].input_tokens, sample.input_tokens)
            for sample in sampler(text)[:1]
            if ModelKey(sample.provider, sample.model) == key
        )
    return records


def write_records(records: Sequence[GraphRecord], path: Path) -> None:
    """Append the records to a JSON lines file.

    Parameters
    ----------
    records: Sequence[GraphRecord]
        The records.
    path: Path
        The file. The function creates it when it does not exist.

    """
    with path.open("ab") as file:
        for record in records:
            file.write(orjson.dumps(attrs.asdict(record)) + b"\n")


def read_records(path: Path, key: ModelKey) -> list[GraphRecord]:
    """Read the records of one model from a JSON lines file.

    Parameters
    ----------
    path: Path
        The file that ``write_records`` wrote.
    key: ModelKey
        The model. Records of other models in the file are left out.

    Returns
    -------
    list[GraphRecord]
        The records of the model.

    """
    records = (GraphRecord(**orjson.loads(line)) for line in path.read_bytes().splitlines() if line.strip())
    return [record for record in records if record.key == key]


def _stats(estimates: npt.NDArray[np.float64], counts: npt.NDArray[np.float64]) -> ErrorStats:
    """Measure the relative error of estimates.

    Parameters
    ----------
    estimates: npt.NDArray[np.float64]
        The estimated tokens.
    counts: npt.NDArray[np.float64]
        The real tokens.

    Returns
    -------
    ErrorStats
        The error in percent.

    """
    error = (estimates - counts) / counts
    return ErrorStats(
        mape=float(np.mean(np.abs(error)) * 100),
        bias=float(np.mean(error) * 100),
        under=float(np.mean(estimates < counts) * 100),
    )


def _mean_stats(folds: Sequence[ErrorStats]) -> ErrorStats:
    """Average the error of the folds.

    Parameters
    ----------
    folds : Sequence[ErrorStats]
        The error of each fold.

    Returns
    -------
    ErrorStats
        The mean of each field.

    """
    return ErrorStats(
        mape=sum(fold.mape for fold in folds) / len(folds),
        bias=sum(fold.bias for fold in folds) / len(folds),
        under=sum(fold.under for fold in folds) / len(folds),
    )


def fit_model(records: Sequence[GraphRecord], *, final: bool = False) -> FitReport:
    """Fit the weights of one model, and measure the fit and the rule of 3 characters per token on held-out samples.

    Parameters
    ----------
    records: Sequence[GraphRecord]
        The records of one model. A record whose graph has no tokens is left out.
    final: bool, default=False
        Whether the entry forbids a learning estimator to change the weights.

    Returns
    -------
    FitReport
        The entry and the held-out error.

    Raises
    ------
    TokenFitError
        If the records come from more than one model, or fewer records than folds remain.

    """
    usable = [record for record in records if record.tokens >= 1]
    keys = {record.key for record in usable}
    if len(keys) != 1 or len(usable) < _FOLDS:
        msg = f"Fitting needs the records of one model and at least {_FOLDS} of them, not {len(usable)} of {len(keys)}."
        raise TokenFitError(msg)
    samples = [TokenSample(TextFeatures.of(record.graph), record.tokens) for record in usable]
    counts = np.array([float(sample.tokens) for sample in samples])
    rule = np.array([math.ceil(len(record.graph) / _CHARS_PER_TOKEN) for record in usable], dtype=float)
    folds = np.array_split(np.random.default_rng(_SEED).permutation(len(samples)), _FOLDS)
    fit_folds: list[ErrorStats] = []
    rule_folds: list[ErrorStats] = []
    for held_out in folds:
        held_list: list[int] = held_out.tolist()
        held = set(held_list)
        weights = fit_weights([sample for index, sample in enumerate(samples) if index not in held])
        estimates = np.array([weights.estimate(samples[index].features) for index in held_list])
        fit_folds.append(_stats(estimates, counts[held_out]))
        rule_folds.append(_stats(rule[held_out], counts[held_out]))
    weights = fit_weights(samples)
    entry = WeightsEntry(
        weights=weights,
        relative_spread=relative_spread(weights, samples),
        samples=len(samples),
        update_allowed=not final,
    )
    return FitReport(usable[0].key, entry, _mean_stats(fit_folds), _mean_stats(rule_folds))


_WEIGHTS_PATTERN = re.compile(r"^DEFAULT_WEIGHTS: TokenWeights = TokenWeights\(.*\)$", re.MULTILINE)
_SPREAD_PATTERN = re.compile(r"^DEFAULT_RELATIVE_SPREAD: float = .*$", re.MULTILINE)


def set_default_constants(source: str, entry: WeightsEntry) -> str:
    """Write the weights and the spread of an entry into the source of ``token_budget.py``.

    Parameters
    ----------
    source : str
        The text of ``src/ebaml/graphs/token_budget.py``.
    entry : WeightsEntry
        The new default entry.

    Returns
    -------
    str
        The source with the new ``DEFAULT_WEIGHTS`` and ``DEFAULT_RELATIVE_SPREAD``.

    Raises
    ------
    TokenFitError
        If the source lacks one of the two assignments.

    """
    weights = entry.weights
    line = (
        f"DEFAULT_WEIGHTS: TokenWeights = TokenWeights(words={round(weights.words, 7)!r}, "
        f"bytes={round(weights.bytes, 7)!r}, digits={round(weights.digits, 7)!r})"
    )
    spread = f"DEFAULT_RELATIVE_SPREAD: float = {round(entry.relative_spread, 4)!r}"
    if not _WEIGHTS_PATTERN.search(source) or not _SPREAD_PATTERN.search(source):
        msg = "token_budget.py has no DEFAULT_WEIGHTS or DEFAULT_RELATIVE_SPREAD line to replace."
        raise TokenFitError(msg)
    return _SPREAD_PATTERN.sub(spread, _WEIGHTS_PATTERN.sub(line, source, count=1), count=1)


def _format_report(report: FitReport) -> str:
    """Write the report of a fit as text.

    Parameters
    ----------
    report : FitReport
        The report.

    Returns
    -------
    str
        The lines to print.

    """
    weights = report.entry.weights
    return "\n".join(
        [
            f"model: {report.key.provider}/{report.key.model} ({report.entry.samples} samples)",
            f"weights: words={weights.words:.4f} bytes={weights.bytes:.4f} digits={weights.digits:.4f}",
            f"relative spread of the fit: {report.entry.relative_spread:.4f}",
            "held-out error     mape%   bias%  under%",
            f"  fit            {report.fit.mape:7.1f} {report.fit.bias:+7.1f} {report.fit.under:7.0f}",
            f"  3 chars/token  {report.rule.mape:7.1f} {report.rule.bias:+7.1f} {report.rule.under:7.0f}",
        ],
    )


def _parse_model(value: str) -> ModelKey:
    """Read ``provider/model`` into a key. The model name may hold more slashes.

    Parameters
    ----------
    value : str
        The shorthand, for example ``openrouter/openai/gpt-6-luna``.

    Returns
    -------
    ModelKey
        The provider and the model.

    Raises
    ------
    argparse.ArgumentTypeError
        If the text has no slash, or a part is empty.

    """
    provider, _, model = value.partition("/")
    if not provider or not model:
        msg = f"expected provider/model, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return ModelKey(provider, model)


def _real_sampler(key: ModelKey) -> Sampler:
    """Build the sampler that sends real requests to a model.

    Parameters
    ----------
    key : ModelKey
        The model.

    Returns
    -------
    Sampler
        A function that sends one request with a graph text and a reply limit of 16 tokens.

    """
    from ebaml.llm.descriptors import sample_graph_tokens  # ruff: ignore[import-outside-top-level] -- the import needs the generated baml_sdk, which the fit step does not
    from ebaml.llm.options import ModelOptions  # ruff: ignore[import-outside-top-level] -- the import needs the generated baml_sdk, which the fit step does not need

    model = ModelOptions(key.provider, key.model, max_output_tokens=_REPLY_LIMIT)
    return lambda text: sample_graph_tokens(text, model)


def _parser() -> argparse.ArgumentParser:
    """Build the parser of the command line.

    Returns
    -------
    argparse.ArgumentParser
        The parser.

    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="send requests and write the records (costs money)")
    collect_parser.add_argument(
        "model",
        type=_parse_model,
        help="provider/model, for example openrouter/openai/gpt-6-luna",
    )
    collect_parser.add_argument("--count", type=int, default=100, help="the number of graph texts to send")
    collect_parser.add_argument(
        "--output",
        type=Path,
        help="the JSON lines file (default: token_samples/<model>.jsonl)",
    )
    collect_parser.add_argument("--yes", action="store_true", help="agree to pay for count + 1 requests")
    fit_parser = commands.add_parser("fit", help="fit the weights from the records and write the table (sends nothing)")
    fit_parser.add_argument("model", type=_parse_model, help="provider/model")
    fit_parser.add_argument("--samples", type=Path, help="the JSON lines file (default: token_samples/<model>.jsonl)")
    fit_parser.add_argument("--table", type=Path, default=TABLE_PATH, help="the table to update")
    fit_parser.add_argument("--final", action="store_true", help="forbid a learning estimator to change the entry")
    fit_parser.add_argument("--set-default", action="store_true", help="also make the entry the default of the package")
    fit_parser.add_argument("--dry-run", action="store_true", help="print the report and write nothing")
    return parser


def _default_samples_path(key: ModelKey) -> Path:
    """Name the default records file of a model.

    Parameters
    ----------
    key : ModelKey
        The model.

    Returns
    -------
    Path
        ``token_samples/<provider>--<model>.jsonl`` with the slashes of the model name replaced.

    """
    return Path("token_samples") / f"{key.provider}--{key.model.replace('/', '--')}.jsonl"


def main(argv: Sequence[str] | None = None, *, sampler: Sampler | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : Sequence[str] | None
        The arguments. ``None`` reads ``sys.argv``.
    sampler : Sampler | None
        The function that sends the requests of ``collect``. ``None`` sends real requests to the model.

    Returns
    -------
    int
        The exit code: 0 on success, 2 when ``collect`` lacks ``--yes``.

    """
    args = _parser().parse_args(argv)
    key: ModelKey = args.model
    if args.command == "collect":
        output = args.output or _default_samples_path(key)
        if not args.yes:
            sys.stderr.write(
                f"This sends {args.count + 1} paid requests to {key.provider}/{key.model}. Add --yes to agree.\n",
            )
            return 2
        output.parent.mkdir(parents=True, exist_ok=True)
        texts = graph_texts([
            EBMGraph.from_ebm(ebm, index) for ebm in build_ebms() for index in range(len(ebm.feature_names_in_))
        ])
        records = collect(texts, sampler or _real_sampler(key), args.count)
        write_records(records, output)
        sys.stdout.write(f"wrote {len(records)} records to {output}\n")
        return 0
    report = fit_model(read_records(args.samples or _default_samples_path(key), key), final=args.final)
    sys.stdout.write(_format_report(report) + "\n")
    if args.dry_run:
        return 0
    table = WeightsTable.from_bytes(args.table.read_bytes()).with_entry(report.key, report.entry)
    if args.set_default:
        table = attrs.evolve(table, default=report.entry)
        ESTIMATOR_PATH.write_text(set_default_constants(ESTIMATOR_PATH.read_text(), report.entry))
    args.table.write_bytes(table.to_bytes())
    sys.stdout.write(f"wrote {args.table}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
