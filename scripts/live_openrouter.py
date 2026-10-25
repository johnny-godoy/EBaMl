"""Live test: real OpenRouter calls through the BAML pipelines.

Run with ``OPENROUTER_API_KEY=sk-or-... uv run python scripts/live_openrouter.py``. Skipped without a key.
Uses the free router, so it is slow and rate limited; it makes 3 + 8 requests, plus retries.
"""

import os
import sys

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from structlog.testing import capture_logs

from ebaml.llm.descriptors import describe_ebm, describe_graph
from ebaml.llm.options import PromptOptions


def _fit_small_ebm() -> ExplainableBoostingClassifier:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + 0.3 * x[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    ebm = ExplainableBoostingClassifier(feature_names=["age", "bmi"], interactions=0, max_rounds=50, random_state=0)
    return ebm.fit(x, y)


def _check(condition: object, message: object) -> None:
    """Exit with a failure message unless ``condition`` is truthy.

    Parameters
    ----------
    condition: object
        The value that must be truthy.
    message: object
        What to report when it is not.

    Raises
    ------
    SystemExit
        If ``condition`` is falsy.

    """
    if not condition:
        msg = f"live test failed: {message}"
        raise SystemExit(msg)


def main() -> int:
    """Run both pipelines against OpenRouter and check the output and usage logs.

    Returns
    -------
    int
        The exit code, 0 on success or when no API key is set.

    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return 0

    ebm = _fit_small_ebm()

    with capture_logs() as logs:
        description = describe_graph(
            ebm,
            0,
            num_sentences=3,
            prompt=PromptOptions(dataset_description="Synthetic data, label is mostly x0 > 0."),
        )
        _check(description.strip(), "empty description")

        summary = describe_ebm(ebm, num_sentences=5, max_concurrency=2)
        _check(summary.strip(), "empty summary")

    usages = [log for log in logs if log["event"] == "usage"]
    _check(usages, logs)
    _check(all(log["input_tokens"] > 0 and log["output_tokens"] > 0 for log in usages), "provider reported no tokens")
    whats = [log["what"] for log in usages]
    _check(any(what.startswith("describe_graph(age)") for what in whats), whats)
    _check(any(what.startswith("describe_ebm:") for what in whats), whats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
