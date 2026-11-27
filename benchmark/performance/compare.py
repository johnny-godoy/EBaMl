"""Compare two runs of the performance benchmark: the base and the head of a pull request, on the same runner.

The comparison reads the median time of each workload. A workload is slower when the median of the head is above the
median of the base by more than the threshold, and by more than a floor in seconds. The floor keeps the jitter of the
fast workloads, which take a few milliseconds, out of the verdict. No limit applies to an absolute time, because a CI
runner changes its speed from one job to the next.
"""

from collections.abc import Sequence

import attrs

from benchmark.performance.summary import RunSummary

# A workload is slower if its median grows by more than this share. ``benchmark/README.md`` says how to calibrate it.
DEFAULT_THRESHOLD = 0.25
# A change smaller than this many seconds is jitter, whatever its share.
DEFAULT_FLOOR_SECONDS = 0.002


@attrs.frozen
class Change:
    """The change of one workload from the base to the head.

    Attributes
    ----------
    name: str
        The name of the workload.
    base_seconds: float
        The median of the base.
    head_seconds: float
        The median of the head.
    is_slower: bool
        Whether the change passes both the threshold and the floor.

    """

    name: str
    base_seconds: float
    head_seconds: float
    is_slower: bool

    @property
    def ratio(self) -> float:
        """Divide the median of the head by the median of the base.

        Returns
        -------
        float
            The ratio: 1.0 for no change, above 1.0 for a slower head.

        """
        return self.head_seconds / self.base_seconds


@attrs.frozen
class Comparison:
    """The comparison of two runs.

    Attributes
    ----------
    changes: list[Change]
        One change for each workload that both runs measured, in the order of the head.
    only_base: list[str]
        The workloads that only the base measured.
    only_head: list[str]
        The workloads that only the head measured, such as a new workload.

    """

    changes: list[Change]
    only_base: list[str]
    only_head: list[str]

    @property
    def slower(self) -> list[Change]:
        """List the workloads that are slower in the head.

        Returns
        -------
        list[Change]
            The changes that pass the threshold and the floor.

        """
        return [change for change in self.changes if change.is_slower]


def compare(
    base: RunSummary,
    head: RunSummary,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    floor_seconds: float = DEFAULT_FLOOR_SECONDS,
) -> Comparison:
    """Compare the median time of each workload of two runs.

    Parameters
    ----------
    base: RunSummary
        The run of the base.
    head: RunSummary
        The run of the head.
    threshold: float
        The share of growth of the median above which a workload is slower.
    floor_seconds: float
        The growth in seconds below which a workload is not slower.

    Returns
    -------
    Comparison
        The change of each workload that both runs measured.

    """
    changes = []
    for head_timing in head.workloads:
        base_timing = base.timing(head_timing.name)
        if base_timing is None:
            continue
        growth = head_timing.median_seconds - base_timing.median_seconds
        changes.append(
            Change(
                name=head_timing.name,
                base_seconds=base_timing.median_seconds,
                head_seconds=head_timing.median_seconds,
                is_slower=growth > threshold * base_timing.median_seconds and growth > floor_seconds,
            ),
        )
    head_names = {timing.name for timing in head.workloads}
    base_names = {timing.name for timing in base.workloads}
    return Comparison(
        changes=changes,
        only_base=[timing.name for timing in base.workloads if timing.name not in head_names],
        only_head=[timing.name for timing in head.workloads if timing.name not in base_names],
    )


def _seconds(value: float) -> str:
    return f"{value * 1000:.1f} ms" if value < 1 else f"{value:.2f} s"


def _names(title: str, names: Sequence[str]) -> list[str]:
    return [f"{title}: {', '.join(f'`{name}`' for name in names)}.", ""] if names else []


def to_markdown(comparison: Comparison, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Write a comparison as the Markdown of the CI report.

    Parameters
    ----------
    comparison: Comparison
        The comparison.
    threshold: float
        The threshold of the comparison, for the text.

    Returns
    -------
    str
        A verdict line and a table with the median of the base and the head of each workload.

    """
    slower = comparison.slower
    verdict = (
        f"{len(slower)} workloads are more than {threshold:.0%} slower: "
        + ", ".join(f"`{change.name}`" for change in slower)
        + "."
        if slower
        else f"No workload is more than {threshold:.0%} slower."
    )
    lines = [
        "## Performance: base against head",
        "",
        verdict,
        "",
        *_names("Only the head measures", comparison.only_head),
        *_names("Only the base measures", comparison.only_base),
        "| Workload | Base median | Head median | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| `{change.name}` | {_seconds(change.base_seconds)} | {_seconds(change.head_seconds)} | "
        f"{change.ratio - 1:+.0%}{' (slower)' if change.is_slower else ''} |"
        for change in comparison.changes
    ]
    return "\n".join(lines) + "\n"
