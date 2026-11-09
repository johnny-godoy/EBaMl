"""The progress bar of the calls that make a person wait.

The bar goes to the error stream, so the output of a program stays clean. If the stream is not a terminal, the bar
writes one new line for each step. A log file then shows the progress, and an agent that polls the console sees it.
"""

import contextlib
import io
import sys
from collections.abc import Callable, Generator

import progressbar


def _skip_step() -> None:
    """Do nothing. ``progress`` yields this function when the bar is off."""


@contextlib.contextmanager
def progress(
    total: int,
    label: str,
    *,
    enabled: bool,
    stream: io.TextIOBase | None = None,
) -> Generator[Callable[[], None]]:
    """Show a progress bar for a number of steps, and finish it when the block ends.

    Call the function that the block receives once for each finished step. Call it from one thread only, because the
    bar is not thread safe. If the block raises an error, the bar stops where it is.

    Parameters
    ----------
    total: int
        The number of steps.
    label: str
        The text before the bar.
    enabled: bool
        Whether to show the bar. If it is false, the function that the block receives does nothing, so the caller needs
        no branch.
    stream: io.TextIOBase | None, default=None
        Where to write the bar. ``None`` selects ``sys.stderr``.

    Yields
    ------
    Callable[[], None]
        The function that counts one finished step.

    """
    if not enabled:
        yield _skip_step
        return
    with progressbar.ProgressBar(max_value=total, prefix=f"{label} ", fd=stream or sys.stderr) as bar:

        def finish_step() -> None:
            # A step is a call of an LLM, which takes seconds, so each step redraws the bar. Without force, the bar
            # skips a redraw that comes soon after the last one.
            bar.increment(force=True)

        yield finish_step
