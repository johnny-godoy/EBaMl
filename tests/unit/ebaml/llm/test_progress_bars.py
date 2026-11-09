"""Tests for the progress bar."""

import io

import pytest

from ebaml.llm.progress_bars import progress


def test_the_bar_writes_a_line_for_each_step_if_the_stream_is_not_a_terminal() -> None:
    stream = io.StringIO()
    with progress(3, "describe", enabled=True, stream=stream) as advance:
        advance()
        advance()
        advance()
    lines = stream.getvalue().splitlines()
    assert [f"({step} of 3)" in lines[step] for step in range(4)] == [True] * 4
    assert all(line.startswith("describe ") for line in lines)
    assert "100%" in lines[3]


def test_a_disabled_bar_writes_nothing(error_stream: io.StringIO) -> None:
    stream = io.StringIO()
    with progress(3, "describe", enabled=False, stream=stream) as advance:
        advance()
    assert not stream.getvalue()
    assert not error_stream.getvalue()


def test_the_bar_writes_to_the_error_stream_by_default(error_stream: io.StringIO) -> None:
    with progress(1, "describe", enabled=True) as advance:
        advance()
    assert "(1 of 1)" in error_stream.getvalue()


def test_a_failed_block_leaves_the_bar_where_it_is() -> None:
    stream = io.StringIO()

    def fail_after_one_step() -> None:
        with progress(3, "describe", enabled=True, stream=stream) as advance:
            advance()
            msg = "failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="failed"):
        fail_after_one_step()
    assert "(1 of 3)" in stream.getvalue()
    assert "(3 of 3)" not in stream.getvalue()
