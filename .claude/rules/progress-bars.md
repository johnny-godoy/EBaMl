# Progress bars

A person who waits for a slow loop, and an agent that polls the console, cannot tell a loop that works from a loop that
hangs. A progress bar shows that the work goes on, and how much is left. Add one where the wait is long, and nowhere
else, because a bar on a fast loop only makes noise.

## Evaluate each new loop

When you write or change a `for` loop, a `while` loop or a pool of workers, ask the questions below. Add a bar if the
answers to the first three are "yes". Write the result in the docstring of the function if you decide against a bar for
a loop that looks slow.

| Question | A bar fits when |
| --- | --- |
| How long does one step take? | A step takes a second or more: an LLM request, a network call, a model fit, a subprocess. |
| How long does the loop take in total? | The loop takes more than a few seconds for a normal input. |
| Is the number of steps known, or bounded? | The caller knows the total, or an upper limit such as `count`. |
| Who waits? | A person or an agent runs the function or the command and sees the console. |

- A loop of fast CPU steps needs no bar: a comprehension, a solver iteration, a fact computation, a loop in a test.
- A loop in `core`, `graphs` or `local` gets no bar. Those layers have no I/O (`architecture.md`). If the caller needs
  the progress, take a `Callable[[], None]` parameter and let the `llm` layer or the script pass the bar function
  (`solid.md`, D).
- A marimo notebook uses the progress widget of marimo, `mo.status.progress_bar`, in its cells.

## Use progressbar2

- Use `progressbar2` (`import progressbar`). Ruff bans `tqdm`, `alive_progress`, `enlighten` and `rich.progress`. The
  message names this file. Do not add an ignore for a banned import. Write the bar with `progressbar2`.
- The reason for one library: one look on every console, one test fixture, and one set of limits to know. Update this
  file when you change the library.
- Write the bar to the error stream, so that the output of the program stays clean. progressbar2 does this by default.
- If the error stream is not a terminal, progressbar2 writes one line for each update. A log file and an agent that
  polls the console then see the progress. Do not turn this off.

## Where to put the bar

| Place | How |
| --- | --- |
| A public function of `ebaml.llm`, such as `describe_ebm` | Add the keyword argument `progress_bar: bool = False`, and use `ebaml.llm.progress_bars.progress`. A library never writes to the console unless the caller asks. |
| A function of `ebaml.benchmark` | Add the keyword argument `progress: bool = False`. The module imports no other module of `ebaml`, so it draws its own bar. |
| A command line (`benchmark/run.py`, `scripts/`) | Add `--progress` and `--no-progress`, on by default, because a person started the command and waits. |

## Write the bar so that it works

1. Count the steps in one thread. progressbar2 is not thread safe. For a pool, call the counting function in the thread
   that waits for the futures (`as_completed`), and not in a worker.
2. Redraw after each step with `bar.increment(force=True)`. Without `force`, progressbar2 skips a redraw that comes soon
   after the last one, and a slow loop with a few steps then shows almost nothing.
3. Give a total of at least 1. A total of 0 draws a spinner that tells nothing, so skip the bar for an empty loop.
4. Count the last step of the work too (for example the summary after the graphs), so that 100 % means done.
5. If a step fails, let the error go on. The context manager of the bar closes the bar and does not hide the error.

## Test the bar

- Use the `error_stream` fixture of `tests/unit/conftest.py`. It gives the stream that progressbar2 writes to when the
  caller passes `sys.stderr`, because progressbar2 swaps `sys.stderr` for the stream that it saw at import, and
  `capsys` does not see it.
- Write three tests for each new bar: one line for each step (check the first and the last line), no output by default
  or with the switch off, and no bar for an empty loop.

## Check

- Search for loops that call an LLM, the network or a fit: `rg -n "for .* in " src benchmark scripts`, and read each
  loop against the table above.
- `uv run ruff check` fails on a banned library, and `prek run --all-files` runs it.
