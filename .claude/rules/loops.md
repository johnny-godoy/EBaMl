# Loops

Two loop forms hide a better one. Each has a semgrep rule that fails `prek run --all-files`.

## No `range(len(items))`

`range(len(items))` makes an index only to reach `items[index]`. The index adds a way to be wrong (an off-by-one, or
the index of another sequence) and hides what the loop does.

| The loop needs | Write |
| --- | --- |
| Each item | `for item in items` |
| The item and its position | `for position, item in enumerate(items)` |
| The items of several sequences | `for first, second in zip(firsts, seconds, strict=True)` |
| A count | `item_count = len(items)`, then use the count |
| A result for each item | A comprehension over the items, not over the positions |

- Prefer the form with no index. Use `enumerate` only if the position is a result.
- If the index addresses several sequences, or the keys of a mapping in order, and `zip` does not fit, keep the
  `range` and add `# nosemgrep: ebaml-no-range-len -- <reason>`.
- The rule `ebaml-no-range-len` in `semgrep/ebaml.yml` checks `range(len(x))` and `range(start, len(x))`. A
  `range(len(x) + 1)` is a count, so it passes.

## Vectorize numpy loops

A Python loop over a numpy array runs in the interpreter, one element at a time. A numpy operation on the whole array
runs in compiled code, and it also shows the intent (a sum, a mask, a lookup).

- Write the operation on the whole array: a ufunc, a boolean mask, `np.where`, `np.searchsorted`, `np.cumsum` or a
  reduction.
- To get Python numbers from an array, call `.tolist()`. Do not loop to convert each element with `float()` or `int()`.
- If the loop must stay, write the reason on the line of the loop:
  `# nosemgrep: ebaml-numpy-loop-needs-reason -- <why a vector operation cannot do it>`. A reason is, for example, a
  step that needs the result of the step before it, or a loop over a few items where a vector form is longer and
  slower to read.
- The rule `ebaml-numpy-loop-needs-reason` checks a loop or comprehension over a value that is annotated
  `np.ndarray`, over the result of an `np.` call, and over `range(array.shape[...])` or `range(array.size)`. Semgrep
  infers a type only from an annotation or a constructor, so an array of another origin escapes the rule. Check the
  loops that the rule cannot see in review.
- Tests are in scope too.
