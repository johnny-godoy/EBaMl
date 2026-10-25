# Return attrs classes, not containers

A function that returns several named values returns an `attrs` class. A `dict` or a `tuple` of fields hides the names
and the types of the fields: a type checker cannot see a wrong key or a swapped position, and a caller must remember
the order. An attrs class has one name and one type for each field, and `beartype` checks them at run time.

## Rule

- Return `@attrs.frozen` (or `@attrs.define`) for a fixed set of fields. Do not return `dict[str, ...]`,
  `tuple[A, B, ...]`, `list[dict[...]]` or `list[tuple[...]]`.
- Put a result class in the module that builds it, and give it a docstring. A private helper may return a private class.
- Pass the fields of an attrs class on with `attrs.asdict(x, recurse=False)` where a `**` call needs them.
- `types.md` still applies: a parameter is abstract, and a return value is concrete.

## When a container is right

A container is right if the structure is the data, not a set of fields. Add an inline
`# nosemgrep: ebaml-return-attrs-not-container -- <reason>` to the line of the return annotation, with the reason:

- The keys are data that change from call to call (the intervals of a graph line), so no fixed fields exist.
- The value is a fragment of a JSON object that the caller merges and serializes, and the keys are the JSON schema.
- The tuple is a homogeneous pair or run, such as an interval `tuple[float, float]`. `tuple[float, ...]` needs no
  ignore.

## Guard

The semgrep rule `ebaml-return-attrs-not-container` in `semgrep/ebaml.yml` fails `prek run --all-files` on a return
annotation of these types in `src/ebaml/`. Its test is in `semgrep/ebaml.py`. Fix the code first. The ignore is a last
resort, like the others in `prek-and-ignores.md`.
