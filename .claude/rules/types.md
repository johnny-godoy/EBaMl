# Type hints

## Inputs abstract, outputs concrete

Give each parameter the most general type that the function body needs, and give each return value the concrete type
that the function builds. The caller then passes any fitting object, and the return value has all the methods that the
caller may use. The runtime type check (`beartype`) enforces the hints, so a hint that is too narrow raises an error
for a valid call.

| Parameter holds | Hint | Not |
| --- | --- | --- |
| Items that the function only reads, in order | `Sequence[T]` | `list[T]` |
| Items that the function loops over once | `Iterable[T]` | `list[T]` |
| Items that the function only tests for membership | `Collection[T]` or `collections.abc.Set[T]` | `set[T]` |
| Key-value pairs that the function only reads | `Mapping[K, V]` | `dict[K, V]` |
| A file system path | `str \| os.PathLike[str]` | `Path` |
| A number | `float` (the numeric tower accepts `int`) | `int \| float` |

- Return `list`, `dict`, `tuple` or `Path`, as built. Do not return `Sequence`, `Mapping` or `Iterable`.
- Use a concrete parameter type only if the body needs its methods (for example `list.append`), and say why in a comment.
- A third-party class with no abstract base (such as `ExplainableBoostingClassifier`) stays as it is.
- An `attrs` field is both an input and stored state. Give it the abstract type that the class reads.
- Put an `Optional` or union in the hint when the function accepts plain strings for an enum (`Normalization | str`).
