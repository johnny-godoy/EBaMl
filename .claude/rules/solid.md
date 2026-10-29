# SOLID

Design code to the five SOLID principles. Each rule below has its reason and the way to apply it in this project.
Review a change against them before you finish. If you break one on purpose, write the reason in a comment.

## S: one reason to change

A module, class or function has one job. If the description needs the word "and", split it.

- Why: a unit with two jobs changes for two reasons, and each change risks the other job.
- Apply: keep the Python code apart from the prompts in `baml_src/`, and the computing of facts apart from their
  rendering as text. A function that reads data, decides and formats the result is three functions.

## O: extend by adding, not by editing

Adding a case (an audience, a feature format, a task) must not need an edit in a long `if`/`match` chain.

- Why: each edit of working code can break the cases that exist.
- Apply: put the cases in a table (`_DESCRIPTIONS`) or in one class per case, and look the case up by key. A chain that tests the same flag in many places (`if expert:`) is the sign to use classes.

### Tell, don't ask

Tell an object what to do, and let its class decide how. Do not ask for its type and decide in the caller.

- Why: a branch on the type (`if isinstance(x, A): ... elif isinstance(x, B): ...`) is a chain on one value. Each new
  class needs an edit of every such branch, and a missed branch fails only at run time.
- Apply: give the classes a common method, abstract in the base class, and call it (`kind.describe()`). If ebaml does
  not own the classes, register one function per class with `functools.singledispatch`. A `match` with two or more
  class patterns on one value is the same branch.
- One check that narrows a union where a value enters ebaml (a string or a list for `classes=`) is not a branch on the
  type. Keep it, and give it a comment if the reason is not clear.
- The semgrep rule `ebaml-polymorphism-not-type-branches` in `semgrep/ebaml.yml` enforces this in `src/ebaml/`. If a
  type branch must stay (a parsed record of an untyped format), add
  `# nosemgrep: ebaml-polymorphism-not-type-branches -- <reason>` on the line of the first branch.

## L: a subtype keeps the promise of its base

A subclass or a fake works wherever its base works: same argument types, same result type, no new errors.

- Why: callers rely on the base contract and cannot check each subtype.
- Apply: do not narrow a parameter type or raise a new error in an override. A test double must obey the contract of
  the real object, or the test proves nothing.

## I: small interfaces

A function takes only what it uses, and a class exposes only what its callers need.

- Why: a wide dependency couples a unit to things it does not use, so unrelated changes break it.
- Apply: pass the values that the function reads, not the whole `ebm` or a `**kwargs` bag that it filters later. Keep an
  abstract base class to a few methods that all subclasses use.

## D: depend on abstractions at the edges

Domain code (graphs, facts, parsing) must not import I/O or vendor code. The edge code (`descriptors.py`, the
`baml_sdk` calls, the LLM clients) depends on the domain, and not the other way round.

- Why: domain code with no vendor import runs in a plain unit test, and a provider change stays at the edge.
- Apply: if domain code needs a service (the LLM, a token counter), take it as a parameter typed by a `Protocol` or a
  `Callable`, and let the edge code pass the real one. Create the dependency in the caller, not inside the function.

## Check

- `architecture.md` lists the layers and the tools that enforce them: import-linter, the layout check and semgrep. They
  cover the mechanical part of S, O, L and D. You judge the rest.
- Ask for each changed unit: what is its one job, what does the next new case edit, and what does it import?
- Do not split code only to satisfy a letter. Three similar lines are better than a class with one user.
