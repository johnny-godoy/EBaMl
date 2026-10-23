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

- Ask for each changed unit: what is its one job, what does the next new case edit, and what does it import?
- Do not split code only to satisfy a letter. Three similar lines are better than a class with one user.
