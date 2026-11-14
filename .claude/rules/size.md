# Size of files, functions and classes

A reader keeps about seven things in mind. A unit that holds more makes the reader lose the overview. The limits below
follow the logical structure of the code, and `scripts/architecture/size.py` enforces them in `prek run --all-files`.

| Unit | Limit | How to split |
| --- | --- | --- |
| Source file | 800 lines | Move a cohesive group of functions or classes into a new module of the same layer. |
| Test file | No limit | A test is a list of independent cases, and it mirrors one source module. Split a test file only when its source module splits. |
| Function | 100 lines, from `def` to the last line, without the docstring | Extract private helpers, each with one job. Ruff limits the statements and branches too. |
| Class | 15 public and 10 private methods | Move a group of methods into a second class. If a class has more private methods than public ones, a second class hides inside it. |

- A docstring does not count, because a limit on it would push authors to write less documentation.
- Dunder methods count as neither public nor private.
- The public limit is higher than the private one because a family of classes, one for each case, must repeat every
  method of its base class (`audience.py`). A class near the limit is still worth a look.
- A split is not a goal. Do not cut a unit in an artificial place only to pass. A table or one class for each case of a
  family (`solid.md`, O) may be long, because each part is simple.
- Keep related small classes in one file. The project does not require one class for each file, because result classes
  live in the module that builds them (`types.md`).
- A split of a module changes its path. Mirror it in the test tree (`architecture.md`) and keep each module name unique.
