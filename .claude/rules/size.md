# Size of files, functions and classes

A reader keeps about seven things in mind. A unit that holds more makes the reader lose the overview. The limits below
follow the logical structure of the code, and `scripts/check_size.py` enforces them in `prek run --all-files`.

| Unit | Limit | How to split |
| --- | --- | --- |
| Source file | 800 lines | Move a cohesive group of functions or classes into a new module of the same layer. |
| Test file | 1200 lines | Split by the behavior under test, into modules that mirror the new source modules. |
| Function | 100 lines, from `def` to the last line | Extract private helpers, each with one job. Ruff limits the statements and branches too. |
| Class | 15 public and 10 private methods | Move a group of methods into a second class. If a class has more private methods than public ones, a second class hides inside it. |

- Dunder methods count as neither public nor private.
- The public limit is higher than the private one because a family of classes, one for each case, must repeat every
  method of its base class (`audience.py`). A class near the limit is still worth a look.
- A split is not a goal. Do not cut a unit in an artificial place only to pass. A table or one class for each case of a
  family (`solid.md`, O) may be long, because each part is simple.
- Keep related small classes in one file. The project does not require one class for each file, because result classes
  live in the module that builds them (`types.md`).
- A split of a module changes its path. Mirror it in the test tree (`architecture.md`) and keep each module name unique.
- `scripts/size_baseline.toml` lists the units that were over a limit before the check. The size of an entry may only
  fall. Lower the entry when you shrink a unit, and delete it when the unit is under the limit. Do not add an entry for
  new code, and do not raise one: split the code.
- `uv run python -m scripts.check_size --baseline` prints the units that are over a limit.
