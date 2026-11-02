### Added

- `ebaml` now exports the functions, the option classes, the local-explanation facts and the audience helpers that the docs use, so `from ebaml import describe_ebm` works. `import ebaml` still loads no BAML code: `describe_graph`, `describe_ebm`, `explain_instance` and `LocalExplanation` load on first use.
