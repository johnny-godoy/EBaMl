### Added

- `EBMGraph.from_ebm` reads the graph of a multiclass classifier: one score per bin and per class, centred over the classes of each bin. `EBMGraph.classes` names the classes, and `to_text` writes one series per class after a `Class:` line. `from_text` reads that text back.
- `GraphOptions.classes` and the `classes` argument of `EBMGraph.from_ebm` select the classes of a multiclass graph, because the text of a model with many classes is long.

### Changed

- `EBMGraph.to_text` raises `EBMGraphShapeError` when the scores and the standard deviations do not have the same shape.
