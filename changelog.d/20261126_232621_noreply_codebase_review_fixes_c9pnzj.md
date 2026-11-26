### Fixed

- `EBMGraph.to_text`, the graph image and `confidence_factor` raise `OptionRangeError` for a `confidence_level` that is not between 0 and 1. A level of 1 wrote infinite bounds that are not valid JSON, and a level such as 95 failed in SciPy.
- `EBMGraph.from_text` and `parse_graph_text` raise `GraphTextError` for bounds at a confidence level of 0%, and for lower and upper bounds at two different levels. The parser divided by zero for the first, and used the wrong level for the second.
