### Added

- `RecourseOptions(costs=...)` makes the counterfactual the cheapest plan under a cost strategy: `UnitCost` (the default, the fewest changes), `DataCost` (distances in the training data, as GAM Coach computes them), `UserCost` and `ScaledCost`. `CostStrategy` is the protocol of a strategy of your own.
- `RecourseOptions(max_changes=...)` caps the number of changes of the counterfactual.
- `Counterfactual.cost` holds the cost of the plan. With costs that are not all equal, `MipCover` solves the integer program of the plans, and `GreedyCover` and `LocalSearchCover` return a valid plan marked `UNPROVED`.
- The counterfactual facts and prompt name the costs and the cap, so the explanation calls a plan the cheapest under the given difficulties and never the fewest changes.

### Changed

- **BREAKING:** `Counterfactual` has the new required field `cost`. Code that builds a `Counterfactual` by hand must pass it.
