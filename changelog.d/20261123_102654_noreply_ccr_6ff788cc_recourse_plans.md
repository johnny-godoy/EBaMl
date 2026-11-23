### Added

- `RecourseOptions(plans=k)` returns up to `k` diverse plans, cheapest first. As in GAM Coach, each plan after the first uses no pair of a feature and a bin of an earlier plan. The facts list the other plans, and the prompt calls each one the cheapest only without the changes of the earlier plans.

### Changed

- **BREAKING:** The field `LocalFacts.counterfactual` is replaced by the list `LocalFacts.counterfactuals`, which the constructor takes; an empty list means that no plan exists. `LocalFacts.counterfactual` is a read-only property that gives the first plan, or `None`.
- **BREAKING:** `MinimalityStatus.is_smallest` is renamed `is_optimal`, because with costs the claim is the cheapest plan, not the smallest set.
- **BREAKING:** `Counterfactual.lower_bound` (a number of changes) is replaced by `Counterfactual.cost_lower_bound` (a cost that no plan beats).
- **BREAKING:** `Change` has the new required field `to_bin`, the index of the new bin.
