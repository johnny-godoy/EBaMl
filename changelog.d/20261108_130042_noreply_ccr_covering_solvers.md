### Added

- `LocalOptions.search` chooses how to solve the covering problems of the sufficient reason with several margins and of the counterfactual of a chosen class: `Exact` (the default, with HiGHS, SCIP or CP-SAT through OR-Tools), `Heuristic`, `Stochastic` or `Greedy`. Each solver search has a `time_limit` in seconds.
- `SufficientReason.lower_bound` and `Counterfactual.lower_bound` give the fewest inputs that any set needs, from the LP relaxation. A set whose size equals the bound is marked as the smallest set.
- `SearchLimitError` tells that a search ended at its time limit with no set of changes for a chosen class, and names the fix.

### Changed

- **BREAKING:** `LocalOptions.reason_search` is replaced by `LocalOptions.search`, and `Counterfactual` and `SufficientReason` have the required field `lower_bound`.
- The counterfactual of a chosen class is the smallest set of changes over every choice of bins when the search proves it, and `is_smallest_set` is then true.
- The package depends on `ortools`, and the lock moves `protobuf` to 6.33.
