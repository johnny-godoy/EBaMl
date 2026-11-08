### Added

- `LocalOptions.search` chooses how to solve the covering problems of the sufficient reason with several margins and of the counterfactual of a chosen class: `Exact` (the default, with HiGHS, SCIP or CP-SAT through OR-Tools), `Heuristic`, `Stochastic` or `Greedy`. Each solver search has a `time_limit` in seconds.
- `SufficientReason.lower_bound` and `Counterfactual.lower_bound` give the fewest inputs that any set needs, from the LP relaxation. A set whose size equals the bound is marked as the smallest set.
- `Audience.lower_bound` decides whether the facts give the lower bound: `DataScientist` writes `smallest_possible_number_of_changes` and `smallest_possible_number_of_features`, and `Layperson` leaves them out. The prompts tell the LLM that it may state the gap when the set is not proved smallest.
- `SearchLimitError` tells that a search ended at its time limit with no set of changes for a chosen class, and names the fix.

### Removed

- **BREAKING:** `ebaml.local.reason_search` with `ReasonSearch`, `Minimal`, `ExactOrMinimal` and `Selection`. Use a `CoveringSearch` in `LocalOptions.search`.

### Changed

- **BREAKING:** `LocalOptions.reason_search` is replaced by `LocalOptions.search`, and `Counterfactual` and `SufficientReason` have the required field `lower_bound`.
- The counterfactual of a chosen class is the smallest set of changes over every choice of bins when the search proves it, and `is_smallest_set` is then true.
- **BREAKING:** `Audience` has the abstract method `lower_bound`, so a custom audience must define it.
- When `is_smallest_set` is false, the prompts tell the LLM to describe the set without a word about its size, in place of calling it minimal.
- The package depends on `ortools`, and the lock moves `protobuf` to 6.33.
