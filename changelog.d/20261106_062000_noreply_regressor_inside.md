### Added

- A regressor can explain the answer "inside" of `Interval(low, high)`. `LocalOptions.reason_search` chooses how to find the sufficient reason of such a decision: `Minimal()` (the default) returns a set from which no input can be removed, and `ExactOrMinimal()` returns the smallest set up to a limit on the number of features.
- The sufficient reason text says whether the set is the smallest or only minimal, and the prompt tells the LLM to use the matching word.

### Changed

- **BREAKING:** `Interval` is open at the low end and closed at the high end, as `Threshold` is. The answers are "at most `low`", "above `low` and at most `high`" and "above `high`". `Side` holds a tuple of `Margin` objects in place of one sign and one threshold.
