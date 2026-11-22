### Added

- `LocalOptions(recourse=RecourseOptions(...))` limits the counterfactual to changes that a person can make: `immutable` names the features that must not change, and `allowed` gives the values of a feature as a `ValueRange` or as `Levels`.
- `ValueRange(match=...)` chooses how a range selects the bins of a continuous feature: `RangeMatch.OVERLAP` (the default) or `RangeMatch.INSIDE`.
- `RecourseOptionError` tells that the recourse options name an unknown feature, do not fit the type of a feature, or leave no plausible value.
- The counterfactual facts and prompt name the immutable features and the allowed values, so the explanation claims "smallest" only within these limits.
