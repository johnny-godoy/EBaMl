### Removed

- **BREAKING:** `LocalFacts.counterfactual` is removed. Read the list `LocalFacts.counterfactuals`: it is empty when no plan exists, and its first item is the cheapest plan.

### Changed

- **BREAKING:** The texts of `facts_to_texts` name the claim of a counterfactual and of a sufficient reason `is_optimal`, the name of the field that they render, in place of `is_smallest_set`. The prompts read the new key.
