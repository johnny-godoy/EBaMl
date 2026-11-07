### Changed

- The counterfactual prompt tells the LLM when the set of changes is not the smallest, and that a flip of a class means that another class becomes the predicted class. The counterfactual facts have the key `is_smallest_set`, and the key `minimum_number_of_changes` is now `number_of_changes`.
- `Counterfactual.minimum` is false for the counterfactual of a chosen target class.
