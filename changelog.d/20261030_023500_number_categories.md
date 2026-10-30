### Fixed

- A local explanation accepts a row of the training data whose categories are numbers or booleans, such as a `pd.Categorical` of `1, 2, 3` or of `True, False`. interpret keeps these categories as text, so such a row raised `InvalidInstanceError: Unknown category`. A value that cannot be a category, such as a complex number, raises `InvalidInstanceError`.
