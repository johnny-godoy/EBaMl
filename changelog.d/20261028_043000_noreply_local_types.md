### Changed

- **BREAKING:** The result types of the local explanations have field names that fit every model kind, so the explanations of regressors and multiclass classifiers can come later with no breaking change. The JSON texts for the prompts, and so the explanations of binary classifiers, do not change.

  | Type | Old field | New field |
  | --- | --- | --- |
  | `Prediction` | `label` | `decision` |
  | `Prediction` | `probability` | `output` |
  | `Prediction` | `logit` | `score` |
  | `Prediction` | `logit_std` | `score_std` |
  | `Counterfactual` | `new_probability` | `new_output` |

### Added

- `Prediction.margin` is the log-odds of the predicted class: the margin of the decision, which means the same for every model kind.
- `Counterfactual.new_decision` holds the class that the prediction flips to.
- `SufficientReason.minimum` tells whether no smaller set fixes the prediction. It is always true for a binary classifier.
- The proof page of the local explanations defines the counterfactual, the sufficient reason and the completions of regressors and multiclass classifiers.

### Fixed

- The local explanations reject an EBM with two classes whose link is not `logit` with `UnsupportedModelError`, because it is not a binary classifier. Every rejection starts with "Local explanations need a binary classifier."
