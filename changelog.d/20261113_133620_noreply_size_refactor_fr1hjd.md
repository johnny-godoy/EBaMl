### Changed

- **BREAKING:** The result classes of the local facts (`Change`, `CompletionStep`, `Counterfactual`, `SufficientReason`, `LocalFacts`, `LocalOptions`, `Prediction`, `Contribution`) moved from `ebaml.local.facts` to `ebaml.local.fact_types`. Import them from `ebaml`, as in `from ebaml import LocalFacts`. `compute_local_facts` stays in `ebaml.local.facts`.
