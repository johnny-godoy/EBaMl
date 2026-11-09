### Added

- `DescribeGraphRequest`, `DescribeEBMRequest` and `ExplainInstanceRequest` hold the input of `describe_graph`, `describe_ebm` and `explain_instance`. Import them from `ebaml`.

### Changed

- **BREAKING:** `describe_graph`, `describe_ebm` and `explain_instance` take one request object and no other argument. Wrap the old arguments in the matching request: `describe_ebm(DescribeEBMRequest(ebm, model="openai/gpt-4o-mini"))`. The fields keep the names, the order and the defaults of the old arguments, and `graph`, `prompt`, `local`, `model` and `language` stay keyword-only.
