### Fixed

- The `usage` log event of `describe_graph` and `describe_ebm` has `truncated_attempts`, the number of attempts that `max_tokens` cut off. The token usage leaves out those attempts, because the provider error carries no count, so it is a lower bound.
