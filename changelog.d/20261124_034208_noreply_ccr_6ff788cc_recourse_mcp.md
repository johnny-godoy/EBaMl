### Added

- The MCP tool `explain_prediction` takes the recourse arguments `immutable`, `allowed`, `difficulty`, `cost_strategy` (`unit` or `data`), `difficulty_from_model`, `max_changes` and `plans`, so an agent can ask for plans that a person can carry out.
- The MCP tool `propose_difficulties` (with `--llm`) asks the LLM how hard each change is, and returns the arguments to pass to `explain_prediction`.
