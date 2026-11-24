### Added

- `propose_difficulties(ProposeDifficultiesRequest(ebm, instance))` asks the LLM how hard the change of each input is, on the scale of `Difficulty`, with a reason for each input. It returns a `DifficultyProposal`, whose `costs` is a `ScaledCost` to pass to `RecourseOptions(costs=...)`. It runs no search, so the caller checks the proposal first.
- `DIFFICULTY_MULTIPLIERS` gives the cost multiplier of each level: each level costs twice the level below it.
- `ScaledCost(proposed_by_model=True)` marks difficulties that a language model proposed, and the facts and the explanation then call them guesses to check.
