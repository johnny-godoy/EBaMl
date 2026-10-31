### Added

- `Audience` is a base class that a user subclasses to explain a prediction to a new kind of reader. A subclass sets the instruction for the LLM (`guidance`) and the keys that the facts of the prompt get. `Layperson` and `DataScientist` are the two built-in audiences, and a subclass of either changes one rule. `explain_instance` and `facts_to_texts` accept an `Audience` object.
- The module `ebaml.local.audience` has the helpers of the built-in audiences: `likelihood_word`, `strength_word`, `round_value` and `points_towards`.

### Changed

- **BREAKING:** `Audience` is no longer an enum. Use `Layperson()` and `DataScientist()` in place of `Audience.LAYPERSON` and `Audience.DATA_SCIENTIST`. The names `"layperson"` and `"data_scientist"` still work as strings.
- **BREAKING:** The BAML functions `explain_local`, `LocalCounterfactual`, `LocalSufficientReason`, `LocalCompletions` and `local_user_msg` take the text `audience_guidance: string` in place of the name of the audience, and `audience_guidance` is no longer a BAML function.
