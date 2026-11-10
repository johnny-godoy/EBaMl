# Semantic contracts

Code that passes every guard can still make an explanation less correct or less honest. The code, the proofs, the
result types, the prompts and the docs must make the same claim about each explanation. These contracts protect that
agreement. They apply to `src/ebaml/local/`, `src/ebaml/graphs/`, `baml_src/` and their tests and docs.
The table below names the test that guards each contract.

## Contracts

1. **Proofs match code.** A Lean theorem describes the model in `proofs/`, not the Python code. When you change code
   that a paper covers, check it against an independent reference implementation of the mathematical specification
   (for example, exhaustive search on a model with four features). Do not weaken the specification to fit the code.
2. **Name the strength of a guarantee.** Keep minimum, minimal and unproved results apart in the result types
   (`is_smallest_set`) and in the text. A fallback never keeps a stronger guarantee than the search established.
3. **The text claims no more than the facts.** Python computes each number. A prompt must not say that a property
   holds (growth, optimality, impossibility, necessity, certainty) unless the facts show it. A failed search does not
   prove that no answer exists, because the search covers only the realistic values.
4. **Do not omit model parts in silence.** An explanation either covers every term in its scope, or it names what it
   leaves out (`interactions_skipped`). A local explanation of an unsupported model fails before any LLM call.
5. **Agree with the model.** The reported class, scores, bins and feature values must match the fitted EBM. A
   counterfactual must flip the decision under the stated assumptions. A sufficient reason must keep the decision
   for every allowed completion.
6. **Samples are estimates.** Describe completion shares as estimates under the stated sampling assumptions
   (independent inputs, training-bin weights, fixed seed). Never describe them as exact frequencies.
7. **Name the uncertainty model.** A confidence-adjusted guarantee holds under the stated assumptions only. More
   uncertainty must never turn a failed robust guarantee into a valid one.

## Guards

| Contract | Guard |
| --- | --- |
| 1 | `test_the_facts_match_the_oracle_of_the_paper` in `tests/unit/ebaml/local/test_facts.py`. The oracle is in `tests/unit/ebaml/local/conftest.py`. |
| 2 | `test_each_way_to_find_a_sufficient_reason_reports_the_guarantee_that_it_has` and `test_a_counterfactual_of_a_target_class_reports_the_guarantee_of_its_search` in `test_facts.py`. |
| 3 | The offline tests of `baml_src/tests/local_explanations_test.baml`, and `test_the_facts_claim_no_condition_that_the_options_turn_off` in `test_fact_text.py`. |
| 4 | `test_each_entry_point_names_or_rejects_a_model_with_an_interaction` and its twin for a removed term in `tests/unit/ebaml/llm/test_descriptors.py`, and `test_the_tools_name_or_reject_a_model_that_they_cover_in_part` in `tests/unit/ebaml/mcp/test_server.py`. |
| 5 | The three `..._fitted_model...` tests in `test_facts.py`. |
| 6 | `test_each_completion_share_estimates_the_exact_weighted_count` in `test_facts.py`. |
| 7 | `test_more_confidence_never_makes_a_claim_easier` in `test_facts.py`. |

## Evidence

When a change touches a contract:

1. Name the contract and the test that guards it.
2. Add a regression, property-based or differential test. For a prompt, add an offline BAML test in
   the test file of the prompt in `baml_src/tests/` that fails on the old wording.
3. Run the checks of the change. Report each guarantee that no test or proof shows.
4. Do not expect a change to a contract, or to a reference oracle, to pass without the approval of the user. A test
   that you rewrite to fit the code proves nothing.
