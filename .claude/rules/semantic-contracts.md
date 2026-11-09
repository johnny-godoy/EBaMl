# Semantic contracts

Code that passes every guard can still make an explanation less correct or less honest. The code, the proofs, the
result types, the prompts and the docs must make the same claim about each explanation. These contracts protect that
agreement. They apply to `src/ebaml/local/`, `src/ebaml/graphs/`, `baml_src/` and their tests and docs.
`docs/plans/semantic-contracts.md` lists the plans that add the checks for each contract.

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

## Evidence

When a change touches a contract:

1. Name the contract and the test that guards it.
2. Add a regression, property-based or differential test. For a prompt, add an offline BAML test in
   `talk_to_ebm_test.baml` that fails on the old wording.
3. Run the checks of the change. Report each guarantee that no test or proof shows.
4. Do not expect a change to a contract, or to a reference oracle, to pass without the approval of the user. A test
   that you rewrite to fit the code proves nothing.
