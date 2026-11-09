# Plan: prompts that claim no more than the facts

You are a senior engineer who writes LLM prompts and their tests. You will make the local explanation prompts of the
EBaMl repository state only what the computed facts support. Read `CLAUDE.md` and every file in `.claude/rules/`
first, especially `semantic-contracts.md` (contract 3) and `writing-style.md`, because the hooks enforce those rules
and `prek run --all-files` must pass before you finish.

<goal>
Each sentence of `counterfactual_question`, `sufficient_reason_question` and `completions_question` in
`baml_src/local_explanations.baml` holds for every option value that `LocalOptions` accepts. A condition that depends
on an option reaches the prompt as a key of the facts, and the prompt reads the key. The prompt does not state the
condition as a fixed sentence.
</goal>

<context>
Pull request johnny-godoy/EBaMl#124 fixed three claims: "the prediction cannot be changed", "every input matters" and
"the share grows". These claims remain, found on 2026-11-09:

1. The counterfactual question says that "the flip holds even if the model's effects are at the unfavourable edge of
   their confidence interval". With `confidence_level=0`, `z` is 0 and nothing moves. The facts say so:
   `Layperson.flip` in `src/ebaml/local/audience.py` gives `flip_holds_even_if_the_model_is_somewhat_off: false`, and
   `DataScientist.flip` gives `flip_holds_at_confidence: 0`. The fixed sentence contradicts the facts.
2. The same sentence says "the model's effects". Only the features that the counterfactual changes move by `z`
   (`_counterfactual` in `src/ebaml/local/facts.py`, and the section "change and robust margin" of
   `proofs/papers/local-explanations/paper.typ`). `docs/local-explanations.md` says "the effects of the changed
   features", which is correct.
3. The sufficient reason question says that the guarantee "holds even when the model's effects are at the
   unfavourable edge of their confidence interval". This sentence is also fixed. For a layperson, the sufficient
   reason facts have no key about confidence at all: `Layperson.margin` returns `{}`.
4. Both questions say "rare extreme values are left out". With `min_support=0`, `_plausible_bins` leaves out nothing.
5. The completions question says "a large random sample". `LocalOptions.n_samples` accepts any count.

The offline tests that pin the wording are in `baml_src/talk_to_ebm_test.baml` ("the questions make no claim that the
facts do not support"). The live prompt tests are in `baml_src/prompt_tests.baml`, and they need a real key.
</context>

<steps>
1. Write the offline tests first and see them fail. For each claim above, assert that the question text makes the
   claim conditional on a named key of the facts.
2. Add the keys in Python, in the audience classes, so that each audience decides how to state them:
   - the counterfactual and the sufficient reason get a key that says whether the scores moved (`confidence_level >
     0`). The sufficient reason needs it for the layperson too;
   - both get a key that says whether rare values were left out (`min_support > 0`);
   - the completions get the sample count for a data scientist, and a word for a layperson.
   `facts_to_texts` reads `LocalFacts`, so `LocalFacts` must carry `min_support` and `n_samples`, as it carries
   `confidence_level`. Keep the rendering in `fact_text.py` and `audience.py`, and the computing in `facts.py`
   (`solid.md`, S).
3. Rewrite the sentences as conditions on the keys ("If `...` is true, say that ..."), and give the reason for each,
   as `writing-style.md` asks for prompts. Name "the changed features" in place of "the model's effects".
4. Add unit tests in `tests/unit/ebaml/local/test_fact_text.py` for each new key and each audience.
5. Add one live prompt test in `prompt_tests.baml` with `confidence_level=0`: the answer must not say that the flip
   survives uncertainty. Run `baml test -i root::Live` with a real key if you have one. If you have none, say so in
   the PR.
6. Run `baml generate`, `baml fmt`, `baml check`, `OPENROUTER_API_KEY=unused baml test -x root::Live` and
   `uv run pytest`.
</steps>

<constraints>
- The new keys are part of the fact text, which is not public API. The `LocalFacts` fields are public, so a new field
  needs a changelog entry under `Added`.
- Do not add a check in Python of the LLM output. Such a check needs a parser of free text, and it costs more than it
  saves. The offline tests guard the prompt, and the live tests guard the model.
</constraints>

<release>
Add a changelog fragment: `Fixed` for the wording, `Added` for the new `LocalFacts` fields.
</release>

<done>
- Every sentence of the three questions holds at `confidence_level=0`, at `min_support=0` and at a small `n_samples`.
- An offline test fails for each claim on the code before the change.
- `prek run --all-files`, `uv run pytest` and the offline BAML tests pass.
</done>
