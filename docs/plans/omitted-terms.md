# Plan: no explanation leaves out a model term in silence

You are a senior Python engineer. You will make each explanation of EBaMl either cover every term in its scope or name
the terms that it leaves out, in its output and not only in a log. Read `CLAUDE.md` and every file in
`.claude/rules/` first, especially `semantic-contracts.md` (contract 4), because the hooks enforce those rules and
`prek run --all-files` must pass before you finish.

<goal>
The reader of a global description of an EBM with interactions learns from the text that the description leaves out
those terms. A matrix of tests fixes the behavior of each entry point for each kind of model.
</goal>

<context>
Found on 2026-11-09:

- `describe_graph` and `describe_ebm` in `src/ebaml/llm/descriptors.py` return `str`. `_warn_skipped_interactions`
  logs an `interactions_skipped` warning. Nothing else tells the caller, and the warning does not reach the LLM: no
  prompt in `baml_src/` mentions interactions. So the summary of `describe_ebm` presents itself as a description of the
  model.
- `README.md` documents that interactions are skipped, and `ROADMAP.md` says that they are not planned yet.
- The MCP tool of the model information names the interaction terms that the other tools skip
  (`src/ebaml/mcp/read_tools.py`, `src/ebaml/mcp/server.py`).
- Local explanations reject interactions, and a feature with no main-effect term, in `_features` of
  `src/ebaml/local/facts.py`. `explain_instance` calls `compute_local_facts` before `baml_sdk.explain_local`, so
  it raises before any LLM call. `test_features_reject_interactions` tests `_features`. Check whether a test of
  `explain_instance` asserts that no SDK call happens.
- Check what `describe_ebm` does with a feature that has no main-effect term (`exclude` or `remove_terms`).
</context>

<steps>
1. Pass the names of the skipped terms to the summary prompt of `describe_ebm`, and to `describe_graph` for the terms
   of its feature. Tell the model, with the reason, to say in one sentence that the description leaves those terms
   out. Add an offline BAML test of the prompt, and a live prompt test.
2. Ask the user whether `describe_ebm` should also return an attrs result with the text and the skipped terms. That
   change of the return type is **BREAKING**. Without it, step 1 is enough for the reader, and the log is enough for
   the program.
3. Add a test that `explain_instance` makes no SDK call for a model with interactions. Use the fake SDK of the tests
   (`fake_sdk`).
4. Write a parametrized test matrix: the model kinds (binary, multiclass, regressor) times "with interactions" and
   "with a removed term", for each public entry point (`describe_graph`, `describe_ebm`, `explain_instance`, and the
   MCP tools). Each cell states the expected behavior: named in the text and the log, rejected before the LLM, or
   named by the tool.
5. Update the "not supported yet" line of `README.md` if the behavior changes.
</steps>

<constraints>
- When interactions get support, the plan of that milestone must keep a test that an additive model gives the same
  guarantees through both code paths: a model with zero interaction effects must satisfy the same contracts as the
  model without the terms, even if a tie gives a different valid set. Write that requirement in `ROADMAP.md` next to
  the line about interactions.
</constraints>

<release>
Add a `Changed` fragment for the prompt, and a **BREAKING** entry if the return type changes.
</release>

<done>
- A description of an EBM with interactions names the skipped terms in its text.
- The test matrix covers each entry point and kind of model.
- `prek run --all-files`, `uv run pytest` and the offline BAML tests pass.
</done>
