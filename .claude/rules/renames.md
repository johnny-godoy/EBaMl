# Renames and names across boundaries

A rename changes a name everywhere it is used, in the same change. Agents tend to keep the old name alive to make the
diff smaller: an alias, a convenience property, or an old key that the prompts still read. Each one leaves the code
with two names for one thing, or with a name that no longer says what the value is. The size of a diff is never a
reason to keep a worse API. Split a large change into commits instead (`commits-and-prs.md`, `Large-Commit`).

## Rename every use

- Change every caller of a renamed name in the same pull request: `src/`, the tests, the docs, the notebooks, the
  scripts, the BAML prompts and their tests, and the output of the MCP tools.
- Do not add a property, an alias or a wrapper that gives the old name to the new value. A public member needs a use
  that the API does not already serve, and a name that says what it returns. `facts.counterfactual` for
  `facts.counterfactuals[0]` fails both tests: the caller can index the list, and the name hides that a list exists.
- Read a list where it is used. If the code needs the first item, write `plans[0]` after the check, or unpack a list
  of one item with `(plan,) = plans`, so the reader sees the assumption.
- The package has no users yet and promises no backward compatibility (`writing-style.md`). Mark the rename
  **BREAKING** in the changelog fragment instead of keeping the old name.

## One name across a boundary

A value that crosses from Python into a prompt, a BAML type, a JSON text or an MCP result keeps the name of the Python
field that it comes from. A second name for the same value makes the reader of a prompt, a log or a tool result
translate between them, and the two drift apart.

- When a field is renamed, rename its key in `fact_text.py`, its quoted key in `baml_src/`, the BAML tests, the live
  tests and the docs that show the JSON. A prompt is code: a larger prompt diff is not a reason to keep a stale key.
- If the meaning of a field changes, rename the field and its key. Do not keep a key and tell the LLM to read it with
  another meaning.

## Guards

| Guard | What it catches |
| --- | --- |
| `test_each_key_that_a_prompt_reads_is_a_key_of_the_facts` in `tests/contract/test_fact_keys.py` | A key that `baml_src/local_explanations.baml` quotes but that the facts no longer emit. |
| `test_each_claim_key_has_the_name_of_the_field_that_it_renders` in the same file | A claim key of the facts that is not the name of the `MinimalityStatus` property that it renders. |
| `tests/api/public_api.json` | Every new public member. Review the diff of the snapshot for an alias. |

No tool sees every alias, and a pattern rule for one kind of alias would only repeat the last mistake. Before you
finish a rename, search for the old name (`rg -n "<old name>"`) over the whole repository, and read each match.
