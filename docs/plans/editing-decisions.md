# Decisions for the editing milestones (0.7.0 to 0.10.0)

This file is the questionnaire for `model-editing.md` and `mcp-edit-tools.md`. Each question changes a semantic
contract (`.claude/rules/semantic-contracts.md`) or the public API, so the implementer does not choose alone.

<how_to_use>
- Read the `Status` of each question before you write code that depends on it.
- `answered`: the `Answer` binds you. Follow it and cite the question ID in the commit body.
- `pending`: ask the user the question, with its options and the recommendation, and wait. Do not code around it.
  If the answer is not needed yet, finish the work that does not depend on it first.
- If an answered question no longer fits what you found in the code, ask the user again. Do not reinterpret it.
- The person who answers writes the `Answer` and sets the status. The implementer never edits an answer.
</how_to_use>

## Q1. Uncertainty of an edited bin (0.7.0)

Contract 7. The local explanations move each score by its standard deviation, so a value of 0 makes an edited score
look certain.

| Option | Effect |
| --- | --- |
| A (recommended) | Set the standard deviation of each edited bin to 0, as GAM Changer does. The report lists the edited bins. The docstrings and `docs/editing.md` say: "a score that a person sets has no fitted uncertainty". |
| B | Keep the fitted standard deviation. An edited score then keeps an interval that no longer describes it. |

Status: pending. Answer:

## Q2. Re-centring (0.7.0)

| Option | Effect |
| --- | --- |
| A (recommended) | No edit re-centres. A person who sets a score to 0.38 reads 0.38. One explicit `Recenter(feature)` moves the weighted mean of a term into the intercept. `Monotonize` adds no constant, so it differs from `interpret`'s `monotonize`, and its docstring says so. |
| B | Every edit keeps the weighted mean of its term, as `interpret` does. Scores then differ from what the person typed. |

Status: pending. Answer:

## Q3. A selection that does not end on a bin edge (0.7.0)

| Option | Effect |
| --- | --- |
| A (recommended) | Raise an error that lists the nearest edges. The person or agent repairs the call. |
| B | Select every bin that overlaps the interval. This changes scores that the person did not name. |

Status: pending. Answer:

## Q4. "This feature must not be used" (0.7.0)

| Option | Effect |
| --- | --- |
| A (recommended) | Offer `ZeroTerm` (scores to 0, the term stays) and `RemoveTerms`. `RemoveFeature` removes every term of the feature, pair terms included. The model keeps its input columns. |
| B | Offer `ZeroTerm` only. |
| C | Offer removal only. |

Status: pending. Answer:

## Q5. Who reviews an edit (0.7.0 and 0.8.0)

GAM Changer refuses to save an edit that no person confirmed. An agent that confirms its own edit proves nothing.

| Option | Effect |
| --- | --- |
| A (recommended) | Each history entry has `reviewed`. `EditSession.save` refuses unreviewed entries unless `allow_unreviewed=True`. The MCP `save_model` asks the person through elicitation when the client supports it, and else writes "not reviewed by a person" in the history file. |
| B | The agent can mark edits as reviewed. |
| C | No review flag. |

Status: pending. Answer:

## Q6. What the MCP read tools show after an edit (0.8.0)

| Option | Effect |
| --- | --- |
| A (recommended) | The read and explain tools use the edited copy, and each result says how many edits the session holds. `get_graph` takes `version` (`original` or `current`). |
| B | The read tools always show the original model. |

Status: pending. Answer:

## Q7. Fixture of a real `.gamchanger` file (0.9.0)

The only example in the GAM Changer repository has an `original` entry and no edit. The round-trip test needs a real
export with a continuous edit, a nominal edit and an `equal-*` edit.

| Option | Effect |
| --- | --- |
| A (recommended) | The user exports one from <https://interpret.ml/gam-changer/> and commits it with its source and licence. |
| B | The implementer writes a file by hand from the JavaScript source. The test then proves only that the code reads what the implementer assumed. |

Status: pending. Answer:

## Q8. Rounding of a loaded `.gamchanger` file (0.9.0)

GAM Changer rounds every score to 4 decimals, and `get_edited_model` writes all scores of an edited feature.

| Option | Effect |
| --- | --- |
| A (recommended) | `load_gamchanger` emits an edit only for a bin whose score differs after rounding to 4 decimals. Unedited bins keep full precision. The result differs from `get_edited_model` by up to 0.00005 on those bins, and the docs say so. |
| B | Match `get_edited_model` exactly, including the rounding of unedited bins. |

Status: pending. Answer:
