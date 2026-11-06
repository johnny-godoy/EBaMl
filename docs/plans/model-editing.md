# Plan: model editing (0.7.0, 0.9.0 and 0.10.0)

You are a senior Python engineer with a strong background in generalized additive models and in the design of
interactive tools. You will implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every
file in `.claude/rules/` first, because the project's hooks enforce those rules and `prek run --all-files` must pass
before you finish.

<goal>
Let a person or an agent change the graphs of a fitted EBM so that the model meets business requirements: for example,
"the risk must not fall when the age goes up", "a missing value scores 0", or "this feature must not be used". EBaMl
builds this in Python, and the MCP server exposes it to agents (`mcp-edit-tools.md`, 0.8.0). The work has three
milestones:

1. **0.7.0:** the edit API in Python, in `ebaml.edit`.
2. **0.9.0:** parity with GAM Changer: business constraints, the effect of an edit on data, and `.gamchanger` files.
3. **0.10.0:** editing works for every model kind.

All three come before 1.0.0, so a milestone can change the API of an earlier one. Mark such a change **BREAKING** in
the changelog fragment.
</goal>

<context>
<gam_changer>
[GAM Changer](https://github.com/interpretml/gam-changer) is a notebook widget to edit EBMs. EBaMl replaces it and
does not depend on it, for these reasons (checked on the `gamchanger` 0.1.13 wheel, the last release, from June 2023):

- Its PyPI classifiers name Python 3.6 to 3.8. EBaMl needs Python 3.13.
- Its only Python way to apply edits is `get_edited_model(ebm, export)`, where `export` is the JSON that the widget
  saves (`*.gamchanger`). An agent cannot use the widget.
- `get_edited_model` reads `export["historyList"]`. It takes, for each feature, the newest entry whose `type` is not
  `"original"`, and reads `state.pointData`: a linked list of points (`x`, `y`, `rightPointID`) for a continuous
  feature, and a map from category to point for a nominal feature. It writes `term_scores_`, `bins_` and
  `feature_bounds_`, and sets the `standard_deviations_` of the changed bins to 0 (all bins of the feature when the bin
  edges change).
- It edits main effects only. It reads `intercept_[0]` for a classifier, so it does not handle multiclass models.
- Its licence is MIT. You can read its code (Python and JavaScript) to learn the file format and the edit operations of
  its interface. Name it as the source where you follow it.
</gam_changer>

<interpret>
`interpret` 0.7.8 has edit methods on a fitted EBM: `monotonize(term, increasing="auto", passthrough=0.0)`,
`remove_terms(terms)`, `remove_features(features)`, `scale(term, factor)`, `sweep(terms, bins, features)` and `copy()`.
Build on them where they fit, and write your own operation only where none fits.
</interpret>

<architecture>
- Editing is domain code: it changes numbers of the model and calls no LLM and no I/O. Add a new layer `ebaml.edit`
  that may import `graphs` and `core`, with an entry in the import-linter layers contract and in
  `.claude/rules/architecture.md`. Export its public names from `ebaml`.
</architecture>
</context>

<design_0_7_0>
- An `EditSession` holds a copy of the EBM (`ebm.copy()`), never the original, and the list of applied edits.
  Its methods: `preview(edit)` (the result of the edit, with no change), `apply(edit)`, `undo()`, `model` (the edited
  copy) and `history` (the edits as replayable JSON, written with `orjson`).
- One frozen `attrs` class per edit operation, behind one small interface (SOLID O and I): a new operation is a new
  class, not a new branch. Start with:

  | Edit | Effect |
  | --- | --- |
  | `Monotonize(feature, direction)` | Makes the graph monotone, with `interpret`'s `monotonize`. |
  | `SetScores(feature, bins, value)` | Sets the scores of some bins (or categories) to one value. |
  | `Shift(feature, bins, delta)` | Adds a value to the scores of some bins. |
  | `Interpolate(feature, start, end)` | Replaces the scores between two bins by a straight line. |
  | `Scale(feature, factor)` | Multiplies the graph, with `interpret`'s `scale`. |
  | `RemoveFeature(feature)` | Removes the term, with `interpret`'s `remove_terms`. |

- The result of `preview` and `apply` is an `EditReport`: the graph text before and after (with `EBMGraph.to_text`), the
  bins that changed, and the largest change of a score.
- An edit sets the standard deviations of the bins that it changes to 0, as GAM Changer does, because the fitted
  uncertainty no longer applies. Write that in the docstring.
- The prediction of an EBM is the intercept plus the scores. Some edits move the mean of a graph. Decide, with the user,
  whether an edit re-centres the graph and moves the mean into the intercept. If it does, test with Hypothesis that
  re-centring alone does not change any prediction (to floating point tolerance, with no "exact" claim).
- Support binary classifiers and regressors first. Raise `UnsupportedModelError` for multiclass models until 0.10.0.
</design_0_7_0>

<design_0_9_0>
- **Business constraints.** One `attrs` class per kind of constraint, in the same pattern as the edits: for example
  `Monotone(feature, direction)`, `Bounded(feature, low, high)`, `FixedScore(feature, category, value)`, `Unused(feature)`.
  `check_constraints(ebm, constraints)` returns the list of violations, each with the bins that break it. Each
  constraint can name the edit that repairs it, so an agent can go from a violation to a fix.
- **The effect on data.** Given data with labels, `EditReport` also holds the metric before and after (AUC and accuracy
  for a classifier, RMSE for a regressor) and the number of rows whose decision changes. GAM Changer shows the same
  numbers, and people use them to judge an edit.
- **`.gamchanger` files.** `load_gamchanger(path)` turns a GAM Changer export into a list of `SetScores` edits (and bin
  changes), and `save_gamchanger(session, path)` writes one that the widget can open. Test both against the behavior of
  `get_edited_model` on a saved example file.
</design_0_9_0>

<design_0_10_0>
- Support multiclass models: an edit names the class (or all classes), and uses the model kind abstraction
  (`src/ebaml/graphs/model_kinds.py`).
- Check every row of the README table "Supported models": each feature works for each model kind.
</design_0_10_0>

<tests>
- Unit tests for each edit and each constraint, on small fitted models in fixtures. Each test checks the scores that
  change and the scores that must not change.
- Hypothesis property tests: `undo` after `apply` gives back the same model; replaying `history` on the original gives
  the same model; re-centring does not change the predictions.
- The `.gamchanger` round trip: load, apply and save, then compare with `get_edited_model` on the same file. Run
  `gamchanger` only in a test that loads it from a fixture file, not as a dependency of the package.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- A new page `docs/editing.md` (and a `zensical.toml` entry): one example per edit, one per constraint, and a section
  "Coming from GAM Changer" that shows how to load a `.gamchanger` file.
- A warning before the first edit example: an edit changes the predictions of the model, so check the effect on data
  before you use the edited model.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Each milestone can take more than one PR. Each PR with a user-visible change adds a changelog fragment, and the release
workflow releases it as a PATCH (`.claude/rules/changelog-and-versioning.md`). The PR that completes a milestone also
sets its version (0.7.0, 0.9.0 or 0.10.0) in `pyproject.toml`, runs `uv lock`, and marks its row of `ROADMAP.md` as
done.
</release>

<done>
- 0.7.0: a person can load an EBM, apply and undo each edit, and save the edited model, from Python.
- 0.9.0: an agent through the MCP server can find the violations of a list of constraints, repair them with edits, and
  see the effect of each edit on the data.
- 0.10.0: every feature in the README table works for every model kind.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass for each milestone.
</done>
