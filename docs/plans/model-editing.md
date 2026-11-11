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

1. **0.7.0:** the edit API in Python, in `ebaml.edit`: every edit of GAM Changer that keeps the bins, on a model that
   stays consistent, with a reviewable history.
2. **0.9.0:** parity with GAM Changer: the edits that change the bins, the effect of an edit on data, business
   constraints, and `.gamchanger` files.
3. **0.10.0:** editing works for every model kind.

All three come before 1.0.0, so a milestone can change the API of an earlier one. Mark such a change **BREAKING** in
the changelog fragment.
</goal>

<context>
<sources>
The facts below come from three sources. Read them again when a detail matters, because the code wins over this text.

- The paper: Wang et al., "Interpretability, Then What? Editing Machine Learning Models to Reflect Human Knowledge and
  Values", KDD 2022, <https://doi.org/10.1145/3534678.3539074> (arXiv:2206.15465). Sections 3, 5.2 and 6 and
  appendix A describe the tools and what people need from them.
- The source of [GAM Changer](https://github.com/interpretml/gam-changer) at commit `2720f0c` (October 2023, the
  last commit). Its licence is MIT. Name it as the source where you follow it. The files that matter:
  `notebook-widget/gamchanger/gamchanger.py` (Python), `src/global-explanation/continuous/cont-edit.js` and
  `categorical/cat-edit.js` (the edits), `continuous/cont-history.js` (the history), `src/ebm.js` (the metrics),
  `src/sidebar/draw-feature.js` (the correlated features) and `src/GAM.svelte` (the saved file).
- `interpret` 0.7.8 as locked in `uv.lock`. Each statement about it below was measured on a fitted model.
</sources>

<gam_changer_edits>
GAM Changer edits the scores of a selection of bins of one main effect. The history names each edit by a `type`:

| `type` | Tool | What it computes |
| --- | --- | --- |
| `move` | Move | Adds one value to each selected score. |
| `increasing`, `decreasing` | Monotone | Isotonic regression of the selected scores only, weighted by the training count of each bin. The rest of the graph stays. |
| `inplace-interpolate` | Interpolate | A straight line between the first and the last selected score. The position of a bin is its left edge. The bins stay. |
| `inplace-regression` | Regression | A least-squares line through the selected scores, weighted by the bin counts, at the left edges. The bins stay. |
| `equal-interpolate`, `equal-regression` | Equal bins | Replaces the selected bins by N bins of equal width, then as above. **The bin edges change.** |
| `align` | Align | Sets each selected score to the score of the left bin, of the right bin, to the count-weighted mean, or to a typed value. |
| `delete` | Delete | Sets each selected score to 0. The term stays in the model. |

- A nominal feature has `move`, `align` and `delete` only.
- The study of the paper (section 5.2.2) rates the monotone tool and the delete tool highest. People use the monotone
  tool on a **part** of a graph (a U-shape that must rise on one side and fall on the other).
- The paper prefers the monotone tool to interpolation because it changes fewer scores. That is a design opinion. Do
  not write "minimal change" in a docstring: it is an exact claim, and `.claude/rules/proofs.md` then asks for a proof.
</gam_changer_edits>

<gam_changer_feedback>
GAM Changer treats "do not make a harmful edit" as its first priority, and gives four kinds of feedback:

- **Metric Panel.** Accuracy, balanced accuracy, ROC AUC and the confusion matrix for a binary classifier. RMSE, MAE and
  MAPE for a regressor. It shows three versions side by side (original, last edit, current) in three scopes: all rows,
  the rows in the selected bins, and a slice (the rows with one level of a nominal feature, for example `female`).
  The paper says why the scopes matter: a global metric hardly moves when a few bins change, so the harm shows only in
  the selected rows or in a subgroup (sections 4.2.1 and 6).
- **Feature Panel.** For each other feature, it compares the bin frequencies of the affected rows with those of all
  rows, and sorts the features by the distance. A person then sees that an edit of high ages affects mostly women. The
  paper names the l2 distance, and the code (`draw-feature.js`) computes the l1 distance.
- **History Panel.** Each edit is a commit: the feature, the type, a generated description, a message that the person
  writes (the reason), a time, a hash and a `reviewed` flag. A person can undo, redo, check out an earlier commit, and
  delete a commit. Deleting a commit also deletes the later commits of the same feature. Participants gave the history
  to auditors as model documentation (section 5.2.3).
- **Confirmation.** Save is refused until every commit is `reviewed`.
</gam_changer_feedback>

<gam_changer_limits>
EBaMl replaces GAM Changer and does not depend on it. These limits are the reasons, and each one is a requirement or a
decision for this plan:

- It needs Python 3.6 to 3.8, and its issues 14 and 18 report that it fails with current `interpret`.
- Its only Python way to apply edits is `get_edited_model(ebm, export)`, where `export` is the file that the widget
  saves. An agent cannot use the widget.
- `get_edited_model` takes, for each feature, the newest history entry whose `type` is not `"original"`. It ignores the
  checked-out commit and the `reviewed` flags.
- It writes `term_scores_`, `bins_[feature][0]` and `feature_bounds_`, and zeroes `standard_deviations_`. It does not
  update `bagged_scores_`, `bin_weights_` or the histograms. After a change of the bins, the shapes of those
  attributes no longer match the scores. If a pair term shares the cuts in `bins_[feature][0]`, it breaks that term.
- `get_model_data` rounds every score to 4 decimals, and `get_edited_model` writes all scores of an edited continuous
  feature. So it also changes the bins that the person did not edit, by up to 0.00005.
- It cannot edit the missing bin or the unseen bin (`term_scores_[term][0]` and `[-1]`): it drops them on export. It
  drops each sample row that has a missing value.
- It shows interaction terms and cannot edit them. It reads `intercept_[0]`, so it handles no multiclass model.
- It accepts `continuous` and `nominal` features only: `ordinal` raises (issue 16).
- It applies the logit link or the identity link only (issue 11), so its metrics are wrong for another objective.
- It fails on a model that `monotonize` changed (issue 15), because the standard deviations are `None`.
- It cannot split or extend the first or the last bin (issue 1). Users asked for it, because expert knowledge matters
  most where the model has no data.
- The saved file holds the sample rows (`sampleData`). Such a file can hold personal data.
</gam_changer_limits>

<interpret>
`interpret` has edit methods on a fitted EBM: `monotonize(term, increasing="auto", passthrough=0.0)`,
`remove_terms(terms)`, `remove_features(features)`, `scale(term, factor)`, `sweep(terms, bins, features)` and `copy()`.
Measured behavior that the design must handle:

- `monotonize` works on the whole term only, on `continuous` and `ordinal` features only, and not on a multiclass
  model. It adds a constant so that the weighted mean of the term does not change (`passthrough=0.0`). GAM Changer adds
  no constant.
- After `monotonize`, `standard_deviations_[term]`, `bagged_scores_[term]` and `bagged_intercept_` are `None`. Then
  `EBMGraph.from_ebm` raises `TypeError`, `predict_with_uncertainty` raises `TypeError`, and `scale` on the same term
  raises `TypeError`. `explain_global`, `predict_proba`, `to_jsonable`, `copy` and a skops dump still work.
- A direct write to `term_scores_` leaves `bagged_scores_` as fitted. `predict` then uses the edit, and
  `predict_with_uncertainty` silently ignores it.
- `scale` multiplies `term_scores_`, `bagged_scores_` and `standard_deviations_`. It leaves the intercept.
- `remove_terms(["age"])` keeps a pair term such as `age & sex`, so the model still uses the feature. The model accepts
  the same columns as before.
- `remove_features` removes the pair terms too, but `predict_proba` then raises `IndexError`, with or without the
  removed column. Do not build on it until a test shows that it works.
- Each fitted term has a weighted mean of 0 (weights `bin_weights_`), and the intercept holds the mean.
- `term_scores_[term]` has one row for the missing bin first and one for the unseen bin last. `EBMGraph` drops both.
- A regressor has a scalar `intercept_`, a binary classifier has shape `(1,)`, and a multiclass model has one column of
  scores for each class. `link_` names the link function.
</interpret>

<architecture>
- Editing is domain code: it changes numbers of the model and calls no LLM and no I/O. Add a new layer `ebaml.edit`
  that may import `graphs` and `core`, with an entry in the import-linter layers contract and a row in
  `.claude/rules/architecture.md`. `ebaml.mcp` and `ebaml.llm` may import it. Export its public names from `ebaml`.
- The clock is I/O. `EditSession` takes a `Callable[[], datetime]` for the time of an edit, so that a test and a replay
  are deterministic (`.claude/rules/solid.md`, D).
- Search for a contract before you add one, and register the edit interface in `architecture/abstractions.yml`
  (`.claude/rules/reuse.md`).
</architecture>
</context>

<decisions>
The user answered these questions. Each answer binds you. If the code shows that an answer no longer fits, ask the
user again, and do not reinterpret it.

1. **Uncertainty of an edited bin (contract 7).** Set the standard deviation of each edited bin to 0, as GAM Changer
   does. The session records the edited bins. The docstrings name the rule ("a score that a person sets has no fitted
   uncertainty"), and `docs/editing.md` says what it means for a local explanation of an edited model.
2. **Re-centring.** A direct edit (`Shift`, `SetScores`, `Align`, `Interpolate`, `FitLine`) never re-centres, so a
   person who sets a score to 0.38 reads 0.38. `Monotonize` adds the constant that keeps the weighted mean of the term,
   as `interpret`'s `monotonize` does with `passthrough=0.0`. `Recenter(feature)` moves the weighted mean of a term
   into the intercept and changes no prediction. Each report holds the shift of the weighted mean of the term. The MCP
   tool descriptions explain why `Recenter` matters (`mcp-edit-tools.md`).
3. **Selection.** The parameter of an edit is the bins, so no selection can end off a bin edge. A continuous or
   ordinal feature is addressed by bin position (see `<selection>`), and a nominal feature by category name. An
   interval of values is a convenience that selects every bin that overlaps it. The report lists the bins that it
   resolved to, so nobody edits a bin unseen.
4. **"This feature must not be used".** Offer `ZeroTerm` (scores to 0, the term stays), `RemoveTerms`, and
   `RemoveFeature`, which removes every term that uses the feature, pair terms included. The model keeps its input
   columns, so the pipeline around it does not change.
5. **Who reviews.** Each history entry has `reviewed: bool`. `EditSession.save` refuses unreviewed entries unless the
   caller passes `allow_unreviewed=True`. `mcp-edit-tools.md` decides how a person confirms the edits of an agent.
6. **Revisions.** Every history entry is a revision with an id, and a read of the model takes a revision (see
   `<session>`).
7. **`.gamchanger` fixture (0.9.0).** The user exports a real file from <https://interpret.ml/gam-changer/> (a
   continuous edit, a nominal edit and an `equal-*` edit). Ask for it before you write the round-trip test.
8. **Rounding of a loaded file (0.9.0).** `load_gamchanger` emits an edit only for a bin whose score differs after
   rounding to 4 decimals. The result differs from `get_edited_model` by up to 0.00005 on the other bins, and the docs
   say so.
</decisions>

<design_0_7_0>
<consistent_model>
The edited model must work everywhere a fitted model works. This is the invariant of the layer, and one private
function that writes scores owns it. No edit writes a fitted attribute itself.

- After each edit, these calls work and agree with the edit: `predict`, `predict_proba` or `decision_function`,
  `predict_with_uncertainty`, `explain_global`, `term_importances`, `EBMGraph.from_ebm`, `explain_instance`, and a
  save and load with `skops.io`.
- Keep `bagged_scores_` equal to the edit: write the new scores of the changed bins into each bag. The standard
  deviation of those bins is then 0 by computation, and `predict_with_uncertainty` gives the edited prediction.
- Never leave `None` in `standard_deviations_`, `bagged_scores_` or `bagged_intercept_`. If you call `monotonize` or
  `remove_terms` of `interpret`, repair the attributes after the call. A test loads a model that the user already
  monotonized with `interpret` (the attributes are `None`) and edits it.
- `EditSession` works on `ebm.copy()`. A test checks that no attribute of the original changes.
</consistent_model>

<selection>
An edit names its bins with a selection, not with positions in `term_scores_`, because a position is off by one
(the missing bin is first) and changes when the bins change.

- `Bins(first, last)` for a continuous or ordinal feature: positions among the real bins, counted from 0, with
  `last` included. They exclude the missing and the unseen bin, so the position of a bin is the same as in the
  graph text and in `list_bins`. `Categories([...])` for a nominal feature, `MissingBin()`, `UnseenBin()` and
  `WholeGraph()`. "A missing value scores 0" is `SetScores(feature, MissingBin(), 0.0)`.
- `Interval(low, high)` selects every bin that overlaps the interval and resolves to `Bins` before the edit runs.
  The report names the resolved bins and their edges.
- An unknown feature, an unknown category, an empty selection, or a selection that the feature type does not allow
  raises an `EBaMlError` subclass with the fix steps, for example the edges or the categories that exist.
- A feature with a pair term: the edit changes the main effect only. The report names the pair terms that still use the
  feature (contract 4, as `interactions_skipped` does), so nobody reads "monotone" as a property of the whole model.
</selection>

<edits>
One frozen `attrs` class per edit, behind one small interface (SOLID O and I): a new edit is a new class, not a new
branch. Each class says in its docstring which GAM Changer tool it follows.

| Edit | Effect | GAM Changer |
| --- | --- | --- |
| `Shift(feature, selection, delta)` | Adds a value to the selected scores. | `move` |
| `SetScores(feature, selection, value)` | Sets the selected scores to one value. | `align` to a value, `delete` |
| `Align(feature, selection, to)` | Sets them to the score of the `left` bin, of the `right` bin, or to the weighted `mean`. | `align` |
| `Monotonize(feature, direction, selection=WholeGraph())` | Isotonic regression of the selected scores, weighted by `bin_weights_`. | `increasing`, `decreasing` |
| `Interpolate(feature, selection)` | A straight line between the first and the last selected score, at the left edges of the bins. | `inplace-interpolate` |
| `FitLine(feature, selection)` | The weighted least-squares line through the selected scores. | `inplace-regression` |
| `Scale(feature, factor)` | Multiplies the graph. | none (`interpret`) |
| `ZeroTerm(feature)`, `RemoveTerms(terms)`, `RemoveFeature(feature)` | See decision 4. | `delete` on all bins |
| `Recenter(feature)` | See decision 2. | none |

- `Monotonize`, `Interpolate` and `FitLine` need an order, so they refuse a nominal feature and the missing and unseen
  bins. `Monotonize(direction="auto")` may follow `interpret`. State the rule that picks the direction.
- Use `sklearn.isotonic.IsotonicRegression` for the selected range. `interpret` cannot do a range. Declare
  `scikit-learn` as a direct dependency, because deptry fails on an import of a transitive one.
- A differential test compares `Monotonize` on the whole graph plus the mean shift with `interpret`'s `monotonize`.
</edits>

<session>
- `EditSession(ebm)`: `preview(edit)` (the report, with no change), `apply(edit, message)`, `undo()`, `redo()`,
  `checkout(revision)`, `review(revision)`, `model` (the edited copy), `model_at(revision)` (a read-only copy),
  `history` and `save(...)` (decision 5).
- A **revision** is `"original"`, `"current"` or the id of a history entry. `model_at` and the read tools take it, so
  a person or an agent can read any version without a checkout.
- `history_table()` returns the commits as rows: id, time, feature, edit type, description, reason, `reviewed`. `save`
  writes the same table next to the model.
- The id of an entry is a time-ordered UUID (version 7). Python 3.13 has no `uuid.uuid7`, and the stdlib has it from
  3.14. Evaluate and choose: (a) a small own function built from the injected clock and an injected random source
  (no dependency, deterministic in tests); (b) the `uuid-utils` package. Prefer (a) and write the reason in the PR.
  Show a short unique prefix of the random tail in tables, because the leading characters are the time.
- A history entry holds the edit, a generated description ("Set 6 bins of age (95.5 to 105) to 0.4574"), the message
  of the person (the reason, as the paper asks), the time, an id and `reviewed`. A new `apply` after an `undo` drops
  the redo stack, as GAM Changer does.
- `history` is replayable JSON, written with `orjson`, with a format version and a fingerprint of the original model
  (feature names, bin edges, a hash of the scores). `EditSession.replay(ebm, history)` refuses a model with another
  fingerprint, because bins that moved make a selection mean other rows.
- Save the model with `skops.io` (`.claude/rules/libraries.md`), and the history next to it.
</session>

<report>
`preview` and `apply` return an `EditReport`, an `attrs` class:

- The graph text before and after (`EBMGraph.to_text`). `EBMGraph` has no missing or unseen bin, so the report holds
  the scores of those two bins as fields of their own.
- The changed bins with the score before and after, the largest change, and the real range that the selection covers.
- The share of the training weight in the changed bins, from `bin_weights_`. It needs no data, and it tells the person
  how many rows the edit reaches. GAM Changer shows the same number as "623/5000 samples selected".
- The change of the intercept, and the pair terms that still use the feature.
- The text claims no more than the numbers show (contract 3): no "safer", "better" or "minimal".
</report>

- Support binary classifiers and regressors with `continuous`, `nominal` and `ordinal` features. Raise
  `UnsupportedModelError` for a multiclass model until 0.10.0.
- Evaluate each loop for a progress bar (`.claude/rules/progress-bars.md`). An edit is fast CPU work, so expect none.
</design_0_7_0>

<design_0_9_0>
- **Edits that change the bins.** `SplitBin(feature, at)`, `Rebin(feature, selection, bin_count)` followed by
  `Interpolate` or `FitLine` (the `equal-*` tools), and `ExtendRange(feature, bound)` for the first or the last bin
  (issue 1 of GAM Changer). Each one updates `bins_`, `term_scores_`, `bagged_scores_`, `standard_deviations_`,
  `bin_weights_` and `feature_bounds_` together. Split the weight of a bin by width, and say in the docstring that it
  is an estimate. If a pair term shares the cuts of the feature, give the main effect its own cuts first. A split
  alone changes no prediction: test that with Hypothesis. Decision 3 can then offer "split at the bounds".
- **The effect on data.** `edit_effect(session, data, labels=None, slice=None)` returns an `attrs` result with:
  - the metrics of the Metric Panel (accuracy, balanced accuracy, ROC AUC and the confusion matrix, or RMSE, MAE and
    MAPE) for the original model, the model before the last edit and the current model;
  - each metric in three scopes: all rows, the rows in the changed bins, and the rows of `slice`;
  - the number of rows whose prediction changes, and for a classifier the rows whose class changes, in each direction;
  - without labels, the counts only.
  Compute each prediction with the model itself (`predict`, `predict_proba`), never with a link that the code assumes.
  Keep the rows that have a missing value: GAM Changer drops them, and an edit of the missing bin affects only them.
  Take the frame with narwhals (`.claude/rules/libraries.md`).
- **The affected rows.** `affected_profile(session, data)` is the Feature Panel: for each other feature, the bin
  frequencies of the affected rows and of all rows, and their distance, sorted. Name the distance in the docstring
  (the code of GAM Changer uses l1). The purpose is fairness: the person sees which subgroup an edit reaches.
- **Business constraints.** One `attrs` class per kind, in the same pattern as the edits: `Monotone(feature,
  direction, selection)` (a range, for a U-shape), `Bounded(feature, low, high)`, `FixedScore(feature, selection,
  value)` (a missing value or a neutral category scores 0), `Unused(feature)` (pair terms included) and
  `MaxStep(feature, step)` (no jump between neighbor bins larger than `step`). `check_constraints(ebm, constraints)`
  returns the violations, each with the bins that break it. Each constraint names the edit that repairs it. A float
  tolerance is a parameter. A test applies the repair and checks that the violation is gone.
- **`.gamchanger` files.** The file is JSON with `modelData`, `sampleData` and `historyList`. An entry has `featureName`,
  `type`, `description`, `time`, `hash`, `reviewed`, `metrics` and `state.pointData`. For a continuous feature,
  `pointData` is a linked list of points (`x` the left edge, `y`, `count`, `rightPointID`) that starts at key `"0"`.
  For a nominal feature it maps the level code to a point whose `x` is the level name.
  - `load_gamchanger(path, ebm)` returns edits for the newest entry of each feature, as `get_edited_model` does, with
    the description and the `reviewed` flag of the entry. Emit `SetScores` only for a bin whose score differs after
    rounding to 4 decimals, and bin edits when the edges differ. The unedited bins then keep their full precision.
    That differs from `get_edited_model` by up to 0.00005 on those bins: document it and test with that tolerance.
  - Refuse a file whose `modelData` does not match the model (names, edges, scores to 4 decimals).
  - `save_gamchanger(session, path, data=None)` writes a file that the widget opens. It writes sample rows only when
    the caller passes `data`, and the docstring warns that the file then holds those rows.
  - `examples/edit-6-14-2023.gamchanger` of the GAM Changer repository holds an `original` entry only. Ask the user for
    a file with real edits from <https://interpret.ml/gam-changer/> (continuous, nominal, and one `equal-*` edit), and
    commit it as a fixture with its source and licence.
</design_0_9_0>

<design_0_10_0>
- Support multiclass models: an edit names the class (or all classes), and uses the model kind abstraction
  (`src/ebaml/graphs/model_kinds.py`). `interpret` refuses `monotonize` on a multiclass model, so the own isotonic code
  of 0.7.0 serves here. Only the differences between the class scores decide a multiclass prediction: say in the docs
  what an edit of one class does to the other classes.
- `edit_effect` reports the multiclass metrics and the confusion matrix of all classes.
- Check every row of the README table "Supported models": each feature works for each model kind. Add the row
  "Editing" with the first milestone and keep it true.
</design_0_10_0>

<tests>
- Unit tests for each edit, each selection and each constraint, on small fitted models in fixtures: with a missing
  value, with a nominal and an ordinal feature, with a pair term. Each test checks the scores that change and the
  scores that must not change.
- The consistency invariant, as a Hypothesis test over sequences of edits: each call of the `consistent_model` list
  works, and `predict_with_uncertainty` agrees with `predict`.
- A differential test against a reference: write each GAM Changer edit again in the test tree, from the JavaScript
  source, as plain loops over a list of points, and compare. Do the same for `get_edited_model`. Do not install
  `gamchanger`: it needs pandas, tqdm and IPython, and it fails with current `interpret`. A reference oracle changes
  only with the approval of the user (`.claude/rules/semantic-contracts.md`, "Evidence").
- Hypothesis property tests: `undo` after `apply` gives back the same model; `redo` after `undo` gives the edited
  model; replaying `history` on the original gives the same model; `Recenter` and `SplitBin` change no prediction (to
  floating point tolerance, with no "exact" claim); `Monotonize` gives a monotone selection.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- A new page `docs/editing.md` (and a `zensical.toml` entry): one example per edit, one per constraint, a table from
  each GAM Changer tool to its edit, and a section "Coming from GAM Changer" that shows how to load a `.gamchanger` file
  and lists the differences (the rounding, the missing bin, the intercept).
- A warning before the first edit example: an edit changes the predictions of the model and can add bias, so check the
  effect on data and on subgroups before you use the edited model. Add a short section "When to edit": the paper finds
  that people disagree on whether, when and how to edit (sections 5.2.4 and 6), so an edit needs a written reason and a
  review, and a retrain is sometimes the better fix.
- A marimo notebook `notebooks/model_editing.py`, because editing is the cornerstone of the milestone
  (`.claude/rules/notebooks.md`): the pneumonia-style examples of the paper (smooth a jump, flatten a dip, make a range
  monotone, neutralize a category), with the report after each edit, and no LLM call. Register it as the rule says.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
Each milestone can take more than one PR. Each PR with a user-visible change adds a changelog fragment, and the release
workflow releases it as a PATCH (`.claude/rules/changelog-and-versioning.md`). The PR that completes a milestone also
sets its version (0.7.0, 0.9.0 or 0.10.0) in `pyproject.toml`, runs `uv lock`, and marks its row of `ROADMAP.md` as
done. A good order for 0.7.0: the layer and the consistent write, the selections, the edits, the session and the
history, then the docs and the notebook.
</release>

<done>
- 0.7.0: a person can load an EBM, apply, undo, redo and review each edit of the table, read the report of each one,
  and save the edited model with its history, from Python. Every function of EBaMl and of `interpret` works on the
  edited model.
- 0.9.0: an agent through the MCP server can find the violations of a list of constraints, repair them with edits,
  and see the effect of each edit on the data, in each scope. A `.gamchanger` file loads and saves.
- 0.10.0: every feature in the README table works for every model kind.
- `prek run --all-files`, `uv run pytest`, mutation testing and the strict docs build pass for each milestone.
- The final report names each contract that the change touches and the test that guards it, and each guarantee that
  no test shows.
</done>
