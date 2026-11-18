# Editing a model

An edit changes the graph of one feature of a fitted EBM, so that the model meets a requirement of the business. For
example: "the risk must not fall when the age goes up", "a missing value scores 0", or "this feature must not be used".
The edit tools of EBaMl follow the tools of [GAM Changer](https://github.com/interpretml/gam-changer) (Wang et al., KDD
2022) that keep the bins. The code calls no LLM.

!!! warning "An edit changes the predictions of the model, and it can add bias"
    An edit moves the scores of the bins that you select. It also moves the predictions for every row in those bins.
    A change of a few bins hardly moves a global metric, but it can move the predictions for one group by a large
    amount. Check the effect on your data, and on each subgroup, before you use an edited model.

## When to edit

People disagree on whether, when and how to edit a model (Wang et al., sections 5.2.4 and 6). So an edit needs a
written reason and a review by a person. A retrain with a constraint, or with more data, is sometimes the better fix.

- Write the reason of each edit in the `message`. The history keeps it as documentation of the model.
- Read the report of an edit before you apply it. It names the bins that change and the share of the training weight
  that they hold.
- Mark an edit as reviewed only after a person has read it. A save refuses edits that nobody reviewed.

## A first edit

An `EditSession` works on a copy of the model, so the model that you pass does not change.

```python
from interpret.glassbox import ExplainableBoostingClassifier

from ebaml import Bins, EditSession, Monotonize

ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=2).fit(X, y)
session = EditSession(ebm)

edit = Monotonize("age", "decreasing", Bins(0, 4))
print(session.preview(edit).to_text())  # the report, and no change of the model

report = session.apply(edit, "The chance of survival must not rise with age")
print(report.bins_changed, report.weight_share)
```

`preview` computes the report and changes nothing. `apply` makes the edit a new revision. The report holds:

- the bins that changed, with the score before and after, and the largest change;
- the share of the training weight in the changed bins, which tells how many training rows the edit reaches;
- the shift of the weighted mean of the graph, and the change of the intercept;
- the pair terms that still use the feature, because an edit changes the main effect only;
- the graph text before and after the edit.

## Select the bins

An edit names its bins with a selection. A selection never names a position of `term_scores_`, because that position
is off by one and moves when the bins change.

| Selection | Bins |
| --- | --- |
| `Bins(first, last)` | The real bins of a continuous or ordinal feature, from bin number `first` to `last`, both included. The numbers start at 0, as in the graph text. |
| `Categories(["a", "b"])` | The bins of some categories of a nominal feature. |
| `ValueInterval(low, high)` | Every bin of a continuous feature that overlaps the interval. A bound of `None` is open. |
| `WholeGraph()` | Every real bin. |
| `MissingBin()` | The bin of the missing values. The graph does not show it. |
| `UnseenBin()` | The bin of the values that the model did not see in training. |
| `EveryBin()` | The missing bin, every real bin and the unseen bin. |

A selection that does not fit the feature raises a `SelectionError` that lists the bins, categories or values that
exist. The report of an edit names the bins that a `ValueInterval` resolved to, so nobody edits a bin unseen.

## The edits

Each edit is a frozen class. The table shows the edit of each tool of GAM Changer.

| Edit | Effect | GAM Changer |
| --- | --- | --- |
| `Shift(feature, selection, delta)` | Adds one value to the selected scores. | Move |
| `SetScores(feature, selection, value)` | Sets the selected scores to one value. | Align to a value, Delete |
| `Align(feature, selection, to)` | Sets the selected scores to the score of the `"left"` bin, the `"right"` bin, or the weighted `"mean"`. | Align |
| `Monotonize(feature, direction, selection)` | Makes the selected scores monotone with a weighted isotonic regression. | Monotone |
| `Interpolate(feature, selection)` | Draws a straight line from the first to the last selected score. | Interpolate |
| `FitLine(feature, selection)` | Replaces the selected scores by the weighted least-squares line. | Regression |
| `Clip(feature, selection, low, high)` | Moves each selected score outside the range from `low` to `high` to the nearer bound. | none |
| `LimitSteps(feature, step, selection)` | Replaces the selected scores by a weighted least-squares fit with no step between neighbour bins above `step`. | none |
| `Scale(feature, factor)` | Multiplies the graph by a factor. | none |
| `ZeroTerm(feature)` | Sets every score of the graph to 0. The term stays in the model. | Delete on all bins |
| `RemoveTerms(terms)`, `RemoveFeature(feature)` | Remove terms from the model. | none |
| `Recenter(feature)` | Moves the weighted mean of the graph into the intercept. | none |
| `SplitBin(feature, at)` | Splits the bin that holds a value into two bins with the same score. | none |
| `Rebin(feature, selection, bin_count)` | Replaces the selected bins by bins of equal width. | Equal bins, with Interpolate or Regression |
| `ExtendRange(feature, low, high)` | Adds an empty bin below the lowest or above the highest value that the model saw. | none |

```python
from ebaml import Align, Clip, EveryBin, FitLine, Interpolate, LimitSteps, MissingBin, SetScores, Shift, ZeroTerm

# "a missing value scores 0"
session.apply(SetScores("age", MissingBin(), 0.0), "A passenger with no age scores like the average")

# "move a range of bins"
session.apply(Shift("fare", Bins(0, 2), 0.1), "The cheapest tickets were under-estimated")

# "smooth a jump"
session.apply(Interpolate("fare", Bins(3, 6)), "The jump of the graph has no cause in the data")

# "flatten a dip"
session.apply(Align("fare", Bins(3, 5), "left"), "The dip has too few passengers to trust")

# "replace a range by a trend"
session.apply(FitLine("fare", Bins(7, 9)), "Fit the trend of the expensive tickets")

# "keep each score in a range", the missing bin and the unseen bin included
session.apply(Clip("fare", EveryBin(), -1.0, 1.0), "No fare may move the log-odds by more than 1")

# "no sudden jump"
session.apply(LimitSteps("fare", 0.3), "A small change of the fare must not change the score much")
```

A direct edit never re-centres the graph, so a person who sets a score to 0.38 reads 0.38. `Monotonize` keeps the
weighted mean of the selected scores, so it needs no re-centring either. `Recenter` moves the weighted mean of a graph
into the intercept and changes no prediction. An edit of `Monotonize` on the whole graph gives the same scores as
`monotonize` of `interpret`.

"This feature must not be used" has three edits. `ZeroTerm` sets the scores to 0 and keeps the term. `RemoveTerms`
removes named terms. `RemoveFeature` removes every term that uses the feature, the pair terms included. The model
keeps its input columns, so the pipeline around it does not change.

```python
session.apply(ZeroTerm("siblings"), "The business must not score on the number of siblings")
print(session.model.predict_proba(X)[:2])  # the edited model works as a fitted model works
```

## Change the bins

The edits above keep the bins of the model. Three edits change the bins of a continuous feature. Each one changes the
cuts, the scores, the bags, the standard deviations and the training weights together. If a pair term reads the same
cuts as the main effect, the main effect gets its own cuts first, so the pair term does not change.

- `SplitBin` cuts one bin in two. Both parts keep the old score, so the split changes no prediction. A later edit can
  then give each part its own score.
- `Rebin` replaces a range of bins by bins of equal width. The score of a new bin is the weighted mean of the old
  scores that it covers. Follow it with `Interpolate` or `FitLine` to do what the equal-bins tools of GAM Changer do.
- `ExtendRange` adds a bin beyond the lowest or the highest value that the model saw. The new bin has the score of the
  outer bin and no training weight, so the edit changes no prediction. Set the score of the new bin with a later edit.

The model keeps the training weight of each bin, not the values of the rows. So an edit that cuts a bin shares the
weight of the bin by width, as if the values were spread evenly over the bin. This weight is an estimate.

```python
from ebaml import ExtendRange, Rebin, SplitBin

session = EditSession(ebm)
cuts = session.model.bins_[1][0]  # the cuts between the bins of fare
session.apply(SplitBin("fare", (cuts[-2] + cuts[-1]) / 2), "The most expensive tickets have their own price class")
session.apply(Rebin("fare", Bins(0, 5), 3), "Smooth the cheapest tickets")
session.apply(Interpolate("fare", Bins(0, 3)), "Draw a straight line over the cheap tickets")
high = session.model.feature_bounds_[1][1]
session.apply(ExtendRange("fare", high=high + 100.0), "A fare above every ticket in the data must score too")
last_bin = len(session.model.bins_[1][0])  # the new bin is the last one
session.apply(SetScores("fare", Bins(last_bin, last_bin), 0.5), "The highest fares score like first class")
```

## Check business constraints

A constraint states a business requirement on a graph, for example "the risk must not fall when the fare goes up".
`check_constraints` reads a model and changes nothing. It returns one `Violation` for each constraint that the model
breaks. A violation names the bins that break the constraint, the largest excess, and the edit that repairs it.

| Constraint | Requirement | Repair |
| --- | --- | --- |
| `Monotone(feature, direction, selection)` | The selected scores do not fall (`"increasing"`) or do not rise (`"decreasing"`) from bin to bin. | `Monotonize` |
| `MaxStep(feature, step, selection)` | No step between the scores of two neighbour bins is larger than `step`. | `LimitSteps` |
| `Bounded(feature, low, high, selection)` | Each selected score is between `low` and `high`. The default selection is `EveryBin()`. | `Clip` |
| `FixedScore(feature, selection, value)` | Each selected score equals `value`. | `SetScores` |
| `Unused(feature)` | No term uses the feature with a score other than 0, the pair terms included. | `RemoveFeature` |

A constraint other than `Unused` reads the main effect of the feature only. A pair term that uses the feature can still
change the effect of the feature on a prediction. A difference at or below `tolerance` (default `1e-9`) is not a
violation.

A repair changes the predictions of the model. Preview each repair before you apply it. A repair of one constraint can
break another constraint, so check all the constraints again after the repairs.

```python
from ebaml import Bounded, FixedScore, MaxStep, Monotone, Unused, check_constraints

constraints = [
    Monotone("fare", "increasing", Bins(0, 5)),
    MaxStep("fare", 0.5),
    Bounded("age", -1.0, 1.0),
    FixedScore("age", MissingBin(), 0.0),
    Unused("siblings"),
]
for violation in check_constraints(session.model, constraints):
    print(violation.description, violation.bins)
    session.apply(violation.repair, f"Meet the business constraint: {violation.description}")
remaining = check_constraints(session.model, constraints)
```

## Check the effect on data

A global metric hardly moves when a few bins change, so the harm of an edit shows only in the rows that it reaches or
in a subgroup. `edit_effect` gives the numbers of the Metric Panel of GAM Changer. It compares three models: the
original model, the model before the last edit, and the current model. It gives each metric in three scopes: all
rows, the rows that the last edit changes, and a subgroup that you choose.

| Model | Metrics |
| --- | --- |
| Binary classifier | Accuracy, balanced accuracy, ROC AUC and the confusion matrix. |
| Regressor | RMSE, MAE and MAPE (a share: 0.1 means 10 %). |

Without labels, the result counts the rows whose prediction changes, and for a classifier the rows whose class changes
in each direction. A row with a missing value stays in every count, because an edit of the missing bin reaches only
such rows.

`affected_profile` gives the numbers of the Feature Panel. For each other feature, it compares the bins of the rows
that the last edit changes with the bins of all rows, by the l1 distance of their shares. A large distance means that
the edit reaches one subgroup more than the others.

```python
import narwhals as nw

from ebaml import Subgroup, affected_profile, edit_effect

session = EditSession(ebm)
session.apply(Shift("age", Bins(0, 2), 0.3), "Young passengers were under-estimated")
rows = nw.from_native(X)
effect = edit_effect(session, rows, y, Subgroup("siblings", 0))
for scope in effect.scopes:
    print(scope.scope, scope.row_count, scope.original.accuracy, scope.current.accuracy)
profile = affected_profile(session, rows)
print(profile.features[0].feature, profile.features[0].distance)  # the feature whose mix moves most
```

## Uncertainty of an edited bin

A score that a person sets has no fitted uncertainty. So the standard deviation of each edited bin is 0, and
`predict_with_uncertainty` of the edited model gives the edited prediction. A local explanation of an edited model reads
the confidence intervals of the model, so its confidence bounds do not include the uncertainty of the edited bins.
A model that `interpret` monotonized has no standard deviations. A session fills them with 0 in its copy.

## History, undo and save

Every edit is a revision with an id. The history keeps the edit, a generated description, your reason, the time, and
whether a person reviewed the edit.

```python
session.undo()  # back to the previous revision
session.redo()  # forward again
session.review()  # a person has read the edit at the head

for row in session.history_table():
    print(row.short_id, row.kind, row.edit_type, row.reason, row.reviewed)

edited = session.model_at("original")  # read any revision, with no checkout
```

A revision is `"original"`, `"current"`, or the id of an edit. `model_at` reads a copy of any revision, so the head
does not move. A new `apply` after an `undo` drops the redo stack, as GAM Changer does.

A save is a commit. It needs a message: the description of the change that the saved model holds, for example the
business requirement that the edits serve.

```python
import tempfile
from pathlib import Path

for entry in session.history:
    session.review(entry.id)  # a person has read each edit
with tempfile.TemporaryDirectory() as directory:
    entry = session.save(Path(directory) / "model.skops", "Meet the pricing rules of 2026", author="Ada")
    print(sorted(path.name for path in Path(directory).iterdir()))
```

The save writes the model with `skops.io`, and the history next to it, in a file with the suffix `.history.json`. A save
never overwrites a file, and it refuses edits that nobody reviewed unless you pass `allow_unreviewed=True`. A `save`
entry holds the ids of the edits since the previous save, the revision, the name of the file, and a hash of the saved
model. Undo, redo and checkout skip the `save` entries.

`session.to_json()` writes the history as JSON. `EditSession.replay(ebm, history)` builds the same session again from
the original model, and it refuses a model whose features, bins or scores differ, because bins that moved make a
selection mean other rows.

## Several threads

Several threads can share one session, for example the tool calls of an agent. A revision never changes, so a reader
needs no lock, and a slow read does not block the writers. A call that changes the head can take `expected_revision`.
If the head differs, the call raises a `StaleRevisionError` and changes nothing. Read the session again, and decide if
the call still fits. The session never retries for you.

## Edit through an agent

The [MCP server](mcp.md#edit-the-model) gives an agent the same edits as tools. The agent applies each edit with a
reason, and the server asks a person to confirm the edits before it saves the model.

## Coming from GAM Changer

| GAM Changer | EBaMl |
| --- | --- |
| `move` | `Shift` |
| `increasing`, `decreasing` | `Monotonize` |
| `inplace-interpolate` | `Interpolate` |
| `inplace-regression` | `FitLine` |
| `align` | `Align`, or `SetScores` for a typed value |
| `delete` | `SetScores` with value 0, or `ZeroTerm` |
| `equal-interpolate`, `equal-regression` | `Rebin`, then `Interpolate` or `FitLine` on the new bins |
| `.gamchanger` files | `load_gamchanger` and `apply_gamchanger`. |

EBaMl differs from GAM Changer in these points:

- EBaMl edits the missing bin and the unseen bin. GAM Changer drops them.
- EBaMl keeps the bagged scores and the standard deviations in step with the scores, so `predict_with_uncertainty`
  and the graph text work on the edited model. GAM Changer writes the scores only.
- EBaMl does not round the scores. GAM Changer rounds every score to 4 decimals, also the bins that you did not edit.
- EBaMl supports binary classifiers and regressors. A multiclass classifier raises `UnsupportedModelError`.
- A feature with a pair term has its main effect edited only. The report names the pair terms.

### Load a `.gamchanger` file

`load_gamchanger` reads the file that GAM Changer saves, and builds the edits of the newest commit of each feature, as
`get_edited_model` of GAM Changer does. It changes no model. `apply_gamchanger` applies the edits to a session. The
reason of each edit names the GAM Changer commit and holds its description. An edit of a commit that a person
confirmed in GAM Changer is marked as reviewed.

<!-- invisible-code-block: python
import orjson
from pathlib import Path
from tempfile import mkdtemp

session = EditSession(ebm)
model_data = {"features": [], "labelEncoder": {}}
for name, feature_type in zip(ebm.feature_names_in_, ebm.feature_types_in_):
    term = list(ebm.term_features_).index((list(ebm.feature_names_in_).index(name),))
    index = list(ebm.feature_names_in_).index(name)
    scores = [round(float(score), 4) for score in ebm.term_scores_[term][1:-1]]
    edges = [float(ebm.feature_bounds_[index][0]), *map(float, ebm.bins_[index][0]), float(ebm.feature_bounds_[index][1])]
    model_data["features"].append({"name": name, "type": "continuous", "additive": scores, "binEdge": edges})
age = model_data["features"][0]
places, scores = age["binEdge"][:-1], [0.25, *age["additive"][1:]]
points = {
    str(point): {"x": place, "y": score, "rightPointID": point + 1 if point + 1 < len(places) else None}
    for point, (place, score) in enumerate(zip(places, scores))
}
commit = {
    "featureName": "age", "type": "align", "description": "Set 1 bins to 0.25.", "time": 1794676628395,
    "hash": "2677726ef90cba7e", "reviewed": True, "state": {"pointData": points},
}
document = {"modelData": model_data, "sampleData": None, "historyList": [commit]}
path = Path(mkdtemp()) / "edit-11-14-2026.gamchanger"
path.write_bytes(orjson.dumps(document))
-->

```python
from ebaml import apply_gamchanger, load_gamchanger

commits = load_gamchanger(path, ebm)  # the model that you edited in GAM Changer, before the edits
for commit in commits:
    print(commit.feature, commit.commit_type, commit.description)  # age align Set 1 bins to 0.25.
reports = apply_gamchanger(session, commits)
assert [row.reason for row in session.history_table()][-1] == "GAM Changer commit 2677726: Set 1 bins to 0.25."
```

The result differs from `get_edited_model` in these points:

- An edit sets only the bins whose score differs from the score of the model rounded to 4 decimals. The other bins keep
  their full precision, so they differ from `get_edited_model` by up to 0.00005.
- `load_gamchanger` refuses a file that GAM Changer made from another model: other features, bin edges or scores.
- A commit of `equal-interpolate` or `equal-regression` changes the bin edges. `load_gamchanger` loads it as a
  `Rebin` of the range that the commit replaced, followed by the scores of the commit.
- The missing bin and the unseen bin keep their scores, because GAM Changer cannot edit them.

The [notebook](editing-notebook.md) shows the edits of this page with the report after each edit.
