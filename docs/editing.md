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
| `Scale(feature, factor)` | Multiplies the graph by a factor. | none |
| `ZeroTerm(feature)` | Sets every score of the graph to 0. The term stays in the model. | Delete on all bins |
| `RemoveTerms(terms)`, `RemoveFeature(feature)` | Remove terms from the model. | none |
| `Recenter(feature)` | Moves the weighted mean of the graph into the intercept. | none |

```python
from ebaml import Align, FitLine, Interpolate, MissingBin, SetScores, Shift, ZeroTerm

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
| `equal-interpolate`, `equal-regression` | Not supported yet: they change the bin edges. |
| `.gamchanger` files | Not supported yet. |

EBaMl differs from GAM Changer in these points:

- EBaMl edits the missing bin and the unseen bin. GAM Changer drops them.
- EBaMl keeps the bagged scores and the standard deviations in step with the scores, so `predict_with_uncertainty`
  and the graph text work on the edited model. GAM Changer writes the scores only.
- EBaMl does not round the scores. GAM Changer rounds every score to 4 decimals, also the bins that you did not edit.
- EBaMl supports binary classifiers and regressors. A multiclass classifier raises `UnsupportedModelError`.
- A feature with a pair term has its main effect edited only. The report names the pair terms.

The [notebook](editing-notebook.md) shows the edits of this page with the report after each edit.
