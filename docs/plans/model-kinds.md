# Plan: model kinds (0.2.0)

You are a senior Python engineer with a strong background in generalized additive models. You will implement one
milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because
the project's hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
Make every feature of EBaMl ready for the three kinds of EBM: binary classifiers, multiclass classifiers and
regressors. This milestone has two parts, and both must land before the API freezes at 1.0.0:

1. Graph text, `describe_graph` and `describe_ebm` work for multiclass classifiers. They already work for binary
   classifiers and regressors.
2. The public result types of the local explanations (`LocalFacts`, `Prediction`, `Counterfactual`,
   `SufficientReason`, `LocalExplanation`) can hold the facts of every kind, although `explain_instance` still accepts
   binary classifiers only. The local explanations of regressors and multiclass classifiers come after 1.0.0
   (`local-explanations-all-kinds.md`). Those releases can then add behavior without a breaking change.
</goal>

<context>
- The README table "Supported models" (between the `supported-models` markers) shows the state of each feature. Keep it
  true in the same PR.
- `interpret` 0.7.8 stores the scores by kind as follows (checked on fitted models):

  | Kind | `link_` | `term_scores_[i]` of a main effect | `intercept_` |
  | --- | --- | --- | --- |
  | Binary classifier | `logit` | `(n_bins + 2,)` | scalar |
  | Regressor | `identity` | `(n_bins + 2,)` | scalar |
  | Multiclass classifier | `mlogit` | `(n_bins + 2, n_classes)` | `(n_classes,)` |
  | Interaction term | as the model | `(n_bins_a, n_bins_b)` (with an extra class axis for multiclass) | |

  `standard_deviations_` has the same shape as `term_scores_`. The first and the last bin hold the missing and the
  unknown values.
- `EBMGraph.from_ebm` (`src/ebaml/graphs/ebm_graph.py`) reads `term_scores_[feature_index][1:-1]` and assumes one score
  per bin. `src/ebaml/local/facts.py` rejects every model that is not a binary classifier without interactions, with
  `UnsupportedModelError`.
- The local explanations are proved exact in Lean (`proofs/`, `.claude/rules/proofs.md`). The definitions that you
  write in part 2 are the base of the next proofs.
</context>

<design>
<model_kind>
Add one abstraction for the kind of a model, in the lowest layer that needs it (`ebaml.core` if the graphs need it,
else `ebaml.graphs`). Use one class per kind (`BinaryClassifier`, `MulticlassClassifier`, `Regressor`) behind a small
`Protocol` or abstract base, and one function that reads the kind from a fitted EBM (`model_kind(ebm)`), so no code
tests `link_` or `classes_` in an `if` chain (SOLID O, `.claude/rules/solid.md`). Each class answers only what its
callers need, for example:

- the names of the outputs (one output for binary and regression, one per class for multiclass),
- what a score means, as text for the prompts ("log-odds of class X", "change of the predicted value, in the units of
  the target"),
- how scores become a prediction (sigmoid, softmax, identity).

Raise `UnsupportedModelError` from `model_kind` for an EBM that it cannot read.
</model_kind>

<multiclass_graphs>
An EBM graph of a multiclass classifier has one score per bin and per class.

- Let `EBMGraph` hold the scores of every output: shape `(n_bins,)` for one output, `(n_bins, n_classes)` for several,
  with the class names. Keep the one-output case unchanged, so the text of binary and regression graphs does not move.
- Softmax does not change when a constant is added to the scores of all classes in a bin, so only the differences
  between the classes carry meaning. Choose and document one representation, for example the scores centred over the
  classes in each bin, and explain it in the prompt text.
- Write the text of a multiclass graph with one series per class, inside the same token budget. Extend
  `EBMGraph.to_text`, `EBMGraph.from_text` and `parse_graph_text` together, and keep the round-trip property tests.
- Add an option to describe a subset of the classes (for example `GraphOptions.classes`), because the text of a model
  with many classes is long.
- Change the prompts in `baml_src/talk_to_ebm.baml` only as much as multiclass needs. Follow the prompt rules of
  `.claude/rules/writing-style.md`. Run the offline BAML tests, and compare old and new prompts on the benchmark.
</multiclass_graphs>

<local_types>
Write the definitions first, then the types. In a new section of `docs/local-explanations-proof.md` (or the paper in
`proofs/papers/local-explanations/`), define the three explanations for each kind. Start from these proposals, and
check each one with the user before you build types on it:

- **Regressor.** The user gives a threshold `t` (or an interval). The decision is "the prediction is above `t`". With
  the identity link, the margin is `prediction - t`, which is a sum of one score per feature, so the counterfactual and
  the sufficient reason are the binary problems with a shifted intercept. The binary proofs should carry over.
- **Multiclass, counterfactual.** "The fewest changes so that the predicted class is no longer `c`." That is the minimum
  over the other classes `j` of the binary problem on the margin `s_c - s_j`, so it is exact by sorting.
  "The fewest changes so that the class becomes `j`" needs all margins at the same time, and sorting does not solve it
  exactly. Prefer the first definition, or say that the second is not proved.
- **Multiclass, sufficient reason.** The prediction stays `c` for every completion when the worst-case margin against
  every other class is positive. Each class gives one linear condition, so the smallest feature set that meets all of
  them is a covering problem, and sorting does not find the minimum in general. Choose between an exact search (the
  number of features is small) and a weaker claim ("minimal": no feature can be removed), and use the words of
  `.claude/rules/proofs.md` correctly.
- **Count completions.** Sampling works for every kind. It needs only "the decision stays the same".

Then make the public types general. For example, `Prediction` holds the decision, the value on the scale of the
link, its standard deviation, and the model output (a probability or a predicted value). `Counterfactual` holds the
new output instead of `new_probability`. Keep the field names that are true for every kind, and rename the rest. This
is a **BREAKING** change, which is allowed before 1.0.0. Update `fact_text.py` and the BAML prompts of
`baml_src/local_explanations.baml` to the new names, with the same text for binary classifiers.
</local_types>

<interactions>
Interaction terms are out of scope. Today the graph descriptions skip them without a word. Log a `structlog` warning
that names the skipped terms, and keep the ⚠️ in the README table. Full support needs a two-dimensional graph text and
new local proofs. Write a short note on that in the PR description, as input for a later plan.
</interactions>
</design>

<tests>
- Unit tests in the mirrored test files for each kind class, `model_kind`, the multiclass graph, the text round trip,
  the class subset option and the skipped-interaction warning. Fit small models in fixtures, as
  `tests/unit/ebaml/conftest.py` does.
- Hypothesis property tests for the invariants: the text round trip of a multiclass graph, and the centring that keeps
  the softmax output the same.
- The existing binary and regression tests pass unchanged, except for the renamed fields of the local types.
- 100% coverage, and zero surviving mutants for the changed modules.
</tests>

<documentation>
- Update the README table "Supported models": multiclass becomes ✅ for the graph text and the descriptions.
- Update `docs/usage.md` and `docs/local-explanations.md` with a multiclass example, and with the new field names.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
This milestone can take more than one PR. Each PR with a user-visible change adds a changelog fragment, and the release
workflow releases it as a PATCH (`.claude/rules/changelog-and-versioning.md`). The PR that completes the milestone also
sets `version = "0.2.0"` in `pyproject.toml`. Mark the renamed fields as **BREAKING** in the fragment, with a table from
the old name to the new name.
</release>

<done>
- `describe_graph` and `describe_ebm` give a correct description of a multiclass EBM in a live run.
- The definitions for every kind are written, and the user has agreed to them.
- `prek run --all-files`, `uv run pytest`, the offline BAML tests, mutation testing and the strict docs build pass.
</done>
