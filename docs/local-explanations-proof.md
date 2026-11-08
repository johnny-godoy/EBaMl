# Proof of the local explanations

The counterfactual and the sufficient reason of [local explanations](local-explanations.md) are exact. This page
states what a proof shows, and where to read it.

<p><a href="proofs/local-explanations.pdf">Read the paper (PDF)</a></p>

## What is proved

The proof covers a binary EBM, a multiclass EBM and a regressor without interaction terms. The score of such a model is a sum of one
term per feature.

- **Minimum change required.** `_counterfactual` returns a set of features whose change flips the prediction. No set
  with fewer features has a change that flips it. If the function returns `None`, no change flips the prediction.
- **Minimum sufficient reason.** `_sufficient_reason` returns a set of features that is sufficient. No set with fewer
  features is sufficient. If the function returns `None`, no set of features is sufficient.

- **Regressors.** The decision of a regressor ("above `t`" or "at most `t`") has the margin of the binary model with the
  intercept moved by `t`. The code does this shift, so the two claims above hold for it. A change that the code finds
  moves the predicted value across `t`.
- **Inside an interval.** When the predicted value is inside `Interval(low, high)`, the decision has two margins. The
  counterfactual has as many features as the smaller of the two answers for the margins.
- **Several margins.** `Minimal` returns a sufficient set from which no feature can be removed. `ExactOrMinimal` returns
  a smallest sufficient set when it runs, and its search visits at most 2 to the power of the number of features. A
  result of `Minimal` with at most one feature is also a smallest set. `Minimal` makes no other claim about the size.

- **Multiclass classifiers.** The predicted class `c` has one margin for each rival class `j`: the score of `c` minus the
  score of `j`. Each margin is the binary model with the intercept `b_c - b_j`, the scores `s_c - s_j` and the
  deviations `σ_c + σ_j`. The counterfactual "the predicted class is no longer `c`" has as many features as the
  smallest of the answers for the rivals, and `_smallest_counterfactual` returns this number. A set keeps the class
  iff it is sufficient for every margin, so the claims of `Minimal` and `ExactOrMinimal` above hold for the sufficient
  reason. The sum of the two deviations is an upper bound of the deviation of the margin for any correlation.

- **Hardness.** The sufficient reason of several margins contains Set Cover. A set of features is sufficient for the
  reduction iff the matching sets cover every element, and a multiclass classifier with one rival class for each
  element has this problem as its sufficient reason. The decision problem is NP-complete, so a polynomial algorithm for
  the smallest sufficient set exists only if P = NP. The cost of `ExactOrMinimal` is therefore expected.
- **The same approximability as Set Cover.** The sufficient sets of the reduction are exactly the covers. So the two
  problems have the same smallest size and the same approximate solutions for every ratio. Lean checks this. The
  hardness result of Dinur and Steurer for Set Cover then holds for the sufficient reason: for every `α > 0`, it is
  NP-hard to find a sufficient set that is at most `(1 - α) ln m` times the smallest, with `m` margins.

Both claims of the first two items come from one abstract result. Sort numbers from the largest to the smallest, and add them until the sum
passes a threshold. The number of terms that this takes is the smallest number of terms of any set that passes the
threshold. For the counterfactual, the numbers are the *drops* of the features. For the sufficient reason, they are
the *gains*.

## What is not proved

- Floating point. The proof works over the real numbers.
- Ties between bins. The proof allows any bin at the minimum, and the code takes the first.
- The completions. They are sampled, so there is nothing exact to prove.
- A change that lands exactly on the threshold. The robust margin is then zero, and the code does not count it as a
  flip. A classifier has the same convention at the log-odds zero.
- The order of `Minimal`. The proof allows any order, and it proves only that no feature can be removed.
- A tie between the two highest classes of a multiclass classifier. The code takes the first class, and the margin is
  then zero. The proof needs a strict winner.
- The counterfactual with a chosen target class `j`. The proof covers "the predicted class is no longer `c`".
- The running time of the reduction, and the membership of the decision problem in NP. Lean checks that the reduction
  is correct. The paper argues the running time and the membership in a few lines, and it cites Karp for the
  NP-hardness of Set Cover.
- The hardness result of Dinur and Steurer and the algorithms of Chvátal and of Kolliopoulos and Young. The paper
  cites them as stated. It proves none of them, and the code implements none of the two algorithms.
- A fixed number of margins that is larger than one. The reduction needs one margin for each element, so a classifier
  needs one rival class for each element.
- The confidence level of the counterfactual. Only the features that the counterfactual changes move by `z` standard
  deviations. The other features keep their mean scores, and the proof covers this behavior.

## Definitions for every model kind

`explain_instance` accepts binary classifiers, multiclass classifiers and regressors. The definitions below set what the
explanations of each kind mean. The result types of `ebaml.local` hold all of them, and each kind has a proof. A
regressor uses the same sorting algorithms on a shifted intercept for each margin, and the answer "inside" uses the
search strategies below. A multiclass classifier uses them on the margin of each rival class. The counterfactual with a
chosen target class `j` has no proof.

### Notation

The score of an EBM without interactions is `s(x) = b + Σ f_i(x_i)`. The intercept is `b`, and `f_i` is the graph of
feature `i`. A multiclass classifier has one score `s_k(x)` for each class `k`. A *decision* is what the explanation
keeps or changes. A *margin* is a number that is positive when the decision holds. Each decision below has one margin or
more, and each margin is a sum of one term for each feature.

| Kind | Decision | Margins | Output |
| --- | --- | --- | --- |
| Binary classifier | The predicted class. | `±s(x)`, with the sign of the predicted class. | The probability of the predicted class. |
| Regressor, threshold `t` | The prediction is above `t`, or not above `t`. | `±(s(x) - t)`. | The predicted value. |
| Regressor, interval `(lo, hi]` | The prediction is at most `lo`, inside the interval, or above `hi`. | At most `lo`: `lo - s(x)`. Inside: `s(x) - lo` and `hi - s(x)`. Above: `s(x) - hi`. | The predicted value. |
| Multiclass classifier | The predicted class `c`. | `s_c(x) - s_j(x)` for each other class `j`. | The probability of `c`. |

### Counterfactual

The counterfactual is the smallest set of changes after which the decision does not hold.

- **One margin.** This is the binary problem on that margin, so sorting solves it exactly. A regressor with a
  threshold is the binary problem with the intercept `b - t`, and so is an interval decision of "below" or "above".
- **Several margins, the decision "inside" or "class `c`".** The decision fails when one margin becomes negative.
  The answer is the smallest of the answers of the binary problems on each margin, so sorting solves it exactly. This
  is the default counterfactual of a multiclass classifier: "the predicted class is no longer `c`".
- **A chosen target class `j`.** An option of the multiclass counterfactual asks for the smallest set of changes after
  which the predicted class is `j`. All the margins of `j` must become positive together, so sorting does not solve it.
  It uses a search strategy of the sufficient reason below, and it is not proved.

### Sufficient reason

The sufficient reason is a set of features. When these features keep their values, the decision holds for every value
of the other features.

- **One margin.** This is the binary problem, so sorting finds the smallest set exactly.
- **Several margins.** The decision holds for every completion when the worst case of each margin is positive. Each
  margin gives one linear condition, so the smallest set that meets all of them is a covering problem. Sorting does
  not solve it in general, and the problem is NP-hard. A strategy object selects the search:

  | Strategy | Result | Claim |
  | --- | --- | --- |
  | Minimal (default) | Sort by the smallest gain over the margins, add features until every condition holds, then remove each feature that is not necessary. | *Minimal*: no feature of the set can be removed. Not *minimum*. |
  | Exact search with a fallback | Search the sets by size, up to a limit on the number of features. Above the limit, use the minimal strategy. | *Minimum* when the search runs. Otherwise *minimal*. |

  `SufficientReason.minimum` tells which claim the result has.

### Count completions

The share of completions that keep the decision, estimated by sampling. This works for every kind, because it needs
only a test of the decision. It is an estimate, so it makes no exact claim.

### Confidence

Each margin moves by `z` standard deviations against the claim, as for the binary case. EBMs store no covariance
between the scores of two classes. The standard deviation of a margin `s_c - s_j` therefore uses the sum of the two
standard deviations. For any correlation, the sum is an upper bound of the true value. The proofs of 1.2.0 must check
this bound.

### Result types

| Field | Binary classifier | Regressor | Multiclass classifier |
| --- | --- | --- | --- |
| `Prediction.decision` | The predicted class. | The decision, for example "above `t`". | The predicted class. |
| `Prediction.output`, `lower`, `upper` | The probability of the class, and its interval. | The predicted value, and its interval. | The probability of the class, and its interval. |
| `Prediction.score`, `score_std` | The log-odds of the second class. | The predicted value. | The score of the predicted class. |
| `Prediction.threshold` | `0`, the log-odds zero. | The threshold of the binding margin. | The highest score of the other classes. |
| `Prediction.margin` | The log-odds of the predicted class. | The margin of the decision. | The smallest margin. |
| `Counterfactual.new_decision` | The other class. | The other decision. | The new predicted class. |
| `Counterfactual.new_output` | The probability of the new class. | The new predicted value. | The probability of the new class. |
| `SufficientReason.guaranteed_margin` | The worst case of the margin. | The worst case of the margin. | The smallest worst case over the margins. |
| `SufficientReason.minimum` | Always true (proved). | True for one margin. | See the strategy. |

## How the proof is checked

A Lean 4 library checks each statement with the Mathlib library. The paper names the Lean theorem of each statement.
Its appendix gives the commands to repeat the check. A CI job repeats it when the Lean files change.

- [Lean files](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/EBaMlProofs)
- [Typst source of the paper](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/papers/local-explanations)

<iframe
  src="proofs/local-explanations.pdf"
  title="The proof of the local explanations, as a PDF"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lightest, #ccc);"
></iframe>
