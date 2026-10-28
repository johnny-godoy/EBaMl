# Proof of the local explanations

The counterfactual and the sufficient reason of [local explanations](local-explanations.md) are exact. This page
states what a proof shows, and where to read it.

<p><a href="proofs/local-explanations.pdf">Read the paper (PDF)</a></p>

## What is proved

The proof covers a binary EBM without interaction terms. The score of such a model is a sum of one term per feature.

- **Minimum change required.** `_counterfactual` returns a set of features whose change flips the prediction. No set
  with fewer features has a change that flips it. If the function returns `None`, no change flips the prediction.
- **Minimum sufficient reason.** `_sufficient_reason` returns a set of features that is sufficient. No set with fewer
  features is sufficient. If the function returns `None`, no set of features is sufficient.

Both claims come from one abstract result. Sort numbers from the largest to the smallest, and add them until the sum
passes a threshold. The number of terms that this takes is the smallest number of terms of any set that passes the
threshold. For the counterfactual, the numbers are the *drops* of the features. For the sufficient reason, they are
the *gains*.

## What is not proved

- Floating point. The proof works over the real numbers.
- Ties between bins. The proof allows any bin at the minimum, and the code takes the first.
- The completions. They are sampled, so there is nothing exact to prove.
- The confidence level of the counterfactual. Only the features that the counterfactual changes move by `z` standard
  deviations. The other features keep their mean scores, and the proof covers this behavior.

## Definitions for every model kind

`explain_instance` accepts binary classifiers only. The definitions below set what the explanations of the other
kinds will mean. The result types of `ebaml.local` can hold all of them. Each definition says whether sorting solves it
exactly. Only the binary case has a proof now.

### Notation

The score of an EBM without interactions is `s(x) = b + Σ f_i(x_i)`. The intercept is `b`, and `f_i` is the graph of
feature `i`. A multiclass classifier has one score `s_k(x)` for each class `k`. A *decision* is what the explanation
keeps or changes. A *margin* is a number that is positive when the decision holds. Each decision below has one margin or
more, and each margin is a sum of one term for each feature.

| Kind | Decision | Margins | Output |
| --- | --- | --- | --- |
| Binary classifier | The predicted class. | `±s(x)`, with the sign of the predicted class. | The probability of the predicted class. |
| Regressor, threshold `t` | The prediction is above `t`, or not above `t`. | `±(s(x) - t)`. | The predicted value. |
| Regressor, interval `[lo, hi]` | The prediction is below, inside or above the interval. | Below: `lo - s(x)`. Inside: `s(x) - lo` and `hi - s(x)`. Above: `s(x) - hi`. | The predicted value. |
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
  not solve it in general. A strategy object selects the search:

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
standard deviations, which is never smaller than the true value.

### Result types

| Field | Binary classifier | Regressor | Multiclass classifier |
| --- | --- | --- | --- |
| `Prediction.decision` | The predicted class. | The decision, for example "above `t`". | The predicted class. |
| `Prediction.output`, `lower`, `upper` | The probability of the class, and its interval. | The predicted value, and its interval. | The probability of the class, and its interval. |
| `Prediction.score`, `score_std` | The log-odds of the second class. | The predicted value. | The smallest margin. |
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
