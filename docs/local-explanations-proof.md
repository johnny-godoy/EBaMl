# Proof of the local explanations

The counterfactual and the sufficient reason of [local explanations](local-explanations.md) are exact. This page
states what a proof shows, and where to read it.

<p><a href="../proofs/local-explanations.pdf">Read the paper (PDF)</a></p>

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

## How the proof is checked

A Lean 4 library checks each statement with the Mathlib library. The paper names the Lean theorem of each statement.
Its appendix gives the commands to repeat the check. The check runs by hand, and it is not part of CI.

- [Lean files](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/EBaMlProofs)
- [Typst source of the paper](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/papers/local-explanations)

<iframe
  src="../proofs/local-explanations.pdf"
  title="The proof of the local explanations, as a PDF"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lightest, #ccc);"
></iframe>
