---
title: Proof of the recourse plans
description: What the Lean proof shows about the cheapest recourse plans of a local explanation.
---

# Proof of the recourse plans

A recourse plan is a counterfactual that a person can carry out. Each change has a cost, and the search looks for the
cheapest plan under the limits of the person. This page states what a proof shows about these plans, and where to
read it. The limits themselves (immutable features and allowed values) are in [Recourse
limits](local-explanations.md#recourse-limits).

<p><a href="proofs/recourse.pdf">Read the paper (PDF)</a></p>

## What is proved

The proof covers a binary EBM without interaction terms, and a decision with one margin. The abstract results hold for
any number of margins.

- **The integer program.** The integer program has one 0/1 variable for each bin that a feature may move to, at most
  one bin for each feature, the margin constraints, a cap on the number of changes and the excluded bins. Its feasible
  points are exactly the valid plans, and its objective is the cost of the plan. So its optimum is a cheapest plan.
- **Unit costs.** When every change costs 1, the sorting search returns a cheapest plan. This is the fewest changes.
- **Other costs.** With other costs, the bin with the largest drop is not always part of a cheapest plan, so sorting
  does not apply. The problem with one margin is NP-hard: every instance of the knapsack problem is the counterfactual
  of a binary EBM.
- **Stricter settings.** A higher confidence level, a smaller allowed set, a new immutable feature, a lower cap and
  more excluded bins never give a cheaper plan, and never make an impossible plan possible.
- **Diverse plans.** Each plan after the first uses no pair (feature, bin) of an earlier plan, as in the code of GAM
  Coach. Each plan is valid, and the costs of the plans do not go down.

## What is not proved

- Floating point and the tolerances of the MIP solver. A proved optimum rests on the certificate of the solver.
- The rule for ties between plans of the same cost. The proof allows any cheapest plan.
- The link to the code. A unit test compares the cost of each plan with an exhaustive oracle on small models. The code
  has no diverse plans.
- The counterfactual of a chosen class of a multiclass classifier, and a lower bound from an LP relaxation.
- The running time of the reduction from the knapsack problem. The NP-completeness of the knapsack problem is a
  citation of Karp.

## How the proof is checked

A Lean 4 library checks each statement with the Mathlib library. The paper names the Lean theorem of each statement.
Its appendix gives the commands to repeat the check. A CI job repeats it when the Lean files change.

- [Lean files](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/EBaMlProofs/Recourse)
- [Typst source of the paper](https://github.com/johnny-godoy/EBaMl/tree/main/proofs/papers/recourse)

<p><a href="proofs/recourse.pdf">Open the paper (PDF) in its own page</a></p>

<iframe
  src="proofs/recourse.pdf"
  title="The proof of the recourse plans, as a PDF"
  loading="lazy"
  style="width: 100%; height: 80vh; border: 1px solid var(--md-default-fg-color--lightest, #ccc);"
></iframe>
