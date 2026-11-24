// Paper: cheapest recourse plans of the local explanations of EBaMl.
// Compile from proofs/ with: typst compile --root papers papers/recourse/paper.typ

#import "/style.typ": *
#show: paper.with(title: "Cheapest Recourse Plans of EBaMl")

#align(center)[
  #text(size: 17pt, weight: "bold")[Cheapest Recourse Plans of EBaMl]
  #v(0.4em)
  #text(size: 11pt)[A proof checked in Lean 4]
]

#v(1em)

*Summary.* A recourse plan is a counterfactual that a person can carry out: a set of feature changes, each with a
positive cost, after which the decision of an Explainable Boosting Machine (EBM) changes. The ideas come from GAM
Coach @wang2023. This paper fixes the model of the plans that EBaMl searches for: costs, a cap on the number of
changes, excluded alternatives and several diverse plans. It proves that an optimum of the integer program is a
cheapest plan, that the sorting search is exact for unit costs, that sorting fails for other costs, that the weighted
problem is NP-hard, and that a stricter setting never gives a cheaper plan. Lean 4 @lean4 with Mathlib
@mathlib2020 checks every statement. The appendix tells the reader how to repeat the check.

= Claims and scope <sec-claims>

The paper builds on the paper of the local explanations @ebamllocal, whose Section 10 defines the allowed bins
$A_i$ and the end bins $E_i = A_i union {c_i}$. It covers a binary EBM without interaction terms, and a decision with
one margin. The abstract results of @sec-plans hold for any number of margins.

*Claim 1 (integer program).* The integer program with one 0/1 variable for each alternative, at most one variable
per group, the margin constraints, the cap and the excluded variables has as its feasible points exactly the valid
plans, and its objective is the cost of the plan. So its optimum is a cheapest plan (Theorem 1, Corollary 1).

*Claim 2 (unit costs).* When every change costs $1$, the sorting search of `_counterfactual` returns a cheapest plan
(Theorem 4). This is why the status `SORTING_PROOF` stays for unit costs.

*Claim 3 (other costs).* With other costs, the alternative with the largest drop is not always part of a cheapest plan
(Lemma 2), and the problem with one margin is NP-hard (Theorem 5). So the code sends a weighted problem to the
MIP solver.

*Claim 4 (contract 7).* A higher confidence level, a smaller allowed set, a new immutable feature, a lower cap and
more excluded alternatives never give a cheaper plan, and never make an impossible plan possible (Theorems 2 and 6).

*Claim 5 (diverse plans).* Plan $k + 1$ is a cheapest plan that uses no pair (feature, bin) of an earlier plan. Each
plan is valid, and the costs of the plans do not go down (Theorem 3).

*Claim 6 (a chosen class).* For a multiclass classifier, the counterfactual of a chosen class `target_class` is a
problem of @sec-plans with one margin for each rival class. Its valid plans are exactly the changes after which the
chosen class beats every rival under the robust scores, and its smallest plan is the fewest such changes (Lemma 3,
Theorem 7). So Claims 1, 4 and 5 hold for it.

= The model <sec-model>

A _problem_ has a finite set $I$ of groups (the features), a finite set $Alpha_i$ of alternatives for each group (the
end bins of feature $i$ other than $c_i$), a set $K$ of margins, the worst case $W_k$ of each margin, the gain
$g_(i,a,k)$ of each alternative for each margin, and the cost $w_(i,a) > 0$ of each alternative. A _plan_ $p$ picks
at most one alternative $p_i$ in each group. Its _margin_ is $W_k + sum_i g_(i,p_i,k)$, where a group with no pick
adds $0$, and its _cost_ is $sum_i w_(i,p_i)$. _Limits_ are a cap $n$ (or none) and a set of _excluded_ alternatives. A
plan is _valid_ when each margin is positive, it changes at most $n$ groups, and it picks no excluded alternative. A
plan is _cheapest_ when it is valid and no valid plan costs less.

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Symbol*], [*Meaning*], [*Lean*],
  [$I$, $Alpha_i$], [features, their alternatives], [`ι`, `α`; `Alt m A` in the application],
  [$W_k$, $g_(i,a,k)$, $w_(i,a)$], [worst case, gain, cost], [`Problem.worst`, `.gain`, `.cost`],
  [$p$], [a plan], [`Plan ι α`],
  [$n$, excluded], [cap, exclusions], [`Limits.cap`, `Limits.excluded`],
  [$x_(i,a) in {0, 1}$], [a point of the integer program], [`Problem.Point`],
)

The code builds the alternatives in `_alternatives` of `counterfactuals.py`, from `Feature.reachable`. `build_model` of
`covering_backend.py` writes the margin constraints, the constraint of one pick per group, the cap (`max_picks`) and
the objective: the sum of the costs (`CoveringModel.costs`), or the number of picks without costs. `MipCover` solves a
problem with costs over every alternative, without the reduction of `reduce_problem`, which keeps the smallest size
and not the smallest cost. A smaller allowed set and an immutable feature remove alternatives. The diverse plans
exclude the alternatives of the earlier plans, as the code of GAM Coach removes their variables.

= The abstract results <sec-plans>

#theorem("the integer program", [
  Let $x$ be a point with $x_(i,a) in {0,1}$. Call $x$ _feasible_ when each group has at most one variable equal to
  $1$, $W_k + sum_(i,a) x_(i,a) g_(i,a,k) > 0$ for each margin, $sum_(i,a) x_(i,a) <= n$, and $x_(i,a) = 0$ for each
  excluded alternative. Then the point of a plan is feasible iff the plan is valid, the plan of a feasible point is
  valid, and in both directions the objective $sum_(i,a) x_(i,a) w_(i,a)$ equals the cost of the plan.
], lean: ("proofs/EBaMlProofs/Recourse/Plans.lean", "EBaMlProofs.Recourse.Problem.ofPlan_feasible_iff", "EBaMlProofs.Recourse.Problem.objective_ofPlan", "EBaMlProofs.Recourse.Problem.toPlan_valid", "EBaMlProofs.Recourse.Problem.objective_toPlan"))

#proof[
  *ASSUME:* a problem and limits. *PROVE:* the three statements.

  #step(1, "1", [In a group, a sum over the variables of the point of a plan is the value at its pick, or $0$
    (`sum_ofPlan`). So the margins, the number of picks and the objective of the point equal those of the plan.])
  #step(1, "2", [A point with at most one $1$ in each group is the point of its plan (`ofPlan_toPlan`), because two
    picks in one group would give the group two variables equal to $1$.])
  #step(1, "3", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ and $chevron.l 1 chevron.r 2$.])
]

#corollary("an optimum is a cheapest plan", [
  If $x$ is feasible and no feasible point has a smaller objective, the plan of $x$ is a cheapest plan.
], lean: ("proofs/EBaMlProofs/Recourse/Plans.lean", "EBaMlProofs.Recourse.Problem.cheapest_of_optimum"))

#proof[
  A valid plan $q$ has a feasible point with objective equal to its cost (Theorem 1). The objective of $x$ is at most
  that number, and it equals the cost of the plan of $x$.
]

#theorem("a harder problem has no cheaper plan", [
  (a) If each valid plan of a problem $P'$ with limits $L'$ is a valid plan of $P$ with limits $L$, and the costs are
  equal, a cheapest plan of $(P, L)$ costs at most a cheapest plan of $(P', L')$. (b) Smaller gains with the same
  worst cases, (c) a lower cap and more exclusions each give such a pair.
], lean: ("proofs/EBaMlProofs/Recourse/Plans.lean", "EBaMlProofs.Recourse.Problem.cost_le_of_valid_imp", "EBaMlProofs.Recourse.Problem.valid_of_gain_le", "EBaMlProofs.Recourse.Problem.valid_of_limits_le"))

#proof[
  (a) A cheapest plan of $(P', L')$ is valid for $(P, L)$, so a cheapest plan of $(P, L)$ costs at most as much.
  (b) Each margin of a plan is at most its margin in $P$. (c) Each limit of $L'$ is at least as strict.
]

#theorem("diverse plans", [
  Let the limits after the plans $p_1, dots, p_k$ exclude each pick of these plans. Then (a) a valid plan after them
  keeps the original limits and uses no pick of an earlier plan, and (b) if $p$ is a cheapest plan after
  $p_1, dots, p_k$ and $p'$ is a cheapest plan after $p_1, dots, p_k, p$, then $p$ costs at most $p'$.
], lean: ("proofs/EBaMlProofs/Recourse/Plans.lean", "EBaMlProofs.Recourse.Problem.valid_excludeUsed_new", "EBaMlProofs.Recourse.Problem.cost_mono_excludeUsed"))

#proof[
  (a) holds by the definition of the limits. (b) The limits after $p$ exclude more, so Theorem 2 (c) and (a) apply.
]

= Application to the counterfactual <sec-application>

For a binary model, its margin $M$ and the allowed bins $A$, the problem of the counterfactual has one group for
each feature, the alternatives $E_i without {c_i}$, one margin with the worst case $-M$, and the gain
$t_i (c_i) - p_i (v)$ of the change of feature $i$ to bin $v$. The notation is that of @ebamllocal.

#lemma("a valid plan flips the prediction", [
  The margin of a plan is minus the robust margin $R(S, w)$ of its change, where $S$ is the set of changed features
  and $w$ the picked bins. With no cap and no exclusion, a plan is valid iff its change flips the prediction.
], lean: ("proofs/EBaMlProofs/Recourse/Counterfactual.lean", "EBaMlProofs.Recourse.margin_cfProblem", "EBaMlProofs.Recourse.valid_cfProblem_iff"))

#proof[
  Lemma 4 of @ebamllocal gives $R(S, w) = M - sum_(i in S) (t_i (c_i) - p_i (w_i))$, and the margin of the plan is
  $-M$ plus the same sum.
]

#theorem("unit costs", [
  Let every change cost $1$. Let $e$ sort the drops $d_i (A)$ in descending order, and let $k$ be the first position
  at which the sum of the first $k$ drops is above $M$. Then a valid plan changes exactly the first $k$ features, and
  every valid plan costs at least $k$.
], lean: ("proofs/EBaMlProofs/Recourse/Counterfactual.lean", "EBaMlProofs.Recourse.sorting_cheapest", "EBaMlProofs.Recourse.cost_unit"))

#proof[
  *ASSUME:* unit costs and the conditions of the statement. *PROVE:* the two parts.

  #step(1, "1", [With unit costs, the cost of a plan is its number of changes (`cost_unit`).])
  #step(1, "2", [Theorem 11 of @ebamllocal gives lowest bins $w$ whose change of the first $k$ features flips the
    prediction, and each of these bins is in $A_i$ and differs from $c_i$. They form a plan, valid by Lemma 1.])
  #step(1, "3", [A valid plan flips the prediction by Lemma 1 with admissible bins, so it changes at least $k$
    features by Theorem 11 (b) of @ebamllocal.])
  #step(1, "4", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ to $chevron.l 1 chevron.r 3$.])
]

#lemma("sorting fails with costs", [
  Take one feature with two alternatives: the first has the gain $10$ and the cost $5$, the second the gain $2$ and the
  cost $1$, and the worst case is $-1$. No cheapest plan picks the first alternative, although it has the largest
  gain.
], lean: ("proofs/EBaMlProofs/Recourse/Counterfactual.lean", "EBaMlProofs.Recourse.sorting_not_cheapest"))

#proof[
  The second alternative alone is a valid plan of cost $1$, and the first costs $5$.
]

The gain of an alternative is the drop of its bin, so the bin with the
largest drop, which sorting picks, is not in a cheapest plan.

#theorem("NP-hardness", [
  Let $v_i > 0$ be values and $V > 0$ a bound. The binary model whose feature $i$ has the bins `true` (score $v_i$, the
  instance) and `false` (score $0$), with no deviation and the margin $V$, has a change of a set $S$ that flips the
  prediction iff $sum_(i in S) v_i > V$. With the cost $w_i$ for the change of feature $i$, a plan of cost at most
  $W$ exists iff a set has value above $V$ and weight at most $W$.
], lean: ("proofs/EBaMlProofs/Recourse/Counterfactual.lean", "EBaMlProofs.Recourse.knapsack_canFlipOn_iff"))

#proof[
  The margin is $V$, and the drop of feature $i$ is $v_i - min(v_i, 0) = v_i$. Lemma 18 of @ebamllocal gives the
  equivalence. The decision problem of the knapsack is NP-complete @karp1972, and the model has the size of the
  instance, so the weighted counterfactual with one margin is NP-hard.
]

#theorem("confidence and allowed sets", [
  (a) A valid plan at the number of deviations $z' >= z$ is valid at $z$, with the same cost. (b) For allowed sets
  $A' subset.eq A$, a valid plan over $A'$ is a valid plan over $A$, with the same cost when the costs over $A'$ are
  those over $A$. So by Theorem 2 a higher confidence level, a smaller allowed set and a new immutable feature never
  give a cheaper plan.
], lean: ("proofs/EBaMlProofs/Recourse/Counterfactual.lean", "EBaMlProofs.Recourse.valid_of_higher_confidence", "EBaMlProofs.Recourse.valid_embed", "EBaMlProofs.Recourse.cost_embed"))

#proof[
  (a) The gain $t_i (c_i) - t_i (v) - z sigma_i (v)$ does not go up with $z$, because $sigma_i >= 0$, and Theorem 2 (b)
  applies. (b) The plan keeps its changed features and its bins, so Lemma 1 gives the same robust margin.
]

= The counterfactual of a chosen class <sec-target>

A multiclass classifier with the classes $K$ predicts the class with the highest score. The counterfactual of a chosen
class $tau$ (`_target_counterfactual`) asks for changes after which $tau$ beats every rival $j eq.not tau$. For a
change $(S, w)$, the robust margin of $tau$ against $j$ is
$ R_j (S, w) = b_tau - b_j + sum_(i in.not S) (s_(tau,i)(c_i) - s_(j,i)(c_i))
  + sum_(i in S) (s_(tau,i)(w_i) - s_(j,i)(w_i) - z (sigma_(tau,i)(w_i) + sigma_(j,i)(w_i))). $
The changed features take the score of the new bin, moved by $z$ deviations against $tau$, as `_alternatives` does
with `view.scores - z * view.stds`. A change _reaches_ $tau$ when each changed feature ends in an allowed bin or in the
bin of the instance, and $R_j (S, w) > 0$ for every rival. The problem of the code has one group for each feature, the
alternatives $E_i without {c_i}$, one margin for each rival with the worst case $R_j (emptyset, c)$, and the gain of a
bin $v$ for the rival $j$ that is the term of $v$ minus the term of $c_i$.

#lemma("a valid plan reaches the class", [
  The margin of a plan for the rival $j$ is $R_j$ of its change. With no cap and no exclusion, a plan is valid iff its
  change makes every margin of $tau$ positive.
], lean: ("proofs/EBaMlProofs/Recourse/Target.lean", "EBaMlProofs.Recourse.Multiclass.margin_targetProblem", "EBaMlProofs.Recourse.Multiclass.valid_targetProblem_iff", "EBaMlProofs.Recourse.Multiclass.reaches_of_valid"))

#proof[
  The margin of a plan is the worst case plus the gains of its picks. A pick adds the term of its bin and removes the
  term of $c_i$, so the sum is the sum of $R_j$.
]

#theorem("the plans of a chosen class", [
  (a) Every change that reaches $tau$ gives a valid plan with at most as many changes. (b) A valid plan exists iff
  some change reaches $tau$. (c) A valid plan with at most $k$ changes exists iff a change of at most $k$ features
  reaches $tau$. So the smallest plan is the fewest changes that reach $tau$, and with costs, Theorem 1 makes an
  optimum of the integer program a cheapest plan of the chosen class.
], lean: ("proofs/EBaMlProofs/Recourse/Target.lean", "EBaMlProofs.Recourse.Multiclass.plan_of_reaching", "EBaMlProofs.Recourse.Multiclass.exists_plan_iff", "EBaMlProofs.Recourse.Multiclass.smallest_plan_iff"))

#proof[
  *ASSUME:* a change $(S, w)$ that reaches $tau$. *PROVE:* (a) to (c).

  #step(1, "1", [Let the plan pick $w_i$ for each $i in S$ with $w_i eq.not c_i$. It changes at most $|S|$ features.])
  #step(1, "2", [For $i in S$ with $w_i = c_i$, the term of the change is the term of $c_i$ minus
    $z (sigma_(tau,i)(c_i) + sigma_(j,i)(c_i)) >= 0$, so the plan has each margin at least $R_j (S, w) > 0$.])
  #step(1, "3", [(a) holds by $chevron.l 1 chevron.r 1$ and $chevron.l 1 chevron.r 2$ with Lemma 3, and (b) and (c)
    follow from (a) and Lemma 3.])
]

= Limits <sec-limits>

- *Floating point and tolerances.* The proofs work over the real numbers. The MIP solver tests each margin against a
  tolerance (`MARGIN_EPSILON`), and its optimum rests on its certificate, which Lean does not check.
- *Ties.* Two plans can have the same cost. The code then prefers fewer changes and lower feature positions. The
  proofs allow any cheapest plan, and they do not model the tie rule.
- *The code.* The test `test_each_plan_matches_the_oracle` compares the cost of the code with an exhaustive oracle of
  this model on small models. It checks the link on some inputs, and it is not a proof. The diverse plans are not in
  the code.
- *Several margins.* The abstract results hold for any number of margins. @sec-application covers one margin, and
  @sec-target covers the margins of a chosen class. The hardness of the chosen class with unit costs is a covering
  problem of the local-explanations paper, which proves its reduction on paper only.
- *The lower bound.* The paper proves no bound of an LP relaxation for the weighted problem.
- *The paper of GAM Coach.* Its appendix A.8 excludes only a whole earlier solution. Its code, which this model
  follows, excludes each pick of an earlier plan, so the plans of this model can differ from those of the appendix.
- *Running time.* The reduction of Theorem 5 has the size of the instance. Lean does not check the running time, and
  the NP-completeness of the knapsack problem is a citation @karp1972.

= Computer check <sec-computer>

The Lean files are `proofs/EBaMlProofs/Recourse/Plans.lean`, `Counterfactual.lean` and `Target.lean`. `Axioms.lean` prints the axioms
of each main theorem. The appendix gives the commands and the expected output.

= References <sec-references>

#bibliography("references.yml", title: none, style: "ieee")

= Appendix: how to repeat the check <sec-appendix>

*Versions.*
#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Item*], [*Value*],
  [Repository], [`https://github.com/johnny-godoy/EBaMl`],
  [Lean sources checked at commit], [`44773091b13e05f338b7bd0f05f52ddbb0dd12d5`],
  [Lean], [`leanprover/lean4:v4.35.0` (commit `eb6cafb7290e`)],
  [Mathlib], [tag `v4.35.0` (commit `56a005855ea17e0b0958f73ec801ae140b0a1679`)],
  [Typst], [`0.15.1`],
)

The files in `proofs/EBaMlProofs/Recourse/`, `proofs/lakefile.toml`, `proofs/lean-toolchain` and
`proofs/lake-manifest.json` are the same at the commit above and at the commit of this paper.

*Commands from a clean clone.*

```sh
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/johnny-godoy/EBaMl && cd EBaMl
git checkout 44773091b13e05f338b7bd0f05f52ddbb0dd12d5
cd proofs
lake exe cache get
lake build
typst compile --root papers papers/recourse/paper.typ
```

*Expected output of `lake build`.* The build ends with `Build completed successfully` and no line that starts with
`error`. The file `Recourse/Axioms.lean` prints these lines:

```text
info: EBaMlProofs/Recourse/Axioms.lean:12:0: 'EBaMlProofs.Recourse.Problem.ofPlan_feasible_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:13:0: 'EBaMlProofs.Recourse.Problem.objective_ofPlan' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:14:0: 'EBaMlProofs.Recourse.Problem.toPlan_valid' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:15:0: 'EBaMlProofs.Recourse.Problem.objective_toPlan' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:16:0: 'EBaMlProofs.Recourse.Problem.cheapest_of_optimum' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:17:0: 'EBaMlProofs.Recourse.Problem.cost_le_of_valid_imp' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:18:0: 'EBaMlProofs.Recourse.Problem.valid_of_gain_le' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:19:0: 'EBaMlProofs.Recourse.Problem.valid_of_limits_le' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:20:0: 'EBaMlProofs.Recourse.Problem.valid_excludeUsed_new' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:21:0: 'EBaMlProofs.Recourse.Problem.cost_mono_excludeUsed' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:22:0: 'EBaMlProofs.Recourse.margin_cfProblem' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:23:0: 'EBaMlProofs.Recourse.valid_cfProblem_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:24:0: 'EBaMlProofs.Recourse.sorting_cheapest' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:25:0: 'EBaMlProofs.Recourse.valid_of_higher_confidence' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:26:0: 'EBaMlProofs.Recourse.valid_embed' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:27:0: 'EBaMlProofs.Recourse.cost_embed' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:28:0: 'EBaMlProofs.Recourse.sorting_not_cheapest' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:29:0: 'EBaMlProofs.Recourse.knapsack_canFlipOn_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:30:0: 'EBaMlProofs.Recourse.Multiclass.margin_targetProblem' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:31:0: 'EBaMlProofs.Recourse.Multiclass.valid_targetProblem_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:32:0: 'EBaMlProofs.Recourse.Multiclass.reaches_of_valid' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:33:0: 'EBaMlProofs.Recourse.Multiclass.plan_of_reaching' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:34:0: 'EBaMlProofs.Recourse.Multiclass.exists_plan_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/Recourse/Axioms.lean:35:0: 'EBaMlProofs.Recourse.Multiclass.smallest_plan_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
```

*Statements and Lean names.* `P.` stands for `EBaMlProofs.Recourse.Problem.`, and `R.` for `EBaMlProofs.Recourse.`.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Statement*], [*Lean name*], [*File*],
  [Theorem 1], [`P.ofPlan_feasible_iff`, `P.objective_ofPlan`, `P.toPlan_valid`, `P.objective_toPlan`], [`Plans.lean`],
  [Corollary 1], [`P.cheapest_of_optimum`], [`Plans.lean`],
  [Theorem 2], [`P.cost_le_of_valid_imp`, `P.valid_of_gain_le`, `P.valid_of_limits_le`], [`Plans.lean`],
  [Theorem 3], [`P.valid_excludeUsed_new`, `P.cost_mono_excludeUsed`], [`Plans.lean`],
  [Lemma 1], [`R.margin_cfProblem`, `R.valid_cfProblem_iff`], [`Counterfactual.lean`],
  [Theorem 4], [`R.sorting_cheapest`, `R.cost_unit`], [`Counterfactual.lean`],
  [Lemma 2], [`R.sorting_not_cheapest`], [`Counterfactual.lean`],
  [Theorem 5], [`R.knapsack_canFlipOn_iff`], [`Counterfactual.lean`],
  [Theorem 6], [`R.valid_of_higher_confidence`, `R.valid_embed`, `R.cost_embed`], [`Counterfactual.lean`],
  [Lemma 3], [`R.Multiclass.margin_targetProblem`, `R.Multiclass.valid_targetProblem_iff`,
    `R.Multiclass.reaches_of_valid`], [`Target.lean`],
  [Theorem 7], [`R.Multiclass.plan_of_reaching`, `R.Multiclass.exists_plan_iff`, `R.Multiclass.smallest_plan_iff`],
    [`Target.lean`],
)
