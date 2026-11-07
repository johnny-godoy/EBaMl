// Paper: why sorting solves two queries of local EBM explanations exactly.
// Compile from proofs/ with: typst compile --root papers papers/local-explanations/paper.typ

#import "/style.typ": *
#show: paper.with(title: "Optimality of the Greedy Local Explanations of EBaMl")




// ---- Title ----------------------------------------------------------------------------------------------------

#align(center)[
  #text(size: 17pt, weight: "bold")[Optimality of the Greedy Local Explanations of EBaMl]
  #v(0.4em)
  #text(size: 11pt)[A proof checked in Lean 4]
]

#v(1em)

*Summary.* EBaMl explains one prediction of a binary, multiclass or regression Explainable Boosting Machine (EBM) in two exact ways. The
_minimum change required_ is the fewest features to change so that the prediction flips. The _minimum sufficient
reason_ is the fewest features to fix so that no values of the other features can flip the prediction. Both queries
come from Barceló, Monet, Pérez and Subercaseaux @barcelo2020. The code in `src/ebaml/local/facts.py` solves both with
one sort. This paper proves that the sort is optimal. A regressor reduces to the binary case by a shift of the intercept, and the answer "inside an interval" has two margins.
The counterfactual of two margins is the smaller of two exact answers. The sufficient reason of two margins is a
covering problem, and the paper proves what two searches return. A multiclass classifier has one margin for each rival
class, so it uses the same results. A Lean 4 file @lean4 checks every statement, and the
appendix tells the reader how to repeat the check.

= Claims and scope <sec-claims>

The proof covers a binary EBM, a multiclass EBM and a regressor, all without interaction terms. The score of such a model is a sum of
one term per feature, and this is the only fact that the proof needs.

*Claim 1 (minimum change required).* The function `_counterfactual` returns a set of features whose change flips
the prediction. No set with fewer features has a change that flips the prediction. If the function returns `None`, no
change of any set of features flips the prediction.

*Claim 2 (minimum sufficient reason).* The function `_sufficient_reason`, with the order of `_fixing_order`, returns
a set of features that is sufficient. No set with fewer features is sufficient. If the function returns `None`, no set
of features is sufficient, and the set of all features is not sufficient either.

*Claim 3 (regressor).* For a regressor with a threshold $tau$, the margin of the decision is the margin of the binary
model with the intercept $b - tau$. Claims 1 and 2 therefore hold for the decisions "above $tau$" and "at most $tau$"
with no new proof (@sec-regressor, Lemmas 7 and 8). The code does this shift in `compute_local_facts`.

*Claim 4 (inside an interval).* When the predicted value is inside the interval, the decision has two margins. The
fewest features that flip it are the fewest of the two answers for the margins, and `_smallest_counterfactual` returns
this number (Theorem 6).

*Claim 5 (searches).* For a decision with several margins, `Minimal` returns a sufficient set from which no feature
can be removed, and `ExactOrMinimal` returns a smallest sufficient set when the number of features is at most its
limit. `Minimal` does not claim a smallest set, except for a set of at most one feature (Theorems 4 and 5, Corollary 3).

*Claim 6 (multiclass classifier).* For a multiclass classifier, the predicted class $w$ has one margin for each rival
class $j$: the score of $w$ minus the score of $j$, with the deviation $sigma_w + sigma_j$. The fewest features that
change the predicted class are the fewest of the answers for the rivals, and `_smallest_counterfactual` returns this
number (Theorem 7). A set keeps the class iff it is sufficient for the margin of every rival, so Claim 5 holds for the
sufficient reason of a multiclass classifier (Lemma 11). The sum $sigma_w + sigma_j$ is an upper bound of the deviation
of the margin for any correlation between the two scores (Lemma 12).

The proof does not cover the other parts of `facts.py`. @sec-limits lists them. The text of the module docstring
differs a little from the code for the counterfactual. @sec-limits explains the difference.

= Model <sec-model>

This section fixes the model and the glossary. The model follows `facts.py` line by line. The glossary gives one
term to each concept, and the text uses this term everywhere.

*Glossary.*
#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Term*], [*Meaning*],
  [feature], [An element $i$ of the finite set $I$. The set $I$ has $n$ elements.],
  [bin], [A value class of a feature. Feature $i$ has the set of bins $B_i$. A categorical value is one bin.],
  [score], [The mean score $s_i (v)$ of bin $v$ of feature $i$. It is a real number.],
  [deviation], [The standard deviation $sigma_i (v) >= 0$ of the score of bin $v$.],
  [plausible bin], [A bin with enough training support. The set $P_i subset.eq B_i$ holds them.],
  [instance], [The input to explain. It has bin $c_i in P_i$ for each feature $i$.],
  [margin], [The distance of the prediction from the decision boundary, seen from the predicted class.],
  [change], [A set $S$ of features and a plausible bin $w_i$ for each feature in $S$.],
  [flip], [A change _flips_ the prediction when its robust margin is below zero.],
  [fixed], [A feature is _fixed_ when its bin is known to be $c_i$.],
  [sufficient], [A set $F$ of fixed features is _sufficient_ when its worst-case margin is above zero.],
  [drop, gain], [The numbers $d_i$ and $h_i$ of @sec-model. The code sorts them.],
  [threshold], [The score $tau$ at which a margin changes sign. It is $0$ for a classifier.],
  [decision], [The claim that the explanations keep. A decision holds while every one of its margins is positive.],
)

The code reads a model with one term per feature, and it rejects interaction terms (`_features`). The model is
therefore the data below. A bin set $B_i$ may be any set. Only $P_i$ must be finite, and it is, because the
code stores one flag per bin.

#notation("data", [
  The finite set $I$ of features. For each feature $i$: the set $B_i$, the score $s_i : B_i -> RR$, the deviation
  $sigma_i : B_i -> RR$ with $sigma_i >= 0$, the finite set $P_i subset.eq B_i$ of plausible bins, and the bin
  $c_i in P_i$ of the instance. The intercept $b in RR$. The number $z >= 0$ of deviations. The code takes $z$ from
  `confidence_level` in $[0, 1)$ with the inverse normal distribution, so $z >= 0$.
])

The code marks $c_i$ as plausible for each feature (in `_features`). Therefore $P_i$ is not empty, and a minimum
over $P_i$ exists.

#notation("sign and margin", [
  The logit is $L = b + sum_(i in I) s_i (c_i)$. The sign is $epsilon = 1$ if $L > 0$, and $epsilon = -1$ otherwise.
  The signed score is $t_i (v) = epsilon dot s_i (v)$. The margin is
  $ M = epsilon b + sum_(i in I) t_i (c_i) = epsilon L. $
])

The sign is $-1$ for $L = 0$. This is the same as in the code (`sign = 1 if logit > 0 else -1`).

#notation("pessimistic and guaranteed score", [
  For $v in P_i$ let $p_i (v) = t_i (v) + z sigma_i (v)$ and $g_i (v) = t_i (v) - z sigma_i (v)$.
  The score $p_i (v)$ is the highest signed score of bin $v$ within $z$ deviations. It is the hardest case for a
  flip. The score $g_i (v)$ is the lowest signed score within $z$ deviations. It is the hardest case for the
  prediction to stay.
])

The code gives the value $+infinity$ to each bin outside $P_i$ (in `_robust_scores`). A minimum over all bins is then
a minimum over $P_i$. The paper restricts the minimum to $P_i$, and so it needs no extended reals.

#notation("change and robust margin", [
  A _change_ is a set $S subset.eq I$ and a bin $w_i in P_i$ for each $i in S$. Its robust margin is
  $ R(S, w) = epsilon b + sum_(i in.not S) t_i (c_i) + sum_(i in S) p_i (w_i). $
  The change _flips_ the prediction iff $R(S, w) < 0$.
])

A feature outside $S$ keeps its score without deviation. The code does this (in `_counterfactual`). The module
docstring says that every score moves by $z$ deviations. @sec-limits discusses this difference.

#notation("drop", [
  $d_i = t_i (c_i) - min_(v in P_i) p_i (v)$. Let $beta_i in P_i$ be a bin where $p_i$ reaches this minimum.
])

#notation("worst-case margin and gain", [
  For a set $F subset.eq I$ of fixed features,
  $ W(F) = epsilon b + sum_(i in F) g_i (c_i) + sum_(i in.not F) min_(v in P_i) g_i (v). $
  The set $F$ is _sufficient_ iff $W(F) > 0$. The gain of feature $i$ is
  $h_i = g_i (c_i) - min_(v in P_i) g_i (v)$. Then $h_i >= 0$, because $c_i in P_i$.
])

#notation("greedy rule", [
  Let $e : {0, dots, n - 1} -> I$ be a bijection such that the values $d_(e(0)), dots, d_(e(n-1))$ do not increase
  (for the gains, the values $h_(e(j))$). Ties may have any order. Let $Pi_k$ be the sum of the first $k$ values.
  The code returns the first $k$ with $Pi_k > M$ for the drops (the search starts at $k = 1$), and the first $k$
  with $W(emptyset) + Pi_k > 0$ for the gains (the search starts at $k = 0$). If no $k <= n$ has this property,
  the code returns `None`.
])

*Correspondence of symbols to the code.* The code is in `src/ebaml/local/facts.py`. The table names the function
that computes each symbol.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Symbol*], [*Computed by*], [*Function*],
  [$z$], [`statistics.NormalDist().inv_cdf((1 + confidence_level) / 2)`], [`compute_local_facts`],
  [$b$], [`intercept = float(np.ravel(ebm.intercept_)[0])`], [`compute_local_facts`],
  [$L$], [`logit = intercept + sum(f.current ...)`], [`compute_local_facts`],
  [$epsilon$], [`sign = 1 if logit > 0 else -1`], [`compute_local_facts`],
  [$t_i (v)$], [`scores = sign * feature.scores`], [`_robust_scores`],
  [$P_i$, $c_i in P_i$], [`_plausible_bins`, then `plausible[bin_index] = True`], [`_plausible_bins, _features`],
  [$p_i$, $g_i$], [`scores + stds` and `scores - stds`, with `np.inf` outside $P_i$], [`_robust_scores`],
  [$M$], [`margin = sign * intercept + sum(sign * f.current ...)`], [`_counterfactual`],
  [$min p_i$, $beta_i$], [`best = int(np.argmin(pessimistic))`], [`_counterfactual`],
  [$d_i$], [`sign * feature.current - float(pessimistic[best])`], [`_counterfactual`],
  [order $e$ for $d$], [`sorted(drops, ..., reverse=True)`], [`_counterfactual`],
  [first $k$ with $Pi_k > M$], [`np.flatnonzero(margin - np.cumsum(...) < 0)`, `flips[0] + 1`], [`_counterfactual`],
  [$h_i$], [`guaranteed[feature.bin_index] - guaranteed.min()`], [`_fixing_order`],
  [order $e$ for $h$], [`sorted(gains, ..., reverse=True)`], [`_fixing_order`],
  [$W(emptyset)$], [`worst = sign * intercept + sum(... .min() ...)`], [`_sufficient_reason`],
  [$W(emptyset) + Pi_k$], [`margins = worst + np.concatenate(([0.0], np.cumsum(...)))`], [`_sufficient_reason`],
  [first $k$ with $W(emptyset) + Pi_k > 0$], [`np.flatnonzero(margins > 0)`, `enough[0]`], [`_sufficient_reason`],
)

The Lean file `proofs/EBaMlProofs/LocalExplanations/Model.lean` states these notations as the structure `EBaMlProofs.LocalExplanations.Model` and the
definitions `logit`, `eps`, `t`, `margin`, `p`, `g`, `pmin`, `drop`, `gain`, `beta`, `robustMargin`, `Admissible`,
`CanFlip`, `worst` and `Sufficient`.

= The abstract result <sec-abstract>

This section knows nothing about EBMs. Let $I$ be a finite set with $n$ elements, and let $d : I -> RR$.
For a finite set $S$, write $d(S) = sum_(i in S) d(i)$.

#definition("top set", [
  A set $T subset.eq I$ is a _top set_ of $d$ iff $d(j) <= d(i)$ for each $i in T$ and each $j in.not T$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.IsTopSet"))

#lemma("top-set bound", [
  If $T$ is a top set of $d$ and $|S| = |T|$, then $d(S) <= d(T)$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.topSet_sum_le"))

#v(0.3em)
*Warning.* The condition $|S| = |T|$ is necessary. Take $d = (5, -1)$, $T = I$ and $S = {1}$. Then $T$ is a top
set, but $d(S) = 5 > 4 = d(T)$.
#v(0.5em)

#proof[
  *ASSUME:* $T$ is a top set of $d$, and $|S| = |T|$. *PROVE:* $d(S) <= d(T)$.

  #step(1, "1", [$d(S) - d(T) = d(S without T) - d(T without S)$, because both sets contain the part $S inter T$.])
  #step(1, "2", [$|S without T| = |T without S|$, because $|S| = |T|$ and both differences remove $|S inter T|$.])
  #step(1, "3", [*Case* $S without T = emptyset$.])
  #step(2, "1", [$T without S = emptyset$ by $chevron.l 1 chevron.r 2$.])
  #step(2, "2", [*Q.E.D.* Both differences are empty sums, so $d(S) - d(T) = 0$ by $chevron.l 1 chevron.r 1$.])
  #step(1, "4", [*Case* $S without T eq.not emptyset$. Choose $i_0 in S without T$ with $d(i_0)$ largest in
    $S without T$.])
  #step(2, "1", [$d(i) <= d(i_0)$ for each $i in S without T$, by the choice of $i_0$.])
  #step(2, "2", [$d(i_0) <= d(j)$ for each $j in T without S$, because $j in T$, $i_0 in.not T$, and
    Definition 1.])
  #step(2, "3", [$d(S without T) <= |S without T| dot d(i_0) = |T without S| dot d(i_0) <= d(T without S)$,
    by $chevron.l 2 chevron.r 1$, $chevron.l 1 chevron.r 2$ and $chevron.l 2 chevron.r 2$.])
  #step(2, "4", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ and $chevron.l 2 chevron.r 3$.])
  #step(1, "5", [*Q.E.D.* by $chevron.l 1 chevron.r 3$ and $chevron.l 1 chevron.r 4$.])
]

The proof compares each value of $S without T$ with one common bound $d(i_0)$. This replaces the pairing of the
two sets by a bijection. It needs no choice of a bijection and gives the same bound.

#lemma("prefixes are top sets", [
  Let $e : {0, dots, n - 1} -> I$ be a bijection such that $d compose e$ does not increase. Let
  $T_k = e({0, dots, k - 1})$ and $Pi_k = sum_(j < k) d(e(j))$. For $k <= n$, the set $T_k$ is a top set of $d$,
  $|T_k| = k$, and $d(T_k) = Pi_k$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.isTopSet_prefixSet", "EBaMlProofs.LocalExplanations.Greedy.card_prefixSet", "EBaMlProofs.LocalExplanations.Greedy.sum_prefixSet"))

#proof[
  *ASSUME:* $e$ is a bijection, $d compose e$ does not increase, and $k <= n$. *PROVE:* the three claims.

  #step(1, "1", [$|T_k| = k$, because $e$ is injective.])
  #step(1, "2", [$d(T_k) = Pi_k$, because $e$ maps the positions $0, dots, k - 1$ one-to-one onto $T_k$.])
  #step(1, "3", [$T_k$ is a top set.])
  #step(2, "1", [*Let* $i in T_k$ and $j in.not T_k$. *Let* $a = e^(-1)(i)$ and $b = e^(-1)(j)$.])
  #step(2, "2", [$a < k <= b$, by the definition of $T_k$.])
  #step(2, "3", [$d(j) = d(e(b)) <= d(e(a)) = d(i)$, because $d compose e$ does not increase and $a < b$.])
  #step(2, "4", [*Q.E.D.* by $chevron.l 2 chevron.r 3$.])
  #step(1, "4", [*Q.E.D.* by $chevron.l 1 chevron.r 1$, $chevron.l 1 chevron.r 2$ and $chevron.l 1 chevron.r 3$.])
]

#theorem("greedy threshold", [
  Let $tau in RR$, and let $e$ be as in Lemma 2. *Assume* $k <= n$, $Pi_k > tau$, and $Pi_j <= tau$ for each
  $j < k$. Then (a) $d(T_k) > tau$, and (b) each set $S$ with $d(S) > tau$ has $|S| >= k$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.threshold_minimal"))

#proof[
  *ASSUME:* $k <= n$, $Pi_k > tau$, and $Pi_j <= tau$ for each $j < k$. *PROVE:* (a) and (b).

  #step(1, "1", [(a) holds, because $d(T_k) = Pi_k > tau$ by Lemma 2.])
  #step(1, "2", [(b) holds. *Assume* $|S| < k$. *Prove* $d(S) <= tau$.])
  #step(2, "1", [$|S| <= n$, so $T_(|S|)$ exists, and $|T_(|S|)| = |S|$ by Lemma 2.])
  #step(2, "2", [$d(S) <= d(T_(|S|)) = Pi_(|S|)$ by Lemma 1, Lemma 2 and $chevron.l 2 chevron.r 1$.])
  #step(2, "3", [$Pi_(|S|) <= tau$ by the assumption, because $|S| < k$.])
  #step(2, "4", [*Q.E.D.* by $chevron.l 2 chevron.r 2$ and $chevron.l 2 chevron.r 3$.])
  #step(1, "3", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ and $chevron.l 1 chevron.r 2$.])
]

#corollary("no answer", [
  If $Pi_k <= tau$ for each $k <= n$, then $d(S) <= tau$ for each set $S$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.no_answer"))

#proof[
  Let $S$ be a set. Then $|S| <= n$, and $d(S) <= d(T_(|S|)) = Pi_(|S|) <= tau$ by Lemma 1, Lemma 2 and the
  assumption.
]

#corollary("chosen values are positive", [
  Under the assumptions of Theorem 1, $d(e(j)) > 0$ for each $j < k$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Greedy.lean", "EBaMlProofs.LocalExplanations.Greedy.pos_of_threshold"))

#proof[
  *ASSUME:* the assumptions of Theorem 1, and $j < k$ (so $k >= 1$). *PROVE:* $d(e(j)) > 0$.

  #step(1, "1", [$d(e(k - 1)) = Pi_k - Pi_(k-1) > 0$, because $Pi_(k-1) <= tau < Pi_k$.])
  #step(1, "2", [$d(e(j)) >= d(e(k - 1))$, because $j <= k - 1$ and $d compose e$ does not increase.])
  #step(1, "3", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ and $chevron.l 1 chevron.r 2$.])
]

= Application to EBMs <sec-application>

This section applies Theorem 1 to the model of @sec-model. The sum of all scores is additive over the features,
and this is the reason why the abstract result applies.

#lemma("the margin is not negative", [
  $M >= 0$. Therefore no change with $S = emptyset$ flips the prediction.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Model.lean", "EBaMlProofs.LocalExplanations.Model.margin_nonneg"))

#proof[
  *ASSUME:* the model of @sec-model. *PROVE:* $M >= 0$.

  #step(1, "1", [*Case* $L > 0$. Then $epsilon = 1$ and $M = L > 0$.])
  #step(1, "2", [*Case* $L <= 0$. Then $epsilon = -1$ and $M = -L >= 0$.])
  #step(1, "3", [*Q.E.D.* by the two cases, which cover all values of $L$.])
  The change $(emptyset, w)$ has $R(emptyset, w) = M >= 0$. The code searches from $k = 1$, so it loses nothing.
]

#lemma("best change of a set", [
  For each change $(S, w)$: $R(S, w) = M - sum_(i in S) (t_i (c_i) - p_i (w_i)) >= M - d(S)$, with equality when
  $w = beta$. Therefore a set $S$ has a change that flips the prediction iff $d(S) > M$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Model.lean", "EBaMlProofs.LocalExplanations.Model.robustMargin_eq", "EBaMlProofs.LocalExplanations.Model.robustMargin_ge", "EBaMlProofs.LocalExplanations.Model.robustMargin_beta", "EBaMlProofs.LocalExplanations.Model.canFlip_iff"))

#proof[
  *ASSUME:* a change $(S, w)$. *PROVE:* the equation, the bound, the equality and the equivalence.

  #step(1, "1", [The equation holds. The sum in $R(S, w)$ differs from the sum in $M$ only in the terms with
    $i in S$. There $p_i (w_i)$ replaces $t_i (c_i)$.])
  #step(1, "2", [The bound holds. For $i in S$ the bin $w_i$ is in $P_i$, so $p_i (w_i) >= min_(v in P_i) p_i (v)$.
    Hence $t_i (c_i) - p_i (w_i) <= d_i$. Sum over $S$ and use $chevron.l 1 chevron.r 1$.])
  #step(1, "3", [The equality holds for $w = beta$, because $p_i (beta_i) = min_(v in P_i) p_i (v)$, so each term
    equals $d_i$.])
  #step(1, "4", [*Case* a change $(S, w)$ flips the prediction. Then $0 > R(S, w) >= M - d(S)$ by
    $chevron.l 1 chevron.r 2$, so $d(S) > M$.])
  #step(1, "5", [*Case* $d(S) > M$. Then $R(S, beta) = M - d(S) < 0$ by $chevron.l 1 chevron.r 3$, and $beta$ is a
    change because $beta_i in P_i$.])
  #step(1, "6", [*Q.E.D.* by $chevron.l 1 chevron.r 4$ and $chevron.l 1 chevron.r 5$.])
]

#lemma("fixing adds the gain", [
  $W(F) = W(emptyset) + sum_(i in F) h_i$. Therefore $F$ is sufficient iff $sum_(i in F) h_i > -W(emptyset)$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Model.lean", "EBaMlProofs.LocalExplanations.Model.worst_eq", "EBaMlProofs.LocalExplanations.Model.sufficient_iff"))

#proof[
  *ASSUME:* a set $F$. *PROVE:* the equation and the equivalence.

  #step(1, "1", [For each feature $i$ the term of $W(F)$ is $min_(v in P_i) g_i (v)$ if $i in.not F$, and
    $g_i (c_i) = min_(v in P_i) g_i (v) + h_i$ if $i in F$.])
  #step(1, "2", [The equation holds. Sum $chevron.l 1 chevron.r 1$ over all features. The sum of the minima is the
    part of $W(emptyset)$ without $epsilon b$, and the extra terms are $sum_(i in F) h_i$.])
  #step(1, "3", [*Q.E.D.* The equivalence follows from the equation and the definition $W(F) > 0$.])
]

#theorem("minimum change", [
  Let $e$ sort the drops $d_i$ in descending order, and let $k$ be the first position with $Pi_k > M$ (so $k <= n$,
  $Pi_k > M$, and $Pi_j <= M$ for $1 <= j < k$). Then
  (a) the change of the first $k$ features to the bins $beta_i$ flips the prediction,
  (b) no change of fewer than $k$ features flips the prediction,
  (c) each of these $k$ features moves to a bin $beta_i eq.not c_i$, and
  (d) if $Pi_k <= M$ for each $k <= n$, no change of any set of features flips the prediction.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Model.lean", "EBaMlProofs.LocalExplanations.Model.min_change (a, b)", "EBaMlProofs.LocalExplanations.Model.chosen_bin_differs (c)", "EBaMlProofs.LocalExplanations.Model.min_change_none (d)"))

#proof[
  *ASSUME:* the model, an ordering $e$ of the drops, and the conditions of the statement. *PROVE:* (a)–(d).

  #step(1, "1", [$Pi_0 = 0 <= M$ by Lemma 3, so $Pi_j <= M$ for each $j < k$ including $j = 0$.])
  #step(1, "2", [Theorem 1 with $d$ the drops and $tau = M$ applies, by $chevron.l 1 chevron.r 1$.])
  #step(1, "3", [(a) holds. Theorem 1(a) gives $d(T_k) > M$, and Lemma 4 gives $R(T_k, beta) = M - d(T_k) < 0$.])
  #step(1, "4", [(b) holds. A change of a set $S$ that flips the prediction gives $d(S) > M$ by Lemma 4.
    Theorem 1(b) gives $|S| >= k$.])
  #step(1, "5", [(c) holds. Let $j < k$ and $i = e(j)$.])
  #step(2, "1", [$d_i > 0$ by Corollary 2.])
  #step(2, "2", [*Assume* $beta_i = c_i$. Then $min_(v in P_i) p_i (v) = p_i (c_i) = t_i (c_i) + z sigma_i (c_i)
    >= t_i (c_i)$, because $z >= 0$ and $sigma_i >= 0$. Hence $d_i <= 0$. This contradicts
    $chevron.l 2 chevron.r 1$.])
  #step(2, "3", [*Q.E.D.* by $chevron.l 2 chevron.r 2$.])
  #step(1, "6", [(d) holds. By Corollary 1, $d(S) <= M$ for each set $S$. Lemma 4 then shows that no change of
    $S$ flips the prediction.])
  #step(1, "7", [*Q.E.D.* by $chevron.l 1 chevron.r 3$ to $chevron.l 1 chevron.r 6$.])
]

#theorem("minimum sufficient reason", [
  Let $e$ sort the gains $h_i$ in descending order, and let $k$ be the first position, counted from $0$, with
  $W(emptyset) + Pi_k > 0$. Then
  (a) the first $k$ features are sufficient,
  (b) no set of fewer than $k$ features is sufficient, and
  (c) if $W(emptyset) + Pi_k <= 0$ for each $k <= n$, no set of features is sufficient. This includes the set of all
  features.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Model.lean", "EBaMlProofs.LocalExplanations.Model.min_sufficient_reason (a, b)", "EBaMlProofs.LocalExplanations.Model.min_sufficient_reason_none (c)"))

#proof[
  *ASSUME:* the model, an ordering $e$ of the gains, and the conditions of the statement. *PROVE:* (a)–(c).

  #step(1, "1", [Theorem 1 with $d$ the gains and $tau = -W(emptyset)$ applies. The condition
    $W(emptyset) + Pi_k > 0$ means $Pi_k > tau$, and $W(emptyset) + Pi_j <= 0$ for $j < k$ means $Pi_j <= tau$.
    For $k = 0$ the second condition is empty.])
  #step(1, "2", [(a) holds. Theorem 1(a) gives $h(T_k) > -W(emptyset)$, so $T_k$ is sufficient by Lemma 5.])
  #step(1, "3", [(b) holds. A sufficient set $F$ has $h(F) > -W(emptyset)$ by Lemma 5, so $|F| >= k$ by
    Theorem 1(b).])
  #step(1, "4", [(c) holds. By Corollary 1, $h(F) <= -W(emptyset)$ for each set $F$, so no set is sufficient by
    Lemma 5.])
  #step(1, "5", [*Q.E.D.* by $chevron.l 1 chevron.r 2$ to $chevron.l 1 chevron.r 4$.])
]

The code returns the first $k$ in each case, and `None` in the case of Theorem 2(d) and Theorem 3(c). Claims 1 and 2 of
@sec-claims follow from Theorems 2 and 3, provided that the code computes the quantities of @sec-model. The
table in @sec-model shows where.

= Decisions with several margins <sec-covering>

A decision such as "inside an interval" holds while two margins are positive at the same time. This section knows
nothing about EBMs. A _problem_ has a finite set $I$ of features, a finite index set $K$ of margins, a number
$W_k$ for each margin (the worst case when no feature is fixed), and a gain $h_(k,i) >= 0$ for each margin $k$ and
feature $i$. A set $F subset.eq I$ is _sufficient_ iff $W_k + sum_(i in F) h_(k,i) > 0$ for each $k in K$. With one
margin this is the definition of @sec-model.

#lemma("monotone", [
  If $F subset.eq G$ and $F$ is sufficient, then $G$ is sufficient.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Covering.lean", "EBaMlProofs.LocalExplanations.Covering.Problem.sufficient_mono"))

#proof[
  Let $k in K$. The gains are not negative, so $sum_(i in F) h_(k,i) <= sum_(i in G) h_(k,i)$. Hence
  $W_k + sum_(i in G) h_(k,i) >= W_k + sum_(i in F) h_(k,i) > 0$.
]

The search `Minimal` has two passes. The first pass orders the features by their smallest gain over the margins and
adds them until the set is sufficient. The second pass, _prune_, visits the added features from the last to the first.
It removes a feature when the set stays sufficient without it. The proof allows any order in both passes. Let
$"prune"(l, C)$ be the result for a list $l$ of features and a set $C$ at the start of the second pass.

#theorem("the removal pass", [
  Let $C$ be sufficient, and let $l$ hold every feature of $C$. Then $"prune"(l, C)$ is a sufficient subset of $C$,
  and no feature can be removed from it: for each $f in "prune"(l, C)$, the set $"prune"(l, C) without {f}$ is not
  sufficient.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Covering.lean", "EBaMlProofs.LocalExplanations.Covering.Problem.prune_minimal", "EBaMlProofs.LocalExplanations.Covering.Problem.prune_invariant"))

#proof[
  *ASSUME:* a sufficient set $C$, and a list $l$ that holds every feature of $C$. *PROVE:* the three claims.
  *Invariant* for a list $l$ and a set $C$: $C$ is sufficient, and each $f in C$ is in $l$ or is needed (that is,
  $C without {f}$ is not sufficient). We show by induction on $l$ that the invariant gives the three claims.

  #step(1, "1", [*Case* $l$ is empty. Then $"prune"(l, C) = C$, and each $f in C$ is needed, because it is not in
    $l$.])
  #step(1, "2", [*Case* $l = f_0 :: l'$. Let $C'$ be $C without {f_0}$ if this set is sufficient, and $C$ otherwise.
    The pass continues with $C'$ and $l'$.])
  #step(2, "1", [$C'$ is sufficient, and $C' subset.eq C$, by the choice of $C'$.])
  #step(2, "2", [Each $f in C'$ is in $l'$ or is needed in $C'$.])
  #step(3, "1", [*Case* $f$ is in $l'$. Done.])
  #step(3, "2", [*Case* $f = f_0$. Then $f_0 in C'$, so the first choice was not taken, $C' = C$, and
    $C without {f_0}$ is not sufficient. Hence $f_0$ is needed in $C'$.])
  #step(3, "3", [*Case* $f$ is needed in $C$. Then $C' without {f} subset.eq C without {f}$. If $C' without {f}$ were
    sufficient, then $C without {f}$ would be sufficient by Lemma 6. So $f$ is needed in $C'$.])
  #step(3, "4", [*Q.E.D.* The three cases cover each $f in C'$, because $f in C$ and the invariant holds for $C$.])
  #step(2, "3", [The invariant holds for $C'$ and $l'$ by $chevron.l 2 chevron.r 1$ and $chevron.l 2 chevron.r 2$.
    The induction hypothesis gives the three claims for the result, and the result is a subset of $C' subset.eq C$.])
  #step(1, "3", [*Q.E.D.* The assumption of the theorem is the invariant for $C$, because each $f in C$ is in $l$.])
]

The search `ExactOrMinimal` tries the sets by size and returns the first sufficient set.

#theorem("the exact search", [
  Let $F$ be sufficient, and let no set with fewer than $|F|$ features be sufficient. Then each sufficient set has
  at least $|F|$ features. The search visits at most $2^n$ sets for $n$ features.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Covering.lean", "EBaMlProofs.LocalExplanations.Covering.Problem.exact_minimum", "EBaMlProofs.LocalExplanations.Covering.Problem.card_search_space"))

#proof[
  A sufficient set $G$ with $|G| < |F|$ would contradict the assumption. The sets that the search visits are
  subsets of $I$, and there are $2^n$ of them.
]

#corollary("a set of at most one feature", [
  Let $R$ be sufficient with $|R| <= 1$. If $R$ is empty, or the empty set is not sufficient, then each sufficient
  set has at least $|R|$ features.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Covering.lean", "EBaMlProofs.LocalExplanations.Covering.Problem.minimum_of_card_le_one"))

#proof[
  If $R$ is empty, the claim is $0 <= |G|$. Otherwise the empty set is not sufficient, so each sufficient set $G$ is
  not empty, and $|G| >= 1 >= |R|$.
]

The code sets the flag `minimum` of the result of `Minimal` to `len(chosen) <= 1`. Corollary 3 proves this flag. The
code does not claim a smallest set for a larger result, and the proof shows no such claim. An example shows why. Take
two margins with $W = (-6, -2)$ and four features with the gains $h_1 = (2, 2)$, $h_2 = (2, 2)$, $h_3 = (3, 3)$ and
$h_4 = (5, 1)$ for the two margins. The first pass adds the features $3, 1, 2$ (in the order of the smallest gain)
and the set is sufficient. No feature can be removed, but $\{3, 4\}$ is also sufficient and has fewer features.

= Regressors and the answer inside an interval <sec-regressor>

A regressor predicts the value $L = b + sum_i s_i (c_i)$. A decision is a claim about $L$. The code gives it as a
list of _margins_ (`Margin` in `decisions.py`). Each margin has a sign $sigma in {1, -1}$ and a threshold $tau$, and its
value is $sigma (L - tau)$. A binary classifier is the case with one margin and $tau = 0$.
The code (`Threshold.side` and `Interval.side`) chooses the margins as follows.

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Claim that holds*], [*Margins $(sigma, tau)$*],
  [above $t$], [$(1, t)$],
  [at most $t$], [$(-1, t)$],
  [above $l$ and at most $u$ (inside)], [$(1, l)$ and $(-1, u)$],
)

The next lemmas apply to the margin with the data $(sigma, tau)$ when $sigma$ is the sign that the claim has. Let
$m_tau$ be the model of @sec-model with the intercept $b - tau$ and the same scores, deviations, plausible bins and
instance bins. The code does this shift: it passes `intercept - margin.threshold` to the algorithms (`_Bound` in
`compute_local_facts`).

#lemma("the shift", [
  (a) The logit of $m_tau$ is $L - tau$. (b) The sign of $m_tau$ is $1$ if $tau < L$, and $-1$ otherwise. (c) The
  margin of $m_tau$ is $epsilon (L - tau)$ with this sign. So a prediction above $tau$ has the margin $L - tau$ of the
  claim "above $tau$", and any other prediction has the margin $tau - L$ of the claim "at most $tau$".
], lean: ("proofs/EBaMlProofs/LocalExplanations/Regressor.lean", "EBaMlProofs.LocalExplanations.Model.shift_logit", "EBaMlProofs.LocalExplanations.Model.shift_eps", "EBaMlProofs.LocalExplanations.Model.shift_margin_eq"))

#proof[
  (a) $(b - tau) + sum_i s_i (c_i) = L - tau$. (b) The sign is $1$ iff the logit $L - tau$ is above $0$, which holds
  iff $tau < L$. (c) The margin is the sign times the logit (Notation of @sec-model), and the logit is $L - tau$.
]

The shifted model $m_tau$ is a model of @sec-model. Theorems 2 and 3 therefore hold for it, and they are the
statements about the regressor decision "above $tau$" or "at most $tau$" (Claim 3). The function
`_counterfactual` and the function `_sufficient_reason` run on $m_tau$.

#lemma("a flip crosses the threshold", [
  Let $z = 0$, and let $x$ be the predicted value after a change $(S, w)$. The robust margin $R(S, w)$ of $m_tau$ is
  negative iff $x < tau$ when $tau < L$, and iff $x > tau$ when $L <= tau$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Regressor.lean", "EBaMlProofs.LocalExplanations.Model.shift_robustMargin_neg_iff"))

#proof[
  With $z = 0$ the pessimistic score is the signed score, so $R(S, w) = epsilon (x - tau)$ for the sign $epsilon$ of
  $m_tau$. For $epsilon = 1$ this is negative iff $x < tau$. For $epsilon = -1$ it is negative iff $x > tau$.
]

A change that the code finds therefore moves the value to the other side of $tau$. The converse fails only at the
point $x = tau$ for a prediction above $tau$. @sec-limits discusses this tie.

When the predicted value is inside the interval $(l, u]$, the two shifted models have the signs $1$ and $-1$.
A change flips the decision "inside" iff it makes one of the two robust margins negative.

#theorem("inside an interval", [
  Let $l < L <= u$. Let $k_1$ and $k_2$ be the positions of Theorem 2 for the models $m_l$ and $m_u$. Then
  (a) the signs of $m_l$ and $m_u$ are $1$ and $-1$,
  (b) a set $S$ flips the decision iff it flips the margin of $m_l$ or the margin of $m_u$,
  (c) the smallest set that flips the decision has $min(k_1, k_2)$ features, and
  (d) if neither model has a position, no set flips the decision.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Regressor.lean", "EBaMlProofs.LocalExplanations.Model.inside_eps (a)", "EBaMlProofs.LocalExplanations.Model.canFlipInside_iff (b)", "EBaMlProofs.LocalExplanations.Model.min_change_inside (c)", "EBaMlProofs.LocalExplanations.Model.min_change_inside_none (d)"))

#proof[
  *ASSUME:* $l < L <= u$ and the positions $k_1$ and $k_2$ of Theorem 2. *PROVE:* (a)–(d).

  #step(1, "1", [(a) holds by Lemma 7(b), because $l < L$ and $not (u < L)$.])
  #step(1, "2", [(b) holds. A change $(S, w)$ flips the decision iff $R_l (S, w) < 0$ or $R_u (S, w) < 0$. The
    plausible bins and the set $S$ are the same for both models. So the change flips one margin iff $S$ can flip
    that margin.])
  #step(1, "3", [(c) holds.])
  #step(2, "1", [The first $min(k_1, k_2)$ features of the model with the smaller position, moved to the bins
    $beta$, flip the decision, by Theorem 2(a) and $chevron.l 1 chevron.r 2$.])
  #step(2, "2", [A set $S$ that flips the decision has $|S| >= k_1$ or $|S| >= k_2$ by Theorem 2(b) and
    $chevron.l 1 chevron.r 2$. So $|S| >= min(k_1, k_2)$.])
  #step(2, "3", [*Q.E.D.* by $chevron.l 2 chevron.r 1$ and $chevron.l 2 chevron.r 2$.])
  #step(1, "4", [(d) holds by Theorem 2(d) for each model and $chevron.l 1 chevron.r 2$.])
  #step(1, "5", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ to $chevron.l 1 chevron.r 4$.])
]

The code takes the first of the two answers when both have the same size (`min` in `_smallest_counterfactual`). The
size is the same, and the named features may differ.

The sufficient reason of a decision with the margins $m_k$ is a covering problem.

#lemma("several margins", [
  Let $m_k$ be the shifted models of the margins of a decision. A set $F$ is sufficient for the problem with
  $W_k = W_k (emptyset)$ and the gains $h_(k,i)$ of $m_k$ iff $F$ is sufficient for each model $m_k$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Regressor.lean", "EBaMlProofs.LocalExplanations.problemOfModels_sufficient"))

#proof[
  By Lemma 5, $F$ is sufficient for $m_k$ iff $W_k (emptyset) + sum_(i in F) h_(k,i) > 0$. This is the condition of
  the problem for the margin $k$. The gains of a model are not negative, so the data form a problem.
]

Lemma 9 puts the sufficient reason of "inside" under Theorems 4 and 5. The function `_gain_table` computes the numbers
$W_k$ and $h_(k,i)$, and the class `Minimal` or `ExactOrMinimal` does the search.

*Correspondence of symbols to the code* for this section.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Symbol*], [*Computed by*], [*Function*],
  [$(sigma, tau)$], [`Margin(sign, threshold, opposite)`], [`Threshold.side`, `Interval.side`],
  [intercept $b - tau$], [`_Bound(intercept - margin.threshold, margin.sign)`], [`compute_local_facts`],
  [the answer for a margin], [`_counterfactual(features, intercept - margin.threshold, margin.sign, ...)`], [`_smallest_counterfactual`],
  [$min(k_1, k_2)$], [`min(found, key=lambda c: len(c.changes), default=None)`], [`_smallest_counterfactual`],
  [$W_k (emptyset)$], [`bound.sign * bound.intercept + sum(scores.min() ...)`], [`_gain_table`],
  [$h_(k,i)$], [`scores[f.bin_index] - scores.min()`], [`_gain_table`],
  [first pass of `Minimal`], [`for feature in [None, *order]` until `_guaranteed(...) > 0`], [`Minimal.select`],
  [$"prune"$], [`for feature in reversed(list(chosen))`], [`Minimal.select`],
  [the exact search], [`itertools.combinations(order, size)` for each size], [`ExactOrMinimal.select`],
  [the flag `minimum`], [`minimum=len(chosen) <= 1` and `minimum=True`], [`Minimal.select`, `ExactOrMinimal.select`],
)


= Multiclass classifiers <sec-multiclass>

A multiclass EBM has one score $s_(k,i) (v)$ for each class $k$, each feature $i$ and each bin $v$, and one intercept
$b_k$ for each class. The score of the class $k$ at the instance is $S_k = b_k + sum_i s_(k,i) (c_i)$. The predicted
class $w$ has the highest score. `MulticlassLocal.frame` gives it one margin for each rival $j != w$: the sign is $1$,
the threshold is $0$, and `Margin.columns` names the pair $(w, j)$. The code reads the scores, the intercepts and the
deviations of the pair in `_Feature.against`, `Margin.offset` and `Margin.value`.

Let $m_j$ be the model of @sec-model with the intercept $b_w - b_j$, the scores $s_(w,i) - s_(j,i)$, the deviations
$sigma_(w,i) + sigma_(j,i)$, and the same plausible bins and instance bins. The deviation of a difference of two scores
has no covariance term, because an EBM stores no covariance between the scores of two classes.

#lemma("the winner", [
  (a) The logit of $m_j$ is $S_w - S_j$. (b) Every logit of $m_j$ with $j != w$ is positive iff $S_j < S_w$ for every
  rival. (c) When the logit of $m_j$ is positive, the sign of $m_j$ is $1$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Multiclass.lean", "EBaMlProofs.LocalExplanations.Multiclass.logit_marginModel (a)", "EBaMlProofs.LocalExplanations.Multiclass.winner_iff (b)", "EBaMlProofs.LocalExplanations.Multiclass.marginModel_eps (c)"))

#proof[
  (a) $(b_w - b_j) + sum_i (s_(w,i) (c_i) - s_(j,i) (c_i)) = S_w - S_j$. (b) follows from (a). (c) The sign is $1$
  iff the logit is above $0$.
]

Lemma 10 shows that the models $m_j$ are the margins of the code, and it holds only when the predicted class wins
every pair strictly. @sec-limits discusses the tie.

#theorem("the class changes", [
  Let $S_w > S_j$ for every rival $j$. Let a set $S$ _change the class_ if some admissible change of $S$ makes the
  robust margin of some $m_j$ negative. Then (a) $S$ changes the class iff it can flip the margin of $m_j$ for some
  rival $j$. Let the rival $j_0$ have the position $k$ of Theorem 2, and let every other rival either have no set that
  flips its margin, or need at least $k$ features. Then (b) the set of the first $k$ features of $j_0$, moved to the
  bins $beta$, changes the class, and no set of fewer than $k$ features changes it. (c) If no set can flip the margin
  of any rival, no set changes the class.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Multiclass.lean", "EBaMlProofs.LocalExplanations.Multiclass.canFlipClass_iff (a)", "EBaMlProofs.LocalExplanations.Multiclass.min_change_class (b)", "EBaMlProofs.LocalExplanations.Multiclass.min_change_class_none (c)"))

#proof[
  *ASSUME:* $S_w > S_j$ for every rival, and the positions of Theorem 2.
  *PROVE:* (a)–(c).

  #step(1, "1", [(a) holds. A change $(S, v)$ changes the class iff $R_j (S, v) < 0$ for some rival $j$. The plausible
    bins and the set $S$ are the same for every rival, so the change flips the margin of $m_j$ iff $S$ can flip it.])
  #step(1, "2", [(b) holds.])
  #step(2, "1", [The first $k$ features of $j_0$, moved to the bins $beta$, flip the margin of $m_(j_0)$ by Theorem 2(a).
    So they change the class by $chevron.l 1 chevron.r 1$.])
  #step(2, "2", [A set $S$ that changes the class can flip the margin of some rival $j$ by $chevron.l 1 chevron.r 1$.
    By the assumption on the rivals, either this is impossible, or $|S| >= k$.])
  #step(2, "3", [*Q.E.D.* by $chevron.l 2 chevron.r 1$ and $chevron.l 2 chevron.r 2$.])
  #step(1, "3", [(c) holds by $chevron.l 1 chevron.r 1$.])
  #step(1, "4", [*Q.E.D.* by $chevron.l 1 chevron.r 1$ to $chevron.l 1 chevron.r 3$.])
]

Each rival has its own position $k_j$ from Theorem 2, or no position. The rival with the smallest position is $j_0$, and
the other rivals have a larger position or none, so the assumption of Theorem 7 holds. The code takes the first rival
when two rivals have the same position (`min` in `_smallest_counterfactual`). The size is the same, and the named
features may differ.

The sufficient reason of the predicted class has one condition for each rival, so it is a covering problem.

#lemma("several rivals", [
  A set $F$ is sufficient for the covering problem of the rivals iff $F$ is sufficient for each model $m_j$ with
  $j != w$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Multiclass.lean", "EBaMlProofs.LocalExplanations.Multiclass.sufficient_class_iff"))

#proof[
  This is Lemma 9 for the family of the models $m_j$, indexed by the rivals.
]

Lemma 11 puts the sufficient reason of a multiclass classifier under Theorems 4 and 5, as Lemma 9 does for the
answer "inside". The function `_gain_table` computes the numbers, and the class `Minimal` or `ExactOrMinimal` does the
search. The flag `minimum` has the same meaning as in Claim 5.

#lemma("the deviation of a margin", [
  Let $sigma_x, sigma_y >= 0$ be the deviations of two scores, and let $rho in [-1, 1]$ be their correlation. The
  variance of their difference, $sigma_x^2 + sigma_y^2 - 2 rho sigma_x sigma_y$, is at most $(sigma_x + sigma_y)^2$.
], lean: ("proofs/EBaMlProofs/LocalExplanations/Multiclass.lean", "EBaMlProofs.LocalExplanations.sd_sum_bound"))

#proof[
  The difference of the two sides is $2 (1 + rho) sigma_x sigma_y >= 0$.
]

Lemma 12 shows that the sum $sigma_w + sigma_j$ is at least the deviation of the margin for every correlation. The model
of @sec-model takes the deviation of each margin as given, so the proof covers the code with this sum.

*Correspondence of symbols to the code* for this section.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Symbol*], [*Computed by*], [*Function*],
  [$w$], [`int(np.argmax(scores))`], [`MulticlassLocal.frame`],
  [margin of $m_j$], [`Margin(1, 0.0, self.classes[rival], (winner, rival))`], [`MulticlassLocal.frame`],
  [$b_w - b_j$], [`intercepts[winner] - intercepts[rival]`], [`Margin.offset`],
  [$s_(w,i) - s_(j,i)$], [`scores[:, winner] - scores[:, rival]`], [`_Feature.scores`],
  [$sigma_(w,i) + sigma_(j,i)$], [`stds[:, winner] + stds[:, rival]`], [`_Feature.stds`],
  [the answer for a rival], [`_counterfactual(... margin.offset(intercepts), margin.sign, ...)`], [`_smallest_counterfactual`],
  [the rival with the smallest position], [`min(found, key=lambda c: len(c.changes), default=None)`], [`_smallest_counterfactual`],
  [$W_j (emptyset)$, $h_(j,i)$], [`_robust_scores(f.against(bound.columns), ...)`], [`_gain_table`],
)

= Limits <sec-limits>

The proof does not cover the following points.

- *Floating point.* The proof works over the real numbers. The code uses 64-bit floating point. A rounding error
  can change the order of two nearly equal values, or the sign of a margin that is nearly zero.
- *Ties in `np.argmin`.* When two bins reach the minimum of $p_i$, `np.argmin` returns the first. The proof allows
  any bin $beta_i$ at the minimum, and the value $d_i$ does not depend on the choice. The size of the answer is the
  same, but the named bin may differ. The proof does not model which one.
- *Completions.* `_completions` samples the unknown features from the training weights. The result is an estimate,
  so there is nothing exact to prove.
- *`new_output`.* The counterfactual reports the probability of the new class from the mean scores, with no
  deviation. It is a report and not an optimality claim.
- *The docstring.* The module docstring says that every score moves by $z$ deviations in the direction that makes
  the claim harder to prove. This holds for the sufficient reason: the fixed and the unknown features both use $g$.
  For the counterfactual, only the changed features use $p$. The unchanged features keep their mean scores in
  $M$ (in `_counterfactual`). The proof covers the code. If the docstring is the intention, the counterfactual needs a
  different definition of $M$, and Theorem 2 needs a new proof.
- *The tie at the threshold.* A change of a prediction above $tau$ that lands exactly on $tau$ gives the robust margin
  $0$, so the code does not count it as a flip, although the value is then "at most $tau$". This is the same
  convention as the logit $0$ of a classifier. Every change that the code finds does flip the decision (Lemma 8), and
  "fewest" counts the changes that cross $tau$ strictly. A point mass at $tau$ has no meaning in floating point.
- *The order of `Minimal`.* The proof allows any order in both passes of `Minimal`. It proves a set from which no feature
  can be removed, and it proves nothing about how large that set is, except for at most one feature (Corollary 3).
- *The cost of `ExactOrMinimal`.* The search visits at most $2^n$ sets (Theorem 5). The code limits $n$ with
  `max_features`, and above the limit it returns the result of `Minimal` with the flag `minimum` false. The proof does
  not model this switch, but each branch has its own statement.
- *The confidence of a decision with several margins.* Each margin moves by $z$ deviations against its own claim, as in
  @sec-model. The proof covers this definition, which is the code. EBMs store no covariance between the scores of two
  margins, and the proof does not need one.
- *Ties between classes.* Lemma 10 needs $S_w > S_j$ for every rival. When two classes have the same top score, the code
  takes the first class, and its margin against the other is $0$. The proof does not cover this case, and
  `points_towards` reads the direction of a contribution wrongly there. A tie of two floating-point scores has no
  meaning.
- *The class $j$ as target.* The proof covers the counterfactual "the predicted class is no longer $w$". It does not
  cover a counterfactual that asks for a chosen class $j$, which needs every margin of $j$ at once.
- *The probability of the new class.* `Counterfactual.new_output` of a multiclass classifier is the softmax probability
  from the mean scores after the change. It is a report and not an optimality claim, as for a binary classifier.
- *The correlation between classes.* Lemma 12 shows an upper bound for the deviation of a margin. The code uses the
  bound, so the claims are conservative, and the proof does not state how much.
- *Interaction terms.* With interaction terms the score is not a sum of one term per feature, and the greedy rule
  is not minimal. Take three features $a$, $b$ and $c$ with bins in ${0, 1}$, and the instance $(0, 0, 0)$. Let the
  score be $f = 1 - 0.6 a - 0.6 b + u(b, c)$, where $u(b, c) = -2$ for $(b, c) = (0, 1)$ and $u(b, c) = 0$
  otherwise. The main effects give the drops $0.6$, $0.6$ and $0$ for $a$, $b$ and $c$, and the margin is $1$. The
  greedy rule needs $0.6 + 0.6 > 1$, so it returns two features. The change of $c$ alone to $1$ gives
  $f = -1 < 0$ and flips the prediction. The code rejects such models (`_features`).
- *The empirical link to the code.* The test `test_matches_brute_force` in
  `tests/unit/ebaml/local/test_facts.py` compares the code with a full enumeration on a small model. It checks
  the link between the code and this model on some inputs. It is not a proof.

= Computer check <sec-computer>

The Lean 4 library in `proofs/` states each lemma, theorem and corollary of @sec-abstract, @sec-application,
@sec-covering, @sec-regressor and @sec-multiclass and proves it with the Mathlib library @mathlib2020. The file
`EBaMlProofs/LocalExplanations/Greedy.lean` holds the abstract part, the file `EBaMlProofs/LocalExplanations/Model.lean`
holds the model and the two theorems, the file `EBaMlProofs/LocalExplanations/Covering.lean` holds the problem of several
margins, the file `EBaMlProofs/LocalExplanations/Regressor.lean` holds the shift and the answer "inside", and the file
`EBaMlProofs/LocalExplanations/Multiclass.lean` holds the models of the rivals. The file `EBaMlProofs/LocalExplanations/Axioms.lean` prints the
axioms of the main theorems. The proof has no `sorry`, no `admit` and no new `axiom`.

A continuous integration job runs `lake build` when the Lean files or the pins change. A reader can also run the check
by hand. @sec-appendix gives the exact commands,
the versions, the expected output and a table from each statement to its Lean name.

= References <sec-references>

#bibliography("references.yml", title: none, style: "ieee")

#pagebreak()

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix: how to repeat the check <sec-appendix>

*Versions.*
#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Item*], [*Value*],
  [Repository], [`https://github.com/johnny-godoy/EBaMl`],
  [Lean sources checked at commit], [`035d4e6cbefbd846fb4d70e08b431b9c959c356b`],
  [Lean], [`leanprover/lean4:v4.35.0` (commit `eb6cafb7290e`)],
  [Mathlib], [tag `v4.35.0` (commit `56a005855ea17e0b0958f73ec801ae140b0a1679`)],
  [Typst], [`0.15.1`],
)

The files in `proofs/EBaMlProofs/`, `proofs/lakefile.toml`, `proofs/lean-toolchain` and `proofs/lake-manifest.json`
are the same at the commit above and at the commit of this paper. The functions of @sec-model are in
`src/ebaml/local/facts.py`. Read the paper together with the version of that file at the commit of this paper.

*Commands from a clean clone.*

```sh
# install elan, then restart the shell
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/johnny-godoy/EBaMl && cd EBaMl
git checkout 035d4e6cbefbd846fb4d70e08b431b9c959c356b
cd proofs
lake exe cache get        # download the compiled Mathlib
lake build                # check every proof
typst compile --root papers papers/local-explanations/paper.typ
```

If `lake exe cache get` cannot reach its server, `lake build` compiles the imported part of Mathlib from source.
This takes tens of minutes, and the result is the same.

*Expected output of `lake build`.* The build ends with `Build completed successfully` and no line that starts
with `error`. The file `Axioms.lean` prints these lines, in this order:

```text
info: EBaMlProofs/LocalExplanations/Axioms.lean:12:0: 'EBaMlProofs.LocalExplanations.Greedy.threshold_minimal' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:13:0: 'EBaMlProofs.LocalExplanations.Model.min_change' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:14:0: 'EBaMlProofs.LocalExplanations.Model.min_change_none' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:15:0: 'EBaMlProofs.LocalExplanations.Model.chosen_bin_differs' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:16:0: 'EBaMlProofs.LocalExplanations.Model.min_sufficient_reason' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:17:0: 'EBaMlProofs.LocalExplanations.Model.min_sufficient_reason_none' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:18:0: 'EBaMlProofs.LocalExplanations.Model.shift_eps' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:19:0: 'EBaMlProofs.LocalExplanations.Model.shift_margin_eq' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:20:0: 'EBaMlProofs.LocalExplanations.Model.shift_robustMargin_neg_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:21:0: 'EBaMlProofs.LocalExplanations.Model.canFlipInside_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:22:0: 'EBaMlProofs.LocalExplanations.Model.min_change_inside' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:23:0: 'EBaMlProofs.LocalExplanations.Model.min_change_inside_none' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:24:0: 'EBaMlProofs.LocalExplanations.problemOfModels_sufficient' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:25:0: 'EBaMlProofs.LocalExplanations.Covering.Problem.sufficient_mono' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:26:0: 'EBaMlProofs.LocalExplanations.Covering.Problem.prune_minimal' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:27:0: 'EBaMlProofs.LocalExplanations.Covering.Problem.exact_minimum' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:28:0: 'EBaMlProofs.LocalExplanations.Covering.Problem.card_search_space' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:29:0: 'EBaMlProofs.LocalExplanations.Covering.Problem.minimum_of_card_le_one' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:30:0: 'EBaMlProofs.LocalExplanations.Multiclass.logit_marginModel' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:31:0: 'EBaMlProofs.LocalExplanations.Multiclass.winner_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:32:0: 'EBaMlProofs.LocalExplanations.Multiclass.marginModel_eps' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:33:0: 'EBaMlProofs.LocalExplanations.Multiclass.canFlipClass_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:34:0: 'EBaMlProofs.LocalExplanations.Multiclass.min_change_class' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:35:0: 'EBaMlProofs.LocalExplanations.Multiclass.min_change_class_none' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:36:0: 'EBaMlProofs.LocalExplanations.Multiclass.sufficient_class_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
info: EBaMlProofs/LocalExplanations/Axioms.lean:37:0: 'EBaMlProofs.LocalExplanations.sd_sum_bound' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The build may print warnings that Mathlib modules are "designed for use with the module system". These warnings
are expected, and they do not affect the result. The output of each `#print axioms` shows only `propext`, `Classical.choice` and `Quot.sound`. The line breaks of long lines may differ. The word `sorryAx`
must not appear. To search for it, run `lake build 2>&1 | grep -c sorry`. The expected result is `0`.

The last command writes `papers/local-explanations/paper.pdf` and prints nothing.

*Statements and Lean names.* The directory is `proofs/EBaMlProofs/LocalExplanations/`. In the table, each prefix
stands for a Lean namespace:

- `Greedy.` stands for `EBaMlProofs.LocalExplanations.Greedy.`
- `Model.` stands for `EBaMlProofs.LocalExplanations.Model.`
- `Covering.` stands for `EBaMlProofs.LocalExplanations.Covering.Problem.`
- `MC.` stands for `EBaMlProofs.LocalExplanations.Multiclass.`
- `LE.` stands for `EBaMlProofs.LocalExplanations.`

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(160),
  inset: 5pt,
  [*Statement*], [*Lean name*], [*File*],
  [Definition 1 (top set)], [`Greedy.IsTopSet`], [`Greedy.lean`],
  [Lemma 1 (top-set bound)], [`Greedy.topSet_sum_le`], [`Greedy.lean`],
  [Lemma 2 (prefixes)], [`Greedy.isTopSet_prefixSet`, `Greedy.card_prefixSet`, `Greedy.sum_prefixSet`], [`Greedy.lean`],
  [Theorem 1 (greedy threshold)], [`Greedy.threshold_minimal`], [`Greedy.lean`],
  [Corollary 1 (no answer)], [`Greedy.no_answer`], [`Greedy.lean`],
  [Corollary 2 (positive values)], [`Greedy.pos_of_threshold`], [`Greedy.lean`],
  [Lemma 3 (margin)], [`Model.margin_nonneg`], [`Model.lean`],
  [Lemma 4 (best change)], [`Model.robustMargin_eq`, `Model.robustMargin_ge`, `Model.robustMargin_beta`,
    `Model.canFlip_iff`], [`Model.lean`],
  [Lemma 5 (gain)], [`Model.worst_eq`, `Model.sufficient_iff`], [`Model.lean`],
  [Theorem 2 (a, b)], [`Model.min_change`], [`Model.lean`],
  [Theorem 2 (c)], [`Model.chosen_bin_differs`], [`Model.lean`],
  [Theorem 2 (d)], [`Model.min_change_none`], [`Model.lean`],
  [Theorem 3 (a, b)], [`Model.min_sufficient_reason`], [`Model.lean`],
  [Theorem 3 (c)], [`Model.min_sufficient_reason_none`], [`Model.lean`],
  [Lemma 6 (monotone)], [`Covering.sufficient_mono`], [`Covering.lean`],
  [Theorem 4 (removal pass)], [`Covering.prune_minimal`, `Covering.prune_invariant`], [`Covering.lean`],
  [Theorem 5 (exact search)], [`Covering.exact_minimum`, `Covering.card_search_space`], [`Covering.lean`],
  [Corollary 3 (at most one feature)], [`Covering.minimum_of_card_le_one`], [`Covering.lean`],
  [Lemma 7 (shift)], [`Model.shift_logit`, `Model.shift_eps`, `Model.shift_margin_eq`], [`Regressor.lean`],
  [Lemma 8 (crossing)], [`Model.shift_robustMargin_neg_iff`], [`Regressor.lean`],
  [Theorem 6 (a)], [`Model.inside_eps`], [`Regressor.lean`],
  [Theorem 6 (b)], [`Model.canFlipInside_iff`], [`Regressor.lean`],
  [Theorem 6 (c)], [`Model.min_change_inside`], [`Regressor.lean`],
  [Theorem 6 (d)], [`Model.min_change_inside_none`], [`Regressor.lean`],
  [Lemma 9 (several margins)], [`LE.problemOfModels_sufficient`], [`Regressor.lean`],
  [Lemma 10 (the winner)], [`MC.logit_marginModel`, `MC.winner_iff`, `MC.marginModel_eps`], [`Multiclass.lean`],
  [Theorem 7 (a)], [`MC.canFlipClass_iff`], [`Multiclass.lean`],
  [Theorem 7 (b)], [`MC.min_change_class`], [`Multiclass.lean`],
  [Theorem 7 (c)], [`MC.min_change_class_none`], [`Multiclass.lean`],
  [Lemma 11 (several rivals)], [`MC.sufficient_class_iff`], [`Multiclass.lean`],
  [Lemma 12 (deviation)], [`LE.sd_sum_bound`], [`Multiclass.lean`],
)
