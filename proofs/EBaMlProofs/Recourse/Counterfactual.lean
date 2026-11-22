import EBaMlProofs.Recourse.Plans
import EBaMlProofs.LocalExplanations.Regressor

/-!
# The counterfactual as a plan problem with costs

This file applies `Plans.lean` to the binary model of `EBaMlProofs.LocalExplanations.Model`, with the allowed bins
`A` of the recourse options. Each feature is a group, and each allowed bin other than the bin of the instance is an
alternative, as `_alternatives` builds them. The margin of a plan is minus the robust margin of its change.

* `cfProblem` is the problem of the counterfactual with one margin (Section 4 of the paper).
* `margin_cfProblem` and `valid_cfProblem_iff` are Lemma 1: a plan is valid iff its change flips the prediction.
* `sorting_cheapest` is Theorem 4: with unit costs, the sorting search returns a cheapest plan. This keeps the status
  `SORTING_PROOF`.
* `sorting_not_cheapest` is Proposition 1: with costs, the bin with the largest drop is not always in a cheapest
  plan, so the code sends a weighted problem to the MIP.
* `knapsack_canFlipOn_iff` is Theorem 5: every instance of the knapsack problem is the counterfactual of a binary EBM,
  so the weighted problem with one margin is NP-hard (Karp's theorem is a citation).
-/

open Finset EBaMlProofs.LocalExplanations EBaMlProofs.LocalExplanations.Greedy

namespace EBaMlProofs.Recourse

variable {ι : Type} [Fintype ι] [DecidableEq ι] {B : ι → Type} [∀ i, DecidableEq (B i)]

/-- The alternatives of feature `i`: its allowed bins other than the bin of the instance. -/
abbrev Alt (m : Model ι B) (A : ∀ i, Finset (B i)) (i : ι) : Type := ↥((A i).filter fun v => v ≠ m.c i)

/-- The problem of the counterfactual with one margin: the worst case `-M`, and the gain `t_i(c_i) - p_i(v)` of each
alternative `v`, which is how much the change to `v` lowers the robust margin. -/
noncomputable def cfProblem (m : Model ι B) (A : ∀ i, Finset (B i)) (cost : ∀ i, Alt m A i → ℝ)
    (hcost : ∀ i a, 0 < cost i a) : Problem ι (Alt m A) Unit where
  worst _ := -m.margin
  gain i v _ := m.t i (m.c i) - m.p i v.1
  cost := cost
  cost_pos := hcost

/-- No cap and no exclusion. -/
def noLimits : Limits ι α := ⟨none, fun _ _ => False⟩

variable (m : Model ι B) (A : ∀ i, Finset (B i))

/-- The bins of a plan: the picked bin of each changed feature, and the bin of the instance otherwise. -/
def planBins (p : Plan ι (Alt m A)) : ∀ i, B i := fun i => ((p i).map Subtype.val).getD (m.c i)

/-- Lemma 1, the margin: the margin of a plan is minus the robust margin of its change. -/
theorem margin_cfProblem (cost : ∀ i, Alt m A i → ℝ) (hcost : ∀ i a, 0 < cost i a) (p : Plan ι (Alt m A)) :
    (cfProblem m A cost hcost).margin p () = -m.robustMargin (Problem.changed p) (planBins m A p) := by
  have this : ∀ i, (cfProblem m A cost hcost).pickGain p i () =
      if (p i).isSome then m.t i (m.c i) - m.p i (planBins m A p i) else 0 := by
    intro i
    cases h : p i <;> simp [Problem.pickGain, cfProblem, planBins, h]
  have hsum : ∑ i, (cfProblem m A cost hcost).pickGain p i () =
      ∑ i ∈ Problem.changed p, (m.t i (m.c i) - m.p i (planBins m A p i)) := by
    rw [Problem.changed, Finset.sum_filter]
    exact Finset.sum_congr rfl fun i _ => this i
  rw [m.robustMargin_eq, ← hsum]
  show -m.margin + _ = _
  ring

/-- Lemma 1: with no cap and no exclusion, a plan is valid iff its change flips the prediction. -/
theorem valid_cfProblem_iff (cost : ∀ i, Alt m A i → ℝ) (hcost : ∀ i a, 0 < cost i a) (p : Plan ι (Alt m A)) :
    (cfProblem m A cost hcost).Valid noLimits p ↔ m.robustMargin (Problem.changed p) (planBins m A p) < 0 := by
  simp only [Problem.Valid, noLimits, reduceCtorEq, false_implies, implies_true, not_false_eq_true, and_true]
  constructor
  · intro h
    have := h ()
    rw [margin_cfProblem] at this
    linarith
  · intro h k
    cases k
    rw [margin_cfProblem]
    linarith

omit [DecidableEq ι] in
/-- The bins of a plan are admissible over `A` for the features that it changes. -/
theorem admissibleOn_planBins (p : Plan ι (Alt m A)) : m.AdmissibleOn A (Problem.changed p) (planBins m A p) := by
  intro i hi
  simp only [Problem.changed, Finset.mem_filter, Finset.mem_univ, true_and] at hi
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hi
  have hvA : v.1 ∈ A i := (Finset.mem_filter.1 v.2).1
  simp only [planBins, hv, Option.map_some, Option.getD_some, Model.bins]
  exact Finset.mem_insert_of_mem hvA

omit [DecidableEq ι] in
/-- With unit costs, the cost of a plan is its number of changes. -/
theorem cost_unit (p : Plan ι (Alt m A)) :
    (cfProblem m A (fun _ _ => 1) fun _ _ => one_pos).cost' p = (Problem.changed p).card := by
  simp only [Problem.cost', Problem.pickCost, cfProblem, Problem.changed]
  rw [Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  cases p i <;> simp

/-- Theorem 4 (unit costs). Let `e` sort the drops over `A`, and let `k` be the first position where the sum of the
first `k` drops passes the margin. Then a valid plan with `k` changes exists, and each valid plan costs at least `k`.
So the sorting search of `_counterfactual` returns a cheapest plan. -/
theorem sorting_cheapest {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.dropOn A ∘ e)) {k : ℕ} (hk : k ≤ n)
    (hflip : m.margin < partialSum (m.dropOn A) e k)
    (hbefore : ∀ j < k, partialSum (m.dropOn A) e j ≤ m.margin) :
    (∃ p, (cfProblem m A (fun _ _ => 1) fun _ _ => one_pos).Valid noLimits p ∧ Problem.changed p = prefixSet e hk) ∧
      ∀ q, (cfProblem m A (fun _ _ => 1) fun _ _ => one_pos).Valid noLimits q →
        (k : ℝ) ≤ (cfProblem m A (fun _ _ => 1) fun _ _ => one_pos).cost' q := by
  obtain ⟨w, hw⟩ := m.exists_isLowest A
  obtain ⟨⟨_, hR⟩, hmin⟩ := m.min_changeOn A he hk hflip (fun j _ hj => hbefore j hj) hw
  have hmem : ∀ i ∈ prefixSet e hk, w i ∈ (A i).filter fun v => v ≠ m.c i := by
    intro i hi
    have hj : ((e.symm i : Fin n) : ℕ) < k := (mem_prefixSet e hk i).1 hi
    have h := m.chosen_bin_differsOn A he hk hflip hbefore hw (e.symm i) hj
    rw [Equiv.apply_symm_apply] at h
    exact Finset.mem_filter.2 ⟨h.2, h.1⟩
  let p : Plan ι (Alt m A) := fun i => if h : i ∈ prefixSet e hk then some ⟨w i, hmem i h⟩ else none
  have hchanged : Problem.changed p = prefixSet e hk := by
    ext i
    by_cases h : i ∈ prefixSet e hk <;> simp [Problem.changed, p, h]
  have hbins : ∀ i ∈ prefixSet e hk, planBins m A p i = w i := by
    intro i hi
    simp [planBins, p, hi]
  refine ⟨⟨p, ?_, hchanged⟩, fun q hq => ?_⟩
  · rw [valid_cfProblem_iff, hchanged]
    have : m.robustMargin (prefixSet e hk) (planBins m A p) = m.robustMargin (prefixSet e hk) w := by
      simp only [Model.robustMargin]
      congr 1
      refine Finset.sum_congr rfl fun i _ => ?_
      split_ifs with hi
      · rw [hbins i hi]
      · rfl
    rw [this]
    exact hR
  · rw [cost_unit]
    have hflipq : m.CanFlipOn A (Problem.changed q) :=
      ⟨planBins m A q, admissibleOn_planBins m A q, (valid_cfProblem_iff m A _ _ q).1 hq⟩
    exact_mod_cast hmin _ hflipq

/-! ### Monotonicity in the confidence level and in the allowed bins (contract 7) -/

/-- The same model with another number of deviations `z'`. -/
def withZ (z' : ℝ) (hz : 0 ≤ z') : Model ι B := { m with z := z', z_nonneg := hz }

omit [DecidableEq ι] in
/-- Theorem 6 (a): a higher confidence level never makes a plan valid. A valid plan at `z'` is valid at `z`, with the
same cost, so the cheapest cost never goes down and an impossible plan never becomes possible. -/
theorem valid_of_higher_confidence {z' : ℝ} (hz : 0 ≤ z') (hle : m.z ≤ z') (cost : ∀ i, Alt m A i → ℝ)
    (hcost : ∀ i a, 0 < cost i a) {L : Limits ι (Alt m A)} {p : Plan ι (Alt m A)}
    (h : (cfProblem (withZ m z' hz) A cost hcost).Valid L p) : (cfProblem m A cost hcost).Valid L p := by
  refine Problem.valid_of_gain_le (cfProblem m A cost hcost) (P' := cfProblem (withZ m z' hz) A cost hcost) rfl
    (fun i a _ => ?_) h
  have hsd := m.sd_nonneg i a.1
  have : m.z * m.sd i a.1 ≤ z' * m.sd i a.1 := mul_le_mul_of_nonneg_right hle hsd
  show m.t i (m.c i) - (m.t i a.1 + z' * m.sd i a.1) ≤ m.t i (m.c i) - (m.t i a.1 + m.z * m.sd i a.1)
  linarith

/-- An alternative over a smaller allowed set is an alternative over a larger one. -/
def inclAlt {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (i : ι) (v : Alt m A' i) : Alt m A i :=
  ⟨v.1, by
    have hv := Finset.mem_filter.1 v.2
    exact Finset.mem_filter.2 ⟨hA i hv.1, hv.2⟩⟩

/-- A plan over a smaller allowed set is a plan over a larger one. -/
def embed {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (p : Plan ι (Alt m A')) : Plan ι (Alt m A) :=
  fun i => (p i).map (inclAlt m A hA i)

omit [DecidableEq ι] in
theorem changed_embed {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (p : Plan ι (Alt m A')) :
    Problem.changed (embed m A hA p) = Problem.changed p := by
  ext i
  simp [Problem.changed, embed]

omit [Fintype ι] [DecidableEq ι] in
theorem planBins_embed {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (p : Plan ι (Alt m A')) :
    planBins m A (embed m A hA p) = planBins m A' p := by
  funext i
  cases hp : p i <;> simp [planBins, embed, hp, inclAlt]

/-- Theorem 6 (b): a smaller allowed set, for example one more immutable feature, never gives a cheaper plan. A valid
plan over `A'` is a valid plan over `A`, and it has the same cost when the costs over `A'` are those over `A`. -/
theorem valid_embed {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (cost : ∀ i, Alt m A i → ℝ)
    (hcost : ∀ i a, 0 < cost i a) (cost' : ∀ i, Alt m A' i → ℝ) (hcost' : ∀ i a, 0 < cost' i a)
    (p : Plan ι (Alt m A')) (h : (cfProblem m A' cost' hcost').Valid noLimits p) :
    (cfProblem m A cost hcost).Valid noLimits (embed m A hA p) := by
  rw [valid_cfProblem_iff] at h ⊢
  rw [changed_embed, planBins_embed]
  exact h

omit [DecidableEq ι] in
theorem cost_embed {A' : ∀ i, Finset (B i)} (hA : ∀ i, A' i ⊆ A i) (cost : ∀ i, Alt m A i → ℝ)
    (hcost : ∀ i a, 0 < cost i a) (cost' : ∀ i, Alt m A' i → ℝ) (hcost' : ∀ i a, 0 < cost' i a)
    (hc : ∀ i v, cost' i v = cost i (inclAlt m A hA i v)) (p : Plan ι (Alt m A')) :
    (cfProblem m A cost hcost).cost' (embed m A hA p) = (cfProblem m A' cost' hcost').cost' p := by
  simp only [Problem.cost', Problem.pickCost, cfProblem]
  refine Finset.sum_congr rfl fun i _ => ?_
  cases hp : p i <;> simp [embed, hp, hc]

/-! ### Why the weighted search does not sort -/

/-- Proposition 1. One feature with two alternatives: the first lowers the robust margin by `10` and costs `5`, the
second lowers it by `2` and costs `1`, and the margin is `1`. The first has the largest drop, but no cheapest plan
picks it, because the second alone is a valid plan of cost `1`. -/
noncomputable def twoBins : Problem Unit (fun _ => Bool) Unit where
  worst _ := -1
  gain _ v _ := if v then 10 else 2
  cost _ v := if v then 5 else 1
  cost_pos _ v := by cases v <;> norm_num

theorem sorting_not_cheapest (p : Plan Unit fun _ => Bool) (hp : twoBins.Cheapest noLimits p) :
    p () ≠ some true := by
  intro htrue
  have hq : twoBins.Valid noLimits (fun _ => some false) := by
    refine ⟨fun _ => ?_, by simp [noLimits], by simp [noLimits]⟩
    simp [Problem.margin, Problem.pickGain, twoBins]
  have hle := hp.2 _ hq
  simp [Problem.cost', Problem.pickCost, twoBins, htrue] at hle

/-! ### NP-hardness: the knapsack problem is a weighted counterfactual -/

/-- The binary model of a knapsack instance with values `v_i > 0` and the bound `V > 0`: feature `i` has the bins
`true` (the instance, score `v_i`) and `false` (score `0`), the margin is `V`, and no score moves. -/
noncomputable def knapsackModel (v : ι → ℝ) (V : ℝ) : Model ι fun _ => Bool where
  b := V - ∑ i, v i
  z := 0
  s i x := if x then v i else 0
  sd _ _ := 0
  P _ := univ
  c _ := true
  z_nonneg := le_rfl
  sd_nonneg _ _ := le_rfl
  c_mem _ := mem_univ _

/-- Theorem 5. In the model of a knapsack instance, a set of features can flip the prediction iff the values of its
features sum to more than `V`. With the cost `w_i` of the change of feature `i`, a plan of cost at most `W` exists iff
a set with value above `V` and weight at most `W` exists. This decision problem is NP-complete (Karp), so the
weighted counterfactual with one margin is NP-hard. -/
theorem knapsack_canFlipOn_iff (v : ι → ℝ) (hv : ∀ i, 0 < v i) {V : ℝ} (hV : 0 < V) (S : Finset ι) :
    (knapsackModel v V).CanFlipOn (fun _ => univ) S ↔ V < ∑ i ∈ S, v i := by
  have hlogit : (knapsackModel v V).logit = V := by
    simp [Model.logit, knapsackModel]
  have heps : (knapsackModel v V).eps = 1 := by simp [Model.eps, hlogit, hV]
  have hmargin : (knapsackModel v V).margin = V := by
    rw [Model.margin_eq, heps, hlogit, one_mul]
  have hdrop : ∀ i, (knapsackModel v V).dropOn (fun _ => univ) i = v i := by
    intro i
    have hmin : (knapsackModel v V).amin (fun _ => univ) (knapsackModel v V).p i = 0 := by
      refine le_antisymm ?_ (Finset.le_inf' _ _ fun x _ => ?_)
      · have := (knapsackModel v V).amin_le (fun _ => univ) (knapsackModel v V).p (i := i) (v := false)
          (by simp [Model.bins])
        simp only [Model.p, Model.t, heps] at this
        simpa [knapsackModel] using this
      · simp only [Model.p, Model.t, heps]
        cases x <;> simp [knapsackModel, (hv i).le]
    rw [Model.dropOn, hmin, Model.t, heps]
    simp [knapsackModel]
  rw [Model.canFlipOn_iff, hmargin]
  simp only [hdrop]

end EBaMlProofs.Recourse
