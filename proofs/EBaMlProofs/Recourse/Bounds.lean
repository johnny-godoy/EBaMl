import EBaMlProofs.Recourse.Plans
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Tactic.Ring

/-!
# The lower bounds of the solvers

This file checks Lemma 17 of the local-explanations paper, which that paper proves on paper only, and the cost bound
of a plan that `_cost_lower_bound` reports.

* `lp_lower_bound` is Lemma 17: each valid plan changes at least `⌈L⌉` groups, where `L` bounds the linear relaxation
  from below. `no_plan_of_relaxation_infeasible` is its case of an infeasible relaxation.
* `card_mul_le_cost` and `cost_lower_bound` give the bound of `_cost_lower_bound`: a number of changes that every
  valid plan needs, times the cheapest cost, is at most the cost of every valid plan. `one_le_card` gives the number
  `1` of `make_weighted_cover`.
-/

open Finset

namespace EBaMlProofs.Recourse

variable {ι : Type} [Fintype ι] [DecidableEq ι] {α : ι → Type} [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)]
  {κ : Type}

namespace Problem

variable (P : Problem ι α κ)

/-! ### Lemma 17: the lower bound of the linear relaxation -/

/-- A feasible point of the linear relaxation of `build_model`: each variable in `[0, 1]`, at most `1` in each group,
and each margin at least `0`. -/
def RelaxedFeasible (x : ∀ i, α i → ℝ) : Prop :=
  (∀ i a, 0 ≤ x i a ∧ x i a ≤ 1) ∧ (∀ i, ∑ a, x i a ≤ 1) ∧ ∀ k, -P.worst k ≤ ∑ i, ∑ a, P.gain i a k * x i a

/-- The point of the relaxation of a plan: `1` for each pick, `0` otherwise. -/
noncomputable def relaxOf (p : Plan ι α) : ∀ i, α i → ℝ := fun i a => if ofPlan p i a then 1 else 0

omit [Fintype ι] [DecidableEq ι] in
theorem sum_relaxOf_group (p : Plan ι α) (i : ι) : ∑ a, relaxOf p i a = if (p i).isSome then 1 else 0 := by
  have := sum_ofPlan p i fun _ => (1 : ℝ)
  simp only [relaxOf] at this ⊢
  rw [this]
  cases p i <;> simp

omit [DecidableEq ι] in
/-- The sum of the point of a plan is its number of changes. -/
theorem sum_relaxOf (p : Plan ι α) : ∑ i, ∑ a, relaxOf p i a = ((changed p).card : ℝ) := by
  simp only [sum_relaxOf_group, changed]
  rw [Finset.sum_boole]

omit [DecidableEq ι] in
/-- The point of a valid plan is feasible for the relaxation, whatever the cap and the exclusions. -/
theorem relaxOf_feasible {L : Limits ι α} {p : Plan ι α} (hp : P.Valid L p) : P.RelaxedFeasible (relaxOf p) := by
  refine ⟨fun i a => ?_, fun i => ?_, fun k => ?_⟩
  · simp only [relaxOf]
    split_ifs <;> norm_num
  · rw [sum_relaxOf_group]
    split_ifs <;> norm_num
  · have hm := hp.1 k
    have hsum : ∑ i, ∑ a, P.gain i a k * relaxOf p i a = ∑ i, P.pickGain p i k := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [pickGain, ← sum_ofPlan p i fun a => P.gain i a k]
      refine Finset.sum_congr rfl fun a _ => ?_
      simp only [relaxOf]
      split_ifs <;> ring
    rw [hsum]
    simp only [margin] at hm
    linarith

omit [DecidableEq ι] in
/-- Lemma 17. If `bound` is at most the sum of every feasible point of the relaxation, each valid plan changes at
least `⌈bound⌉` groups, whatever the cap and the exclusions. -/
theorem lp_lower_bound {L : Limits ι α} {bound : ℝ}
    (hbound : ∀ x, P.RelaxedFeasible x → bound ≤ ∑ i, ∑ a, x i a) {p : Plan ι α} (hp : P.Valid L p) :
    ⌈bound⌉ ≤ ((changed p).card : ℤ) := by
  have h := hbound _ (P.relaxOf_feasible hp)
  rw [sum_relaxOf] at h
  exact Int.ceil_le.2 (by exact_mod_cast h)

omit [DecidableEq ι] in
/-- Lemma 17, the infeasible case: if the relaxation has no feasible point, no plan is valid. -/
theorem no_plan_of_relaxation_infeasible (h : ∀ x, ¬ P.RelaxedFeasible x) (L : Limits ι α) (p : Plan ι α) :
    ¬ P.Valid L p := fun hp => h _ (P.relaxOf_feasible hp)

/-! ### The cost bound of `_cost_lower_bound` -/

omit [DecidableEq ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- A plan costs at least its number of changes times the cheapest cost of an alternative. -/
theorem card_mul_le_cost {cheapest : ℝ} (hcheap : ∀ i a, cheapest ≤ P.cost i a) (p : Plan ι α) :
    ((changed p).card : ℝ) * cheapest ≤ P.cost' p := by
  have hterm : ∀ i, (if (p i).isSome then cheapest else 0) ≤ P.pickCost p i := by
    intro i
    cases hpi : p i with
    | none => simp [pickCost, hpi]
    | some a => simpa [pickCost, hpi] using hcheap i a
  calc ((changed p).card : ℝ) * cheapest = ∑ i, (if (p i).isSome then cheapest else 0) := by
        rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
        rfl
    _ ≤ P.cost' p := Finset.sum_le_sum fun i _ => hterm i

omit [DecidableEq ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- The bound of `_cost_lower_bound`: if every valid plan changes at least `bound` groups, and `cheapest` is at most
every cost and not negative, then `bound * cheapest` is at most the cost of every valid plan. -/
theorem cost_lower_bound {L : Limits ι α} {bound : ℕ} (hbound : ∀ q, P.Valid L q → bound ≤ (changed q).card)
    {cheapest : ℝ} (hnonneg : 0 ≤ cheapest) (hcheap : ∀ i a, cheapest ≤ P.cost i a) {p : Plan ι α}
    (hp : P.Valid L p) : (bound : ℝ) * cheapest ≤ P.cost' p :=
  calc (bound : ℝ) * cheapest ≤ ((changed p).card : ℝ) * cheapest :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast hbound p hp) hnonneg
    _ ≤ P.cost' p := P.card_mul_le_cost hcheap p

omit [DecidableEq ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- The bound `1` of `make_weighted_cover`: if the plan with no change is not valid, each valid plan changes at least
one group. -/
theorem one_le_card {L : Limits ι α} (hempty : ¬ ∀ k, 0 < P.worst k) {p : Plan ι α} (hp : P.Valid L p) :
    1 ≤ (changed p).card := by
  by_contra h
  have hzero : changed p = ∅ := Finset.card_eq_zero.1 (by omega)
  have hnone : ∀ i, p i = none := by
    intro i
    have : i ∉ changed p := by simp [hzero]
    simpa [changed] using this
  refine hempty fun k => ?_
  have := hp.1 k
  simpa [margin, pickGain, hnone] using this

end Problem

end EBaMlProofs.Recourse
