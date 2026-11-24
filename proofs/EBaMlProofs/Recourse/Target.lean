import EBaMlProofs.Recourse.Plans
import EBaMlProofs.LocalExplanations.Multiclass

/-!
# The counterfactual of a chosen class

This file models `_target_counterfactual` of `counterfactuals.py`: the changes after which a chosen class `τ` of a
multiclass classifier beats every other class under the robust scores. Each feature is a group, each allowed bin other
than the bin of the instance is an alternative (`_alternatives`), and each rival class is a margin
(`MulticlassLocal.target_margins`). The covering problem is a `Recourse.Problem`, so Theorem 1 of the recourse paper
(the integer program) applies to it with costs, a cap and exclusions.

* `rivalMargin` is the robust margin of `τ` against a rival after a change: the changed features take the score of the
  new bin moved by `z` deviations against `τ`, and the other features keep their scores.
* `margin_targetProblem` and `valid_targetProblem_iff` are Lemma 3: a plan is valid iff its change makes every
  margin of `τ` positive.
* `plan_of_reaching` and `exists_plan_iff` are Theorem 7: every change that reaches `τ` gives a valid plan with at most
  as many changes, so the valid plans are exactly the changes that reach `τ`, and the smallest plan is the fewest
  changes.
-/

open Finset EBaMlProofs.LocalExplanations

namespace EBaMlProofs.Recourse

variable {ι : Type} [Fintype ι] [DecidableEq ι] {B : ι → Type} [∀ i, DecidableEq (B i)] {κ : Type}

namespace Multiclass

variable (M : LocalExplanations.Multiclass ι B κ) (A : ∀ i, Finset (B i)) (τ : κ)

/-- The alternatives of feature `i`: its allowed bins other than the bin of the instance. -/
abbrev TargetAlt (i : ι) : Type := ↥((A i).filter fun v => v ≠ M.c i)

/-- The term of feature `i` in the margin of `τ` against `j` when it takes the bin `v`, moved by `z` deviations
against `τ`, as `view.scores - z * view.stds` in `_alternatives`. -/
noncomputable def movedTerm (j : κ) (i : ι) (v : B i) : ℝ :=
  M.s τ i v - M.s j i v - M.z * (M.sd τ i v + M.sd j i v)

/-- The term of feature `i` in the margin of `τ` against `j` at the bin of the instance (`view.current`). -/
noncomputable def keptTerm (j : κ) (i : ι) : ℝ := M.s τ i (M.c i) - M.s j i (M.c i)

/-- The robust margin of `τ` against the rival `j` after the change of the features in `S` to the bins `w`. -/
noncomputable def rivalMargin (j : κ) (S : Finset ι) (w : ∀ i, B i) : ℝ :=
  M.b τ - M.b j + ∑ i, if i ∈ S then movedTerm M τ j i (w i) else keptTerm M τ j i

/-- A change reaches `τ` when each changed feature ends in an allowed bin or the bin of the instance, and every margin
of `τ` is positive. -/
def Reaches (S : Finset ι) (w : ∀ i, B i) : Prop :=
  (∀ i ∈ S, w i ∈ insert (M.c i) (A i)) ∧ ∀ j, j ≠ τ → 0 < rivalMargin M τ j S w

/-- The covering problem of `_target_counterfactual`: the worst case of each margin is its value at the instance
(`margin.offset + sum(view.current)`), and the gain of a bin is how much it raises the margin. -/
noncomputable def targetProblem (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a) :
    Problem ι (TargetAlt M A) {j // j ≠ τ} where
  worst j := M.b τ - M.b j.1 + ∑ i, keptTerm M τ j.1 i
  gain i v j := movedTerm M τ j.1 i v.1 - keptTerm M τ j.1 i
  cost := cost
  cost_pos := hcost

/-- The bins of a plan: the picked bin of each changed feature, and the bin of the instance otherwise. -/
def targetBins (p : Plan ι (TargetAlt M A)) : ∀ i, B i := fun i => ((p i).map Subtype.val).getD (M.c i)

/-- Lemma 3, the margin: the margin of a plan for the rival `j` is the robust margin of `τ` against `j`. -/
theorem margin_targetProblem (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a)
    (p : Plan ι (TargetAlt M A)) (j : {j // j ≠ τ}) :
    (targetProblem M A τ cost hcost).margin p j =
      rivalMargin M τ j.1 (Problem.changed p) (targetBins M A p) := by
  have hterm : ∀ i, (targetProblem M A τ cost hcost).pickGain p i j + keptTerm M τ j.1 i =
      if i ∈ Problem.changed p then movedTerm M τ j.1 i (targetBins M A p i) else keptTerm M τ j.1 i := by
    intro i
    cases h : p i <;> simp [Problem.pickGain, targetProblem, targetBins, Problem.changed, h]
  simp only [Problem.margin, rivalMargin, targetProblem]
  rw [← Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib]
  simp only [targetProblem]
  ring

/-- Lemma 3: with no cap and no exclusion, a plan is valid iff its change makes every margin of `τ` positive. -/
theorem valid_targetProblem_iff (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a)
    (p : Plan ι (TargetAlt M A)) :
    (targetProblem M A τ cost hcost).Valid ⟨none, fun _ _ => False⟩ p ↔
      ∀ j, j ≠ τ → 0 < rivalMargin M τ j (Problem.changed p) (targetBins M A p) := by
  simp only [Problem.Valid, reduceCtorEq, false_implies, implies_true, not_false_eq_true, and_true]
  constructor
  · intro h j hj
    have := h ⟨j, hj⟩
    rwa [margin_targetProblem] at this
  · intro h j
    rw [margin_targetProblem]
    exact h j.1 j.2

omit [DecidableEq ι] in
theorem targetBins_mem (p : Plan ι (TargetAlt M A)) (i : ι) (hi : i ∈ Problem.changed p) :
    targetBins M A p i ∈ insert (M.c i) (A i) := by
  simp only [Problem.changed, Finset.mem_filter, Finset.mem_univ, true_and] at hi
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hi
  simp only [targetBins, hv, Option.map_some, Option.getD_some]
  exact Finset.mem_insert_of_mem (Finset.mem_filter.1 v.2).1

/-- The change of a valid plan reaches `τ`. -/
theorem reaches_of_valid (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a)
    {p : Plan ι (TargetAlt M A)} (h : (targetProblem M A τ cost hcost).Valid ⟨none, fun _ _ => False⟩ p) :
    Reaches M A τ (Problem.changed p) (targetBins M A p) :=
  ⟨targetBins_mem M A p, (valid_targetProblem_iff M A τ cost hcost p).1 h⟩

/-- The plan of a change: each changed feature whose bin differs from the bin of the instance picks that bin. -/
noncomputable def planOf (S : Finset ι) (w : ∀ i, B i) (hw : ∀ i ∈ S, w i ∈ insert (M.c i) (A i)) :
    Plan ι (TargetAlt M A) := fun i =>
  if h : i ∈ S ∧ w i ≠ M.c i then
    some ⟨w i, Finset.mem_filter.2
      ⟨(Finset.mem_insert.1 (hw i h.1)).resolve_left h.2, h.2⟩⟩
  else none

/-- Theorem 7 (a): a change that reaches `τ` gives a valid plan with at most as many changes. A changed feature that
keeps the bin of the instance only lowers each margin, because `z` and the deviations are not negative. -/
theorem plan_of_reaching (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a) {S : Finset ι}
    {w : ∀ i, B i} (h : Reaches M A τ S w) :
    (targetProblem M A τ cost hcost).Valid ⟨none, fun _ _ => False⟩ (planOf M A S w h.1) ∧
      (Problem.changed (planOf M A S w h.1)).card ≤ S.card := by
  have hchanged : Problem.changed (planOf M A S w h.1) = S.filter fun i => w i ≠ M.c i := by
    ext i
    simp only [Problem.changed, Finset.mem_filter, Finset.mem_univ, true_and]
    by_cases hi : i ∈ S ∧ w i ≠ M.c i
    · simp [planOf, hi]
    · simp only [planOf, hi, ↓reduceDIte, Option.isSome_none, Bool.false_eq_true]
  refine ⟨(valid_targetProblem_iff M A τ cost hcost _).2 fun j hj => ?_, ?_⟩
  · refine lt_of_lt_of_le (h.2 j hj) ?_
    simp only [rivalMargin, hchanged]
    refine add_le_add le_rfl (Finset.sum_le_sum fun i _ => ?_)
    by_cases hi : i ∈ S ∧ w i ≠ M.c i
    · have hb : targetBins M A (planOf M A S w h.1) i = w i := by simp [targetBins, planOf, hi]
      simp [hi.1, hi.2, hb]
    · have hb : targetBins M A (planOf M A S w h.1) i = M.c i := by simp [targetBins, planOf, hi]
      by_cases hS : i ∈ S
      · have hc : w i = M.c i := by tauto
        have hz : 0 ≤ M.z * (M.sd τ i (M.c i) + M.sd j i (M.c i)) :=
          mul_nonneg M.z_nonneg (add_nonneg (M.sd_nonneg _ _ _) (M.sd_nonneg _ _ _))
        simp only [hS, ite_true, Finset.mem_filter, true_and, hc, ne_eq, not_true_eq_false, ite_false,
          movedTerm, keptTerm]
        linarith
      · simp [hS]
  · rw [hchanged]
    exact Finset.card_filter_le _ _

/-- Theorem 7 (b): a valid plan exists iff some change reaches `τ`. -/
theorem exists_plan_iff (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a) :
    (∃ p, (targetProblem M A τ cost hcost).Valid ⟨none, fun _ _ => False⟩ p) ↔
      ∃ S w, Reaches M A τ S w := by
  constructor
  · rintro ⟨p, hp⟩
    exact ⟨_, _, reaches_of_valid M A τ cost hcost hp⟩
  · rintro ⟨S, w, h⟩
    exact ⟨_, (plan_of_reaching M A τ cost hcost h).1⟩

/-- Theorem 7 (c): with unit costs, the smallest valid plan has as many changes as the fewest changes that reach `τ`:
a plan of `k` changes reaches `τ`, and every change that reaches `τ` gives a plan with at most as many changes. -/
theorem smallest_plan_iff (cost : ∀ i, TargetAlt M A i → ℝ) (hcost : ∀ i a, 0 < cost i a) (k : ℕ) :
    (∃ p, (targetProblem M A τ cost hcost).Valid ⟨none, fun _ _ => False⟩ p ∧ (Problem.changed p).card ≤ k) ↔
      ∃ S w, Reaches M A τ S w ∧ S.card ≤ k := by
  constructor
  · rintro ⟨p, hp, hk⟩
    exact ⟨_, _, reaches_of_valid M A τ cost hcost hp, hk⟩
  · rintro ⟨S, w, h, hk⟩
    obtain ⟨hv, hcard⟩ := plan_of_reaching M A τ cost hcost h
    exact ⟨_, hv, hcard.trans hk⟩

end Multiclass

end EBaMlProofs.Recourse
