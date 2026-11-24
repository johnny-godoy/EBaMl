import EBaMlProofs.Recourse.Target
import EBaMlProofs.LocalExplanations.Hardness

/-!
# The counterfactual of a chosen class contains Set Cover

This file proves the reduction of the hardness result for the counterfactual of a chosen class, which `Target.lean`
models. It builds, from an instance of Set Cover, a multiclass classifier and a target class whose valid plans are
exactly the covers. The costs are any positive numbers, so the reduction holds for the fewest changes (unit costs) and
for the cheapest plan.

* `coverMulticlass` is the classifier. Each set is a feature with the bins `false` (the instance) and `true`. The
  target is the class `none`, and each element `k` is a rival class `some k`. A change of feature `i` raises the margin
  of the target against `some k` by `1` if the set `i` holds `k`. Every margin starts at `-1/2`.
* `targetBins_cover` shows that a changed feature of a plan takes the bin `true`.
* `target_valid_iff_cover` is Theorem 8 (a) of the recourse paper: a plan is valid iff its changed features are a
  cover.
* `target_changed_eq_cover`, `target_isOptimum_iff` and `target_isApprox_iff` are Theorem 8 (b): the sets of changes
  of the valid plans are the covers, so the two problems have the same optimum and the same `ρ`-approximate sets.
-/

open Finset EBaMlProofs.LocalExplanations EBaMlProofs.LocalExplanations.Hardness

namespace EBaMlProofs.Recourse.Multiclass

variable {ι κ : Type} [Fintype ι] [DecidableEq ι] [DecidableEq κ]

/-- The classifier of an instance of Set Cover with a target class. The classes are the target `none` and one rival
`some k` for each element `k`. The predicted class `some w` is one of the rivals, which all have the score `1/2` at the
instance. The lemmas of `Target.lean` do not read the predicted class. The deviations are zero. -/
noncomputable def coverMulticlass (I : SetCover ι κ) (w : κ) :
    LocalExplanations.Multiclass ι (fun _ => Bool) (Option κ) where
  w := some w
  b j := j.elim 0 fun _ => 1 / 2
  z := 0
  s j i v := j.elim 0 fun k => if v then -I.gain k i else 0
  sd _ _ _ := 0
  P _ := Finset.univ
  c _ := false
  z_nonneg := le_rfl
  sd_nonneg _ _ _ := le_rfl
  c_mem _ := Finset.mem_univ _

/-- Every bin of every feature is allowed. -/
def allBins : ∀ _ : ι, Finset Bool := fun _ => Finset.univ

omit [DecidableEq ι] in
/-- A changed feature of a plan takes the bin `true`, the only bin other than the bin of the instance. -/
theorem targetBins_cover (I : SetCover ι κ) (w : κ) (p : Plan ι (TargetAlt (coverMulticlass I w) allBins))
    (i : ι) (hi : i ∈ Problem.changed p) : targetBins (coverMulticlass I w) allBins p i = true := by
  simp only [Problem.changed, Finset.mem_filter, Finset.mem_univ, true_and] at hi
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hi
  have hne : v.1 ≠ false := (Finset.mem_filter.1 v.2).2
  simp only [targetBins, hv, Option.map_some, Option.getD_some]
  simpa using hne

/-- The robust margin of the target against the rival `some k` is `-1/2` plus the gains of the changed sets. -/
theorem rivalMargin_cover (I : SetCover ι κ) (w : κ) (p : Plan ι (TargetAlt (coverMulticlass I w) allBins)) (k : κ) :
    rivalMargin (coverMulticlass I w) none (some k) (Problem.changed p) (targetBins (coverMulticlass I w) allBins p) =
      -1 / 2 + ∑ i ∈ Problem.changed p, I.gain k i := by
  have hterm : ∀ i, (if i ∈ Problem.changed p then
        movedTerm (coverMulticlass I w) none (some k) i (targetBins (coverMulticlass I w) allBins p i)
      else keptTerm (coverMulticlass I w) none (some k) i) =
      if i ∈ Problem.changed p then I.gain k i else 0 := by
    intro i
    by_cases hi : i ∈ Problem.changed p
    · simp only [hi, ↓reduceIte, targetBins_cover I w p i hi]
      simp [movedTerm, coverMulticlass]
    · simp only [hi, ↓reduceIte]
      simp [keptTerm, coverMulticlass]
  simp only [rivalMargin, hterm, Finset.sum_ite_mem, Finset.univ_inter]
  simp [coverMulticlass]
  ring

/-- Theorem 8 (a). A plan of the counterfactual of the target class is valid iff the sets of its changed
features cover every element, whatever the positive costs. -/
theorem target_valid_iff_cover (I : SetCover ι κ) (w : κ)
    (cost : ∀ i, TargetAlt (coverMulticlass I w) allBins i → ℝ) (hcost : ∀ i a, 0 < cost i a)
    (p : Plan ι (TargetAlt (coverMulticlass I w) allBins)) :
    (targetProblem (coverMulticlass I w) allBins none cost hcost).Valid ⟨none, fun _ _ => False⟩ p ↔
      I.IsCover (Problem.changed p) := by
  rw [valid_targetProblem_iff]
  constructor
  · intro h k
    have := h (some k) (Option.some_ne_none k)
    rw [rivalMargin_cover] at this
    exact (SetCover.half_lt_sum_iff I k _).1 (by linarith)
  · intro h j hj
    cases j with
    | none => exact absurd rfl hj
    | some k =>
      rw [rivalMargin_cover]
      have := (SetCover.half_lt_sum_iff I k _).2 (h k)
      linarith

/-- The plan that changes the features of `S`. -/
def coverPlan (I : SetCover ι κ) (w : κ) (S : Finset ι) : Plan ι (TargetAlt (coverMulticlass I w) allBins) :=
  fun i => if i ∈ S then some ⟨true, by simp [allBins, coverMulticlass]⟩ else none

theorem changed_coverPlan (I : SetCover ι κ) (w : κ) (S : Finset ι) :
    Problem.changed (coverPlan I w S) = S := by
  ext i
  by_cases hi : i ∈ S <;> simp [Problem.changed, coverPlan, hi]

/-- The sets of changes of the valid plans are the covers, as one property of sets. -/
theorem target_changed_eq_cover (I : SetCover ι κ) (w : κ)
    (cost : ∀ i, TargetAlt (coverMulticlass I w) allBins i → ℝ) (hcost : ∀ i a, 0 < cost i a) :
    (fun S => ∃ p, (targetProblem (coverMulticlass I w) allBins none cost hcost).Valid ⟨none, fun _ _ => False⟩ p ∧
      Problem.changed p = S) = I.IsCover := by
  funext S
  apply propext
  constructor
  · rintro ⟨p, hp, rfl⟩
    exact (target_valid_iff_cover I w cost hcost p).1 hp
  · intro h
    refine ⟨coverPlan I w S, ?_, changed_coverPlan I w S⟩
    rw [target_valid_iff_cover, changed_coverPlan]
    exact h

/-- Theorem 8 (b), the optimum: the fewest changes of the target class equal the size of a smallest cover. -/
theorem target_isOptimum_iff (I : SetCover ι κ) (w : κ)
    (cost : ∀ i, TargetAlt (coverMulticlass I w) allBins i → ℝ) (hcost : ∀ i a, 0 < cost i a) (opt : ℕ) :
    IsOptimum (fun S => ∃ p, (targetProblem (coverMulticlass I w) allBins none cost hcost).Valid
      ⟨none, fun _ _ => False⟩ p ∧ Problem.changed p = S) opt ↔ IsOptimum I.IsCover opt := by
  rw [target_changed_eq_cover]

/-- Theorem 8 (b), the approximate sets: a `ρ`-approximate set of changes is a `ρ`-approximate cover. -/
theorem target_isApprox_iff (I : SetCover ι κ) (w : κ)
    (cost : ∀ i, TargetAlt (coverMulticlass I w) allBins i → ℝ) (hcost : ∀ i a, 0 < cost i a) (ρ : ℝ) (opt : ℕ)
    (F : Finset ι) :
    IsApprox (fun S => ∃ p, (targetProblem (coverMulticlass I w) allBins none cost hcost).Valid
      ⟨none, fun _ _ => False⟩ p ∧ Problem.changed p = S) ρ opt F ↔ IsApprox I.IsCover ρ opt F := by
  rw [target_changed_eq_cover]

end EBaMlProofs.Recourse.Multiclass
