import EBaMlProofs.LocalExplanations.Model
import EBaMlProofs.LocalExplanations.Regressor

/-!
# Multiclass classifiers: the reduction to the binary model

This file proves that the local explanations of a multiclass classifier are the explanations of binary models,
one for each rival class. The code does this in `MulticlassLocal.frame` and `compute_local_facts`: the margin of the
predicted class `w` against a rival `j` is `s_w - s_j`, and its deviation is the sum of the two deviations.

* `Multiclass` is the model of Section 2 for several classes, and `marginModel` is the binary model of one margin.
* `winner_iff` and `marginModel_eps` are Lemma 10: the predicted class is the one whose margins are all positive,
  and then every margin model has the sign `1`.
* `canFlipClass_iff` is Theorem 7, part 1: the class changes when one margin fails.
* `min_change_class` and `min_change_class_none` are Theorem 7, parts 2 and 3: the smallest change of the class is the
  smallest of the answers for the rivals.
* `sufficient_class_iff` is Lemma 11: a set keeps the class iff it is sufficient for each margin. So the
  sufficient reason is the covering problem of `Covering.lean`.
* `sd_sum_bound` is Lemma 12: the sum of two deviations bounds the deviation of their difference for any correlation.
-/

open Finset EBaMlProofs.LocalExplanations.Greedy

namespace EBaMlProofs.LocalExplanations

/-- A multiclass EBM without interactions, the instance to explain, and the confidence level. The classes are
`κ`, the predicted class is `w`, and each class has its own intercept, scores and deviations. -/
structure Multiclass (ι : Type) (B : ι → Type) (κ : Type) where
  /-- The predicted class. -/
  w : κ
  /-- The intercept `b_k` of each class. -/
  b : κ → ℝ
  /-- The number of standard deviations `z`. -/
  z : ℝ
  /-- The mean score `s_{k,i}` of each bin, for each class. -/
  s : κ → ∀ i, B i → ℝ
  /-- The standard deviation `σ_{k,i}` of the score of each bin, for each class. -/
  sd : κ → ∀ i, B i → ℝ
  /-- The plausible bins `P_i`. -/
  P : ∀ i, Finset (B i)
  /-- The bin `c_i` of the instance. -/
  c : ∀ i, B i
  z_nonneg : 0 ≤ z
  sd_nonneg : ∀ k i v, 0 ≤ sd k i v
  c_mem : ∀ i, c i ∈ P i

namespace Multiclass

variable {ι : Type} [Fintype ι] {B : ι → Type} {κ : Type} (M : Multiclass ι B κ)

/-- The score `S_k` of a class: the intercept and the scores of the bins of the instance. -/
noncomputable def score (k : κ) : ℝ := M.b k + ∑ i, M.s k i (M.c i)

/-- The binary model of the margin of the predicted class against the rival `j`. It has the intercept
`b_w - b_j`, the scores `s_w - s_j` and the deviations `σ_w + σ_j`. -/
noncomputable def marginModel (j : κ) : Model ι B where
  b := M.b M.w - M.b j
  z := M.z
  s i v := M.s M.w i v - M.s j i v
  sd i v := M.sd M.w i v + M.sd j i v
  P := M.P
  c := M.c
  z_nonneg := M.z_nonneg
  sd_nonneg _ _ := add_nonneg (M.sd_nonneg _ _ _) (M.sd_nonneg _ _ _)
  c_mem := M.c_mem

/-- The logit of the margin model is the difference of the two scores. -/
theorem logit_marginModel (j : κ) : (M.marginModel j).logit = M.score M.w - M.score j := by
  simp only [Model.logit, marginModel, score, Finset.sum_sub_distrib]
  ring

/-- Lemma 10, part 1. Every margin is positive iff the predicted class has a higher score than every rival. -/
theorem winner_iff : (∀ j, j ≠ M.w → 0 < (M.marginModel j).logit) ↔ ∀ j, j ≠ M.w → M.score j < M.score M.w := by
  constructor <;> intro h j hj
  · have := h j hj
    rw [logit_marginModel] at this
    linarith
  · rw [logit_marginModel]
    linarith [h j hj]

/-- Lemma 10, part 2. When the margin is positive, the margin model has the sign `1`, as `Margin(1, 0, ...)` has in
`MulticlassLocal.frame`. -/
theorem marginModel_eps {j : κ} (h : 0 < (M.marginModel j).logit) : (M.marginModel j).eps = 1 := by
  simp [Model.eps, h]

variable [DecidableEq ι]

/-- Some admissible change of the features in `S` makes the predicted class lose to a rival: the robust margin of
at least one rival is negative. -/
def CanFlipClass (S : Finset ι) : Prop :=
  ∃ v : ∀ i, B i, (∀ i ∈ S, v i ∈ M.P i) ∧ ∃ j, j ≠ M.w ∧ (M.marginModel j).robustMargin S v < 0

/-- Theorem 7, part 1: the class changes when one margin fails. A set can change it iff it can flip the margin
of one rival. -/
theorem canFlipClass_iff (S : Finset ι) :
    M.CanFlipClass S ↔ ∃ j, j ≠ M.w ∧ (M.marginModel j).CanFlip S := by
  constructor
  · rintro ⟨v, hv, j, hj, h⟩
    exact ⟨j, hj, v, hv, h⟩
  · rintro ⟨j, hj, v, hv, h⟩
    exact ⟨v, hv, j, hj, h⟩

/-- Theorem 7, part 2 (minimum change of the class). Let the rival `j₀` have the position `k` of `min_change`,
and let every other rival either be impossible to flip, or need at least `k` features. Then `k` is the size of the
smallest set that changes the class, and the set of `j₀` changes it. -/
theorem min_change_class {n : ℕ} {e : Fin n ≃ ι} {j₀ : κ} (hj₀ : j₀ ≠ M.w)
    (he : Antitone ((M.marginModel j₀).drop ∘ e)) {k : ℕ} (hk : k ≤ n)
    (hflip : (M.marginModel j₀).margin < partialSum (M.marginModel j₀).drop e k)
    (hbefore : ∀ j, 1 ≤ j → j < k → partialSum (M.marginModel j₀).drop e j ≤ (M.marginModel j₀).margin)
    (hrivals : ∀ j, j ≠ M.w →
      (∀ S, ¬ (M.marginModel j).CanFlip S) ∨ ∀ S, (M.marginModel j).CanFlip S → k ≤ S.card) :
    (∃ S : Finset ι, M.CanFlipClass S ∧ S.card = k) ∧ ∀ S : Finset ι, M.CanFlipClass S → k ≤ S.card := by
  obtain ⟨⟨ha, hb⟩, _⟩ := (M.marginModel j₀).min_change he hk hflip hbefore
  refine ⟨⟨prefixSet e hk, ⟨(M.marginModel j₀).beta, ha, j₀, hj₀, hb⟩, card_prefixSet e hk⟩, fun S hS => ?_⟩
  obtain ⟨j, hj, hflipj⟩ := (M.canFlipClass_iff S).1 hS
  rcases hrivals j hj with hnone | hcard
  · exact absurd hflipj (hnone S)
  · exact hcard S hflipj

/-- Theorem 7, part 3 (no answer): if no set can flip the margin of any rival, no set changes the class. -/
theorem min_change_class_none (hnone : ∀ j, j ≠ M.w → ∀ S, ¬ (M.marginModel j).CanFlip S) (S : Finset ι) :
    ¬ M.CanFlipClass S := by
  rw [canFlipClass_iff]
  rintro ⟨j, hj, h⟩
  exact hnone j hj S h

/-- The sufficient-reason problem of the predicted class: one margin for each rival. -/
noncomputable def problem : Covering.Problem ι {j // j ≠ M.w} :=
  problemOfModels fun j => M.marginModel j.1

/-- Lemma 11. A set is sufficient for the covering problem iff it is sufficient for the margin of each rival. So the
sufficient reason of a multiclass classifier is the covering problem of `Covering.lean`, and `prune_minimal` and
`exact_minimum` apply to it. -/
theorem sufficient_class_iff (F : Finset ι) :
    M.problem.Sufficient F ↔ ∀ j, j ≠ M.w → (M.marginModel j).Sufficient F := by
  rw [problem, problemOfModels_sufficient]
  exact ⟨fun h j hj => h ⟨j, hj⟩, fun h j => h j.1 j.2⟩

end Multiclass

/-- Lemma 12. The deviation of the difference of two scores is at most the sum of the two deviations, for any
correlation `ρ` between `-1` and `1`. The code uses this sum, because an EBM stores no covariance between classes. -/
theorem sd_sum_bound {sx sy ρ : ℝ} (hx : 0 ≤ sx) (hy : 0 ≤ sy) (hρ : -1 ≤ ρ) :
    sx ^ 2 + sy ^ 2 - 2 * ρ * sx * sy ≤ (sx + sy) ^ 2 := by
  have : 0 ≤ sx * sy := mul_nonneg hx hy
  nlinarith

end EBaMlProofs.LocalExplanations
