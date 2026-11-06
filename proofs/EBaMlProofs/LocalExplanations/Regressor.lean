import EBaMlProofs.LocalExplanations.Model
import EBaMlProofs.LocalExplanations.Covering

/-!
# Regressors: the reduction to the binary model

This file proves that the local explanations of a regressor are the explanations of the binary model of
`Model.lean` with the intercept moved. The code does this in `compute_local_facts`: it passes
`intercept - threshold` to the algorithms of a margin.

* `shift` is the model of a margin with the threshold `τ`. `shift_eps` and `shift_margin_eq` are Lemma 7.
* `shift_robustMargin_neg_iff` is Lemma 8: a change flips the margin iff the new score is on the other side of `τ`.
* `canFlipInside_iff`, `min_change_inside` and `min_change_inside_none` are Theorem 6.
* `ofModels_sufficient` is Lemma 9, and it links the sufficient reason of several margins to `Covering`.
-/

open Finset EBaMlProofs.LocalExplanations.Greedy

namespace EBaMlProofs.LocalExplanations

namespace Model

variable {ι : Type} [Fintype ι] {B : ι → Type} (m : Model ι B)

/-- The model of a margin of a regressor with the threshold `τ`: the same scores, and the intercept `b - τ`. -/
def shift (τ : ℝ) : Model ι B := { m with b := m.b - τ }

omit [Fintype ι] in
@[simp] theorem shift_s (τ : ℝ) : (m.shift τ).s = m.s := rfl
omit [Fintype ι] in
@[simp] theorem shift_sd (τ : ℝ) : (m.shift τ).sd = m.sd := rfl
omit [Fintype ι] in
@[simp] theorem shift_P (τ : ℝ) : (m.shift τ).P = m.P := rfl
omit [Fintype ι] in
@[simp] theorem shift_c (τ : ℝ) : (m.shift τ).c = m.c := rfl
omit [Fintype ι] in
@[simp] theorem shift_z (τ : ℝ) : (m.shift τ).z = m.z := rfl

/-- Lemma 7, the logit: the logit of the shifted model is the predicted value minus `τ`. -/
theorem shift_logit (τ : ℝ) : (m.shift τ).logit = m.logit - τ := by
  simp only [logit, shift]
  ring

/-- Lemma 7, the sign: the sign of the shifted model is `1` if the predicted value is above `τ`, else `-1`.
This is the sign `sign` of `Margin` in `decisions.py`. -/
theorem shift_eps (τ : ℝ) : (m.shift τ).eps = if τ < m.logit then 1 else -1 := by
  simp only [eps, shift_logit, sub_pos]

/-- The margin is the sign times the logit. -/
theorem margin_eq : m.margin = m.eps * m.logit := by
  simp only [margin, logit, t, mul_add, Finset.mul_sum]

/-- Lemma 7, the margin: the margin of the shifted model is the distance of the predicted value from `τ`,
on the side of the decision. -/
theorem shift_margin_eq (τ : ℝ) :
    (m.shift τ).margin = (if τ < m.logit then 1 else -1) * (m.logit - τ) := by
  rw [margin_eq, shift_eps, shift_logit]

variable [DecidableEq ι]

/-- The predicted value when the features in `S` take the bins `w`. -/
noncomputable def newScore (S : Finset ι) (w : ∀ i, B i) : ℝ :=
  m.b + ∑ i, if i ∈ S then m.s i (w i) else m.s i (m.c i)

/-- Lemma 8 (a flip is a crossing of the threshold). With no confidence margin, the robust margin of the shifted
model is the sign times the new predicted value minus `τ`. So it is negative iff the new value is below `τ`
when the value of the instance is above `τ`, and above `τ` otherwise. -/
theorem shift_robustMargin_neg_iff (hz : m.z = 0) (τ : ℝ) (S : Finset ι) (w : ∀ i, B i) :
    (m.shift τ).robustMargin S w < 0 ↔
      (if τ < m.logit then m.newScore S w < τ else τ < m.newScore S w) := by
  have h : (m.shift τ).robustMargin S w = (m.shift τ).eps * (m.newScore S w - τ) := by
    simp only [robustMargin, newScore, p, t, shift_s, shift_sd, shift_z, shift_c, hz, zero_mul, add_zero]
    have hb : (m.shift τ).b = m.b - τ := rfl
    have key : ∑ i, (if i ∈ S then (m.shift τ).eps * m.s i (w i) else (m.shift τ).eps * m.s i (m.c i)) =
        (m.shift τ).eps * ∑ i, (if i ∈ S then m.s i (w i) else m.s i (m.c i)) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by split_ifs <;> rfl
    rw [hb, key]
    ring
  rw [h, shift_eps]
  split_ifs with hτ
  · constructor <;> intro h' <;> linarith
  · constructor <;> intro h' <;> linarith

/-! ### Theorem 6: the decision "inside" -/

/-- The answer "inside": the predicted value is above `lo` and at most `hi`. -/
def Inside (lo hi : ℝ) : Prop := lo < m.logit ∧ m.logit ≤ hi

omit [DecidableEq ι] in
/-- When the value is inside, the margin of the low end has the sign `1` and the margin of the high end has
the sign `-1`. These are the signs of `Interval.side`. -/
theorem inside_eps {lo hi : ℝ} (h : m.Inside lo hi) : (m.shift lo).eps = 1 ∧ (m.shift hi).eps = -1 := by
  rw [shift_eps, shift_eps]
  exact ⟨by simp [h.1], by simp [not_lt.2 h.2]⟩

/-- Some admissible change of the features in `S` makes at least one margin of "inside" fail. -/
def CanFlipInside (lo hi : ℝ) (S : Finset ι) : Prop :=
  ∃ w, m.Admissible S w ∧ ((m.shift lo).robustMargin S w < 0 ∨ (m.shift hi).robustMargin S w < 0)

/-- Theorem 6, part 1: the answer "inside" fails when one margin fails. A set can flip it iff it can flip
the margin of the low end or the margin of the high end. -/
theorem canFlipInside_iff (lo hi : ℝ) (S : Finset ι) :
    m.CanFlipInside lo hi S ↔ (m.shift lo).CanFlip S ∨ (m.shift hi).CanFlip S := by
  constructor
  · rintro ⟨w, hw, h | h⟩
    · exact Or.inl ⟨w, hw, h⟩
    · exact Or.inr ⟨w, hw, h⟩
  · rintro (⟨w, hw, h⟩ | ⟨w, hw, h⟩)
    · exact ⟨w, hw, Or.inl h⟩
    · exact ⟨w, hw, Or.inr h⟩

/-- Theorem 6, part 2 (minimum change of "inside"). Let each margin have the position `k` of `min_change`.
Then the smaller of the two positions is the size of the smallest set that flips the answer, and the set of the
margin with the smaller position flips it. -/
theorem min_change_inside {lo hi : ℝ} {n : ℕ} {e₁ e₂ : Fin n ≃ ι}
    (he₁ : Antitone ((m.shift lo).drop ∘ e₁)) {k₁ : ℕ} (hk₁ : k₁ ≤ n)
    (hflip₁ : (m.shift lo).margin < partialSum (m.shift lo).drop e₁ k₁)
    (hbefore₁ : ∀ j, 1 ≤ j → j < k₁ → partialSum (m.shift lo).drop e₁ j ≤ (m.shift lo).margin)
    (he₂ : Antitone ((m.shift hi).drop ∘ e₂)) {k₂ : ℕ} (hk₂ : k₂ ≤ n)
    (hflip₂ : (m.shift hi).margin < partialSum (m.shift hi).drop e₂ k₂)
    (hbefore₂ : ∀ j, 1 ≤ j → j < k₂ → partialSum (m.shift hi).drop e₂ j ≤ (m.shift hi).margin) :
    (∃ S : Finset ι, m.CanFlipInside lo hi S ∧ S.card = min k₁ k₂) ∧
      ∀ S : Finset ι, m.CanFlipInside lo hi S → min k₁ k₂ ≤ S.card := by
  obtain ⟨⟨ha₁, hb₁⟩, hc₁⟩ := (m.shift lo).min_change he₁ hk₁ hflip₁ hbefore₁
  obtain ⟨⟨ha₂, hb₂⟩, hc₂⟩ := (m.shift hi).min_change he₂ hk₂ hflip₂ hbefore₂
  refine ⟨?_, fun S hS => ?_⟩
  · rcases le_total k₁ k₂ with hle | hle
    · exact ⟨prefixSet e₁ hk₁, ⟨(m.shift lo).beta, ha₁, Or.inl hb₁⟩, by
        rw [card_prefixSet, min_eq_left hle]⟩
    · exact ⟨prefixSet e₂ hk₂, ⟨(m.shift hi).beta, ha₂, Or.inr hb₂⟩, by
        rw [card_prefixSet, min_eq_right hle]⟩
  · rcases (m.canFlipInside_iff lo hi S).1 hS with h | h
    · exact (min_le_left _ _).trans (hc₁ S h)
    · exact (min_le_right _ _).trans (hc₂ S h)

/-- Theorem 6, part 3 (no answer): if neither margin can be flipped by any set, "inside" cannot be flipped. -/
theorem min_change_inside_none {lo hi : ℝ} {n : ℕ} {e₁ e₂ : Fin n ≃ ι}
    (he₁ : Antitone ((m.shift lo).drop ∘ e₁)) (h₁ : ∀ k ≤ n, partialSum (m.shift lo).drop e₁ k ≤ (m.shift lo).margin)
    (he₂ : Antitone ((m.shift hi).drop ∘ e₂)) (h₂ : ∀ k ≤ n, partialSum (m.shift hi).drop e₂ k ≤ (m.shift hi).margin)
    (S : Finset ι) : ¬ m.CanFlipInside lo hi S := by
  rw [canFlipInside_iff]
  rintro (h | h)
  · exact (m.shift lo).min_change_none he₁ h₁ S h
  · exact (m.shift hi).min_change_none he₂ h₂ S h

end Model

/-! ### Lemma 9: the sufficient reason of several margins -/

section Several

variable {ι : Type} [Fintype ι] [DecidableEq ι] {B : ι → Type} {κ : Type}

/-- The covering problem of a decision with the margins `ms k`: the worst case of each margin and the gains. -/
noncomputable def problemOfModels (ms : κ → Model ι B) : Covering.Problem ι κ where
  worst k := (ms k).worst ∅
  gain k := (ms k).gain
  gain_nonneg k := (ms k).gain_nonneg

/-- Lemma 9. A set is sufficient for the covering problem iff it is sufficient for each margin. So the
sufficient reason of a decision with several margins is the covering problem of `Covering.lean`. -/
theorem problemOfModels_sufficient (ms : κ → Model ι B) (F : Finset ι) :
    (problemOfModels ms).Sufficient F ↔ ∀ k, (ms k).Sufficient F := by
  constructor
  · intro h k
    rw [Model.sufficient_iff]
    have := h k
    simp only [problemOfModels] at this
    linarith
  · intro h k
    have := (Model.sufficient_iff (ms k) F).1 (h k)
    simp only [problemOfModels]
    linarith

end Several

end EBaMlProofs.LocalExplanations
