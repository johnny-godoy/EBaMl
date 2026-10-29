import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith

/-!
# Softmax depends only on the differences of the scores

This file is the abstract part of the proof. It knows nothing about EBMs.
It proves that adding one constant to all scores does not change softmax, and that two score vectors with the
same softmax differ by one constant.

* `softmax` is Definition 1.
* `sum_exp_pos` is Lemma 1.
* `softmax_add_const` is Theorem 1.
* `softmax_eq_iff` is Theorem 2.
-/

open Finset

namespace EBaMlProofs.MulticlassScores.Softmax

variable {κ : Type*} [Fintype κ]

/-- Definition 1 (softmax). The probability of class `k` for the scores `s`. -/
noncomputable def softmax (s : κ → ℝ) (k : κ) : ℝ :=
  Real.exp (s k) / ∑ j, Real.exp (s j)

/-- Lemma 1. The denominator of softmax is positive when there is a class. -/
theorem sum_exp_pos [Nonempty κ] (s : κ → ℝ) : 0 < ∑ j, Real.exp (s j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (s j)) Finset.univ_nonempty

/-- Theorem 1 (shift). Adding the constant `c` to the score of every class does not change softmax. -/
theorem softmax_add_const (s : κ → ℝ) (c : ℝ) : softmax (fun k => s k + c) = softmax s := by
  funext k
  simp only [softmax, Real.exp_add, ← Finset.sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- Theorem 2 (only the differences count). Two score vectors have the same softmax if and only if they differ
by one constant, that is, if all their differences between two classes are equal. -/
theorem softmax_eq_iff [Nonempty κ] (s t : κ → ℝ) : softmax t = softmax s ↔ ∃ c : ℝ, ∀ k, t k = s k + c := by
  constructor
  · intro h
    refine ⟨Real.log (∑ j, Real.exp (t j)) - Real.log (∑ j, Real.exp (s j)), fun k => ?_⟩
    have hk := congrArg Real.log (congrFun h k)
    simp only [softmax] at hk
    rw [Real.log_div (Real.exp_pos _).ne' (sum_exp_pos t).ne', Real.log_div (Real.exp_pos _).ne'
      (sum_exp_pos s).ne', Real.log_exp, Real.log_exp] at hk
    linarith
  · rintro ⟨c, hc⟩
    rw [show t = fun k => s k + c from funext hc, softmax_add_const]

end EBaMlProofs.MulticlassScores.Softmax
