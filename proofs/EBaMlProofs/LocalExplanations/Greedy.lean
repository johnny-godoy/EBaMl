import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Max
import Mathlib.Tactic.Linarith

/-!
# The greedy threshold rule

This file is the abstract part of the proof. It knows nothing about EBMs.
It proves that sorting values in descending order and adding them up until a threshold is passed
finds the smallest set whose sum passes the threshold.

* `IsTopSet` is Definition 1.
* `topSet_sum_le` is Lemma 1.
* `prefixSet`, `isTopSet_prefixSet`, `card_prefixSet` and `sum_prefixSet` are Lemma 2.
* `threshold_minimal` is Theorem 1.
* `no_answer` is Corollary 1, and `pos_of_threshold` is Corollary 2.
-/

open Finset

namespace EBaMlProofs.LocalExplanations.Greedy

section TopSet

variable {ι : Type*}

/-- Definition 1 (top set). Every value inside `T` is at least as large as every value outside it. -/
def IsTopSet (d : ι → ℝ) (T : Finset ι) : Prop :=
  ∀ i ∈ T, ∀ j ∉ T, d j ≤ d i

/-- Lemma 1 (top-set bound). A top set has the largest sum among the sets of its size.
The size condition cannot be dropped: with `d = (5, -1)` and `T = {both}`,
the set `S = {first}` has a larger sum than `T`. -/
theorem topSet_sum_le [DecidableEq ι] {d : ι → ℝ} {T S : Finset ι} (hT : IsTopSet d T) (hcard : S.card = T.card) :
    ∑ i ∈ S, d i ≤ ∑ i ∈ T, d i := by
  have h1 := Finset.card_sdiff_add_card_inter S T
  have h2 := Finset.card_sdiff_add_card_inter T S
  rw [Finset.inter_comm T S] at h2
  have hcd : (S \ T).card = (T \ S).card := by omega
  have e1 : ∑ i ∈ S ∩ T, d i + ∑ i ∈ S \ T, d i = ∑ i ∈ S, d i :=
    Finset.sum_inter_add_sum_sdiff S T d
  have e2 : ∑ i ∈ T ∩ S, d i + ∑ i ∈ T \ S, d i = ∑ i ∈ T, d i :=
    Finset.sum_inter_add_sum_sdiff T S d
  rw [Finset.inter_comm T S] at e2
  have key : ∑ i ∈ S \ T, d i ≤ ∑ i ∈ T \ S, d i := by
    rcases (S \ T).eq_empty_or_nonempty with hE | hne
    · have hE' : T \ S = ∅ := by
        rw [← Finset.card_eq_zero, ← hcd, hE, Finset.card_empty]
      rw [hE, hE']
    · obtain ⟨i0, hi0, hmax⟩ := Finset.exists_max_image (S \ T) d hne
      have hlow : ∀ j ∈ T \ S, d i0 ≤ d j := fun j hj =>
        hT j (Finset.mem_sdiff.1 hj).1 i0 (Finset.mem_sdiff.1 hi0).2
      calc ∑ i ∈ S \ T, d i ≤ (S \ T).card • d i0 := Finset.sum_le_card_nsmul _ _ _ hmax
        _ = (T \ S).card • d i0 := by rw [hcd]
        _ ≤ ∑ j ∈ T \ S, d j := Finset.card_nsmul_le_sum _ _ _ hlow
  linarith

end TopSet

section Ordering

variable {ι : Type*} {n : ℕ}

/-- The value in position `j` of the ordering, and `0` beyond the end. -/
def seq (d : ι → ℝ) (e : Fin n ≃ ι) (j : ℕ) : ℝ :=
  if h : j < n then d (e ⟨j, h⟩) else 0

/-- `Π_k`: the sum of the first `k` values of the ordering. -/
def partialSum (d : ι → ℝ) (e : Fin n ≃ ι) (k : ℕ) : ℝ :=
  ∑ j ∈ range k, seq d e j

theorem castLE_injective (e : Fin n ≃ ι) {k : ℕ} (hk : k ≤ n) :
    Function.Injective fun j : Fin k => e (Fin.castLE hk j) :=
  fun _ _ h => Fin.castLE_injective hk (e.injective h)

/-- `T_k`: the first `k` elements of the ordering. -/
def prefixSet (e : Fin n ≃ ι) {k : ℕ} (hk : k ≤ n) : Finset ι :=
  (univ : Finset (Fin k)).map ⟨fun j => e (Fin.castLE hk j), castLE_injective e hk⟩

theorem card_eq [Fintype ι] (e : Fin n ≃ ι) : Fintype.card ι = n :=
  (Fintype.card_congr e.symm).trans (Fintype.card_fin n)

/-- Lemma 2, size: `T_k` has `k` elements. -/
theorem card_prefixSet (e : Fin n ≃ ι) {k : ℕ} (hk : k ≤ n) : (prefixSet e hk).card = k := by
  simp [prefixSet]

theorem mem_prefixSet (e : Fin n ≃ ι) {k : ℕ} (hk : k ≤ n) (i : ι) :
    i ∈ prefixSet e hk ↔ ((e.symm i : Fin n) : ℕ) < k := by
  constructor
  · intro h
    obtain ⟨j, -, rfl⟩ := Finset.mem_map.1 h
    simp
  · intro h
    refine Finset.mem_map.2 ⟨⟨(e.symm i : Fin n), h⟩, Finset.mem_univ _, ?_⟩
    simp

/-- Lemma 2, top set: `T_k` is a top set when the ordering is antitone. -/
theorem isTopSet_prefixSet {d : ι → ℝ} {e : Fin n ≃ ι} (he : Antitone (d ∘ e)) {k : ℕ}
    (hk : k ≤ n) : IsTopSet d (prefixSet e hk) := by
  intro i hi j hj
  rw [mem_prefixSet] at hi hj
  have hle : e.symm i ≤ e.symm j := Fin.le_def.2 (by omega)
  have := he hle
  simpa using this

/-- Lemma 2, sum: the sum of `d` over `T_k` is `Π_k`. -/
theorem sum_prefixSet (d : ι → ℝ) (e : Fin n ≃ ι) {k : ℕ} (hk : k ≤ n) :
    ∑ i ∈ prefixSet e hk, d i = partialSum d e k := by
  rw [prefixSet, Finset.sum_map, partialSum, ← Fin.sum_univ_eq_sum_range (seq d e) k]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hj : (j : ℕ) < n := lt_of_lt_of_le j.2 hk
  simp only [seq, hj, dite_true]
  rfl

variable [Fintype ι] [DecidableEq ι]

/-- Theorem 1 (greedy threshold). Let the ordering be antitone, let `Π_k` pass `τ`, and let no earlier
`Π_j` pass it. Then (a) the first `k` elements have a sum above `τ`, and (b) every set with a sum above
`τ` has at least `k` elements. -/
theorem threshold_minimal {d : ι → ℝ} {e : Fin n ≃ ι} (he : Antitone (d ∘ e)) {τ : ℝ} {k : ℕ}
    (hk : k ≤ n) (hgt : τ < partialSum d e k) (hbefore : ∀ j < k, partialSum d e j ≤ τ) :
    τ < ∑ i ∈ prefixSet e hk, d i ∧ ∀ S : Finset ι, τ < ∑ i ∈ S, d i → k ≤ S.card := by
  refine ⟨by rw [sum_prefixSet]; exact hgt, fun S hS => ?_⟩
  by_contra hlt
  replace hlt := not_le.1 hlt
  have hSn : S.card ≤ n := (Finset.card_le_univ S).trans (card_eq e).le
  have hb := topSet_sum_le (isTopSet_prefixSet he hSn) (card_prefixSet e hSn).symm
  rw [sum_prefixSet] at hb
  have := hbefore S.card hlt
  linarith

/-- Corollary 1 (no answer). If no `Π_k` passes `τ`, then no set has a sum above `τ`. -/
theorem no_answer {d : ι → ℝ} {e : Fin n ≃ ι} (he : Antitone (d ∘ e)) {τ : ℝ}
    (h : ∀ k ≤ n, partialSum d e k ≤ τ) (S : Finset ι) : ∑ i ∈ S, d i ≤ τ := by
  have hSn : S.card ≤ n := (Finset.card_le_univ S).trans (card_eq e).le
  have hb := topSet_sum_le (isTopSet_prefixSet he hSn) (card_prefixSet e hSn).symm
  rw [sum_prefixSet] at hb
  exact hb.trans (h _ hSn)

omit [Fintype ι] [DecidableEq ι] in
/-- Corollary 2 (chosen values are positive). Under the assumptions of `threshold_minimal`,
each of the first `k` values is positive. -/
theorem pos_of_threshold {d : ι → ℝ} {e : Fin n ≃ ι} (he : Antitone (d ∘ e)) {τ : ℝ} {k : ℕ}
    (hk : k ≤ n) (hgt : τ < partialSum d e k) (hbefore : ∀ j < k, partialSum d e j ≤ τ)
    (j : Fin n) (hjk : (j : ℕ) < k) : 0 < d (e j) := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
  have hm : m < n := by omega
  have hstep : partialSum d e (m + 1) = partialSum d e m + d (e ⟨m, hm⟩) := by
    simp [partialSum, Finset.sum_range_succ, seq, hm]
  have hprev := hbefore m (Nat.lt_succ_self m)
  have hlast : 0 < d (e ⟨m, hm⟩) := by linarith
  have hle : j ≤ (⟨m, hm⟩ : Fin n) := Fin.le_def.2 (by simp; omega)
  have := he hle
  simp only [Function.comp] at this
  linarith

end Ordering

end EBaMlProofs.LocalExplanations.Greedy
