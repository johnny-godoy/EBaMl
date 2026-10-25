import EBaMlProofs.LocalExplanations.Greedy
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The EBM model and the two optimality theorems

This file formalizes the model of `src/ebaml/local/facts.py` and applies the abstract result of
`EBaMlProofs.Greedy` to it.

* `Model` and the definitions that follow it are Section 2 of the paper.
* `margin_nonneg` is Lemma 3, and `canFlip_iff` (with the lemmas before it) is Lemma 4.
* `worst_eq` and `sufficient_iff` are Lemma 5.
* `min_change`, `min_change_none` and `chosen_bin_differs` are Theorem 2.
* `min_sufficient_reason` and `min_sufficient_reason_none` are Theorem 3.
-/

open Finset EBaMlProofs.LocalExplanations.Greedy

namespace EBaMlProofs.LocalExplanations

/-- A binary EBM without interactions, the instance to explain, and the confidence level.
Feature `i` has the bins `B i`. The plausible bins `P i` are a finite set that holds the bin `c i` of the
instance. The field `z` is the number of standard deviations. -/
structure Model (ι : Type) (B : ι → Type) where
  /-- The intercept `b`. -/
  b : ℝ
  /-- The number of standard deviations `z`. -/
  z : ℝ
  /-- The mean score `s_i` of each bin. -/
  s : ∀ i, B i → ℝ
  /-- The standard deviation `σ_i` of the score of each bin. -/
  sd : ∀ i, B i → ℝ
  /-- The plausible bins `P_i`. -/
  P : ∀ i, Finset (B i)
  /-- The bin `c_i` of the instance. -/
  c : ∀ i, B i
  z_nonneg : 0 ≤ z
  sd_nonneg : ∀ i v, 0 ≤ sd i v
  c_mem : ∀ i, c i ∈ P i

namespace Model

variable {ι : Type} [Fintype ι] {B : ι → Type} (m : Model ι B)

/-- The logit `L`. -/
noncomputable def logit : ℝ := m.b + ∑ i, m.s i (m.c i)

/-- The sign `ε`: `1` if `L > 0`, else `-1`. -/
noncomputable def eps : ℝ := if 0 < m.logit then 1 else -1

/-- The signed score `t_i(v) = ε s_i(v)`. -/
noncomputable def t (i : ι) (v : B i) : ℝ := m.eps * m.s i v

/-- The margin `M = ε b + Σ t_i(c_i)`. -/
noncomputable def margin : ℝ := m.eps * m.b + ∑ i, m.t i (m.c i)

/-- The pessimistic score `p_i(v) = t_i(v) + z σ_i(v)`. -/
noncomputable def p (i : ι) (v : B i) : ℝ := m.t i v + m.z * m.sd i v

/-- The guaranteed score `g_i(v) = t_i(v) - z σ_i(v)`. -/
noncomputable def g (i : ι) (v : B i) : ℝ := m.t i v - m.z * m.sd i v

/-- The minimum of a score over the plausible bins of feature `i`. -/
noncomputable def pmin (f : ∀ i, B i → ℝ) (i : ι) : ℝ :=
  (m.P i).inf' ⟨m.c i, m.c_mem i⟩ (f i)

/-- The drop `d_i = t_i(c_i) - min_{v ∈ P_i} p_i(v)`. -/
noncomputable def drop (i : ι) : ℝ := m.t i (m.c i) - m.pmin m.p i

/-- The gain `h_i = g_i(c_i) - min_{v ∈ P_i} g_i(v)`. -/
noncomputable def gain (i : ι) : ℝ := m.g i (m.c i) - m.pmin m.g i

/-- A bin `β_i` at which `p_i` reaches its minimum over `P_i`. -/
noncomputable def beta (i : ι) : B i :=
  Classical.choose (Finset.exists_mem_eq_inf' (s := m.P i) ⟨m.c i, m.c_mem i⟩ (m.p i))

theorem beta_mem (i : ι) : m.beta i ∈ m.P i :=
  (Classical.choose_spec
    (Finset.exists_mem_eq_inf' (s := m.P i) ⟨m.c i, m.c_mem i⟩ (m.p i))).1

theorem p_beta (i : ι) : m.p i (m.beta i) = m.pmin m.p i :=
  ((Classical.choose_spec
    (Finset.exists_mem_eq_inf' (s := m.P i) ⟨m.c i, m.c_mem i⟩ (m.p i))).2).symm

omit [Fintype ι] in
theorem pmin_le (f : ∀ i, B i → ℝ) {i : ι} {v : B i} (hv : v ∈ m.P i) : m.pmin f i ≤ f i v :=
  Finset.inf'_le (f i) hv

variable [DecidableEq ι]

/-- The robust margin `R(S, w)`: the features in `S` take the bins `w_i`, and the others keep the bin of the
instance. -/
noncomputable def robustMargin (S : Finset ι) (w : ∀ i, B i) : ℝ :=
  m.eps * m.b + ∑ i, if i ∈ S then m.p i (w i) else m.t i (m.c i)

/-- A change is admissible when each new bin is plausible. -/
def Admissible (S : Finset ι) (w : ∀ i, B i) : Prop := ∀ i ∈ S, w i ∈ m.P i

/-- Some admissible change of the features in `S` flips the prediction. -/
def CanFlip (S : Finset ι) : Prop :=
  ∃ w, m.Admissible S w ∧ m.robustMargin S w < 0

/-- The worst-case margin `W(F)` when the features in `F` are fixed. -/
noncomputable def worst (F : Finset ι) : ℝ :=
  m.eps * m.b + ∑ i, if i ∈ F then m.g i (m.c i) else m.pmin m.g i

/-- `F` is sufficient when `W(F) > 0`. -/
def Sufficient (F : Finset ι) : Prop := 0 < m.worst F

/-! ### Lemma 3 -/

omit [DecidableEq ι] in
/-- Lemma 3 (the margin is not negative). -/
theorem margin_nonneg : 0 ≤ m.margin := by
  have h : m.margin = m.eps * m.logit := by
    simp only [margin, logit, t, mul_add, Finset.mul_sum]
  rw [h, eps]
  split_ifs with hL
  · linarith
  · linarith [not_lt.1 hL]

/-! ### Lemma 4 -/

/-- Lemma 4, the exact form of the robust margin. -/
theorem robustMargin_eq (S : Finset ι) (w : ∀ i, B i) :
    m.robustMargin S w = m.margin - ∑ i ∈ S, (m.t i (m.c i) - m.p i (w i)) := by
  have hpt : ∀ i, (if i ∈ S then m.p i (w i) else m.t i (m.c i)) =
      m.t i (m.c i) - (if i ∈ S then m.t i (m.c i) - m.p i (w i) else 0) := by
    intro i
    split_ifs <;> ring
  rw [robustMargin, margin]
  simp_rw [hpt]
  rw [Finset.sum_sub_distrib, Finset.sum_ite_mem, Finset.univ_inter]
  ring

/-- Lemma 4, the lower bound: an admissible change cannot beat the drops. -/
theorem robustMargin_ge {S : Finset ι} {w : ∀ i, B i} (hw : m.Admissible S w) :
    m.margin - ∑ i ∈ S, m.drop i ≤ m.robustMargin S w := by
  rw [robustMargin_eq]
  have : ∑ i ∈ S, (m.t i (m.c i) - m.p i (w i)) ≤ ∑ i ∈ S, m.drop i := by
    refine Finset.sum_le_sum fun i hi => ?_
    have := m.pmin_le m.p (hw i hi)
    simp only [drop]
    linarith
  linarith

/-- Lemma 4, equality: the bins `β` reach the lower bound. -/
theorem robustMargin_beta (S : Finset ι) :
    m.robustMargin S m.beta = m.margin - ∑ i ∈ S, m.drop i := by
  rw [robustMargin_eq]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [drop, p_beta]

omit [DecidableEq ι] in
theorem admissible_beta (S : Finset ι) : m.Admissible S m.beta := fun i _ => m.beta_mem i

/-- Lemma 4: `S` can flip the prediction iff the drops of `S` sum to more than the margin. -/
theorem canFlip_iff (S : Finset ι) : m.CanFlip S ↔ m.margin < ∑ i ∈ S, m.drop i := by
  constructor
  · rintro ⟨w, hw, hR⟩
    have := m.robustMargin_ge hw
    linarith
  · intro h
    refine ⟨m.beta, m.admissible_beta S, ?_⟩
    rw [robustMargin_beta]
    linarith

/-! ### Lemma 5 -/

omit [DecidableEq ι] in
theorem gain_nonneg (i : ι) : 0 ≤ m.gain i := by
  have := m.pmin_le m.g (m.c_mem i)
  simp only [gain]
  linarith

/-- Lemma 5 (fixing adds the gain). -/
theorem worst_eq (F : Finset ι) : m.worst F = m.worst ∅ + ∑ i ∈ F, m.gain i := by
  have hpt : ∀ i, (if i ∈ F then m.g i (m.c i) else m.pmin m.g i) =
      m.pmin m.g i + (if i ∈ F then m.gain i else 0) := by
    intro i
    split_ifs <;> simp [gain]
  have hempty : m.worst ∅ = m.eps * m.b + ∑ i, m.pmin m.g i := by simp [worst]
  rw [hempty, worst]
  simp_rw [hpt]
  rw [Finset.sum_add_distrib, Finset.sum_ite_mem, Finset.univ_inter]
  ring

/-- Lemma 5: `F` is sufficient iff its gains sum to more than `-W(∅)`. -/
theorem sufficient_iff (F : Finset ι) : m.Sufficient F ↔ -m.worst ∅ < ∑ i ∈ F, m.gain i := by
  rw [Sufficient, worst_eq]
  constructor <;> intro h <;> linarith

/-! ### Theorem 2: minimum change -/

/-- Theorem 2 (minimum change). Let `e` sort the drops in descending order, and let `k` be the first
position where the sum of the first `k` drops passes the margin. The code searches from `k = 1`,
and `margin_nonneg` shows that this loses nothing. Then the change of the first `k` features to the bins `β`
flips the prediction, and no change of fewer features flips it. -/
theorem min_change {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e)) {k : ℕ} (hk : k ≤ n)
    (hflip : m.margin < partialSum m.drop e k)
    (hbefore : ∀ j, 1 ≤ j → j < k → partialSum m.drop e j ≤ m.margin) :
    (m.Admissible (prefixSet e hk) m.beta ∧ m.robustMargin (prefixSet e hk) m.beta < 0) ∧
      ∀ S : Finset ι, m.CanFlip S → k ≤ S.card := by
  have hbefore' : ∀ j < k, partialSum m.drop e j ≤ m.margin := by
    intro j hj
    rcases Nat.eq_zero_or_pos j with rfl | hpos
    · simpa [partialSum] using m.margin_nonneg
    · exact hbefore j hpos hj
  obtain ⟨ha, hb⟩ := threshold_minimal he hk hflip hbefore'
  refine ⟨⟨m.admissible_beta _, ?_⟩, fun S hS => hb S ((m.canFlip_iff S).1 hS)⟩
  rw [robustMargin_beta]
  linarith

/-- Theorem 2, no answer: if no sum of the first `k` drops passes the margin, no change flips the
prediction. -/
theorem min_change_none {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e))
    (h : ∀ k ≤ n, partialSum m.drop e k ≤ m.margin) (S : Finset ι) : ¬ m.CanFlip S := by
  rw [canFlip_iff, not_lt]
  exact no_answer he h S

omit [DecidableEq ι] in
/-- Theorem 2, the chosen bins: each of the first `k` features moves to a bin `β_i` other than `c_i`. -/
theorem chosen_bin_differs {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e)) {k : ℕ}
    (hk : k ≤ n) (hflip : m.margin < partialSum m.drop e k)
    (hbefore : ∀ j < k, partialSum m.drop e j ≤ m.margin) (j : Fin n) (hjk : (j : ℕ) < k) :
    m.beta (e j) ≠ m.c (e j) := by
  have hpos := pos_of_threshold he hk hflip hbefore j hjk
  intro heq
  have h1 := m.p_beta (e j)
  rw [heq] at h1
  have h2 : 0 ≤ m.z * m.sd (e j) (m.c (e j)) := mul_nonneg m.z_nonneg (m.sd_nonneg _ _)
  simp only [drop] at hpos
  simp only [p] at h1
  linarith

/-! ### Theorem 3: minimum sufficient reason -/

/-- Theorem 3 (minimum sufficient reason). Let `e` sort the gains in descending order, and let `k` be the
first position, counted from `0`, where `W(∅)` plus the sum of the first `k` gains is above zero. Then the
first `k` features are sufficient, and no set of fewer features is sufficient. -/
theorem min_sufficient_reason {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.gain ∘ e)) {k : ℕ}
    (hk : k ≤ n) (hsuff : 0 < m.worst ∅ + partialSum m.gain e k)
    (hbefore : ∀ j < k, m.worst ∅ + partialSum m.gain e j ≤ 0) :
    m.Sufficient (prefixSet e hk) ∧ ∀ F : Finset ι, m.Sufficient F → k ≤ F.card := by
  obtain ⟨ha, hb⟩ := threshold_minimal (τ := -m.worst ∅) he hk (by linarith)
    (fun j hj => by linarith [hbefore j hj])
  exact ⟨(m.sufficient_iff _).2 ha, fun F hF => hb F ((m.sufficient_iff F).1 hF)⟩

/-- Theorem 3, no answer: if no `k` works, no set is sufficient, and the set of all features is not
sufficient either. -/
theorem min_sufficient_reason_none {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.gain ∘ e))
    (h : ∀ k ≤ n, m.worst ∅ + partialSum m.gain e k ≤ 0) (F : Finset ι) : ¬ m.Sufficient F := by
  rw [sufficient_iff, not_lt]
  exact no_answer he (fun k hk => by linarith [h k hk]) F

end Model

end EBaMlProofs.LocalExplanations
