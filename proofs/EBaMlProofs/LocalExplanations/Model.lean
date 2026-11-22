import EBaMlProofs.LocalExplanations.Greedy
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The EBM model and the two optimality theorems

This file formalizes the model of the algorithms in `src/ebaml/local/` and applies the abstract result of
`EBaMlProofs.Greedy` to it.

* `Model` and the definitions that follow it are Section 2 of the paper.
* `margin_nonneg` is Lemma 3, and `canFlip_iff` (with the lemmas before it) is Lemma 4.
* `worst_eq` and `sufficient_iff` are Lemma 5.
* `min_change`, `min_change_none` and `chosen_bin_differs` are Theorem 2.
* The section "The general model" restates Lemma 4 and Theorem 2 for the allowed bins `A` of the recourse options
  (`canFlipOn_iff`, `min_changeOn`, `min_changeOn_none`, `chosen_bin_differsOn`). `dropOn_P`, `canFlipOn_P` and
  `isLowest_beta` show that `A = P` gives the old definitions, and Lemma 4 and Theorem 2 are their corollaries.
  `dropOn_mono` and `canFlipOn_mono` show that fewer allowed bins never make a flip easier.
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

/-! ### The general model: allowed bins

A person can move feature `i` only to an allowed bin of `A i`. The code builds `A i` from the recourse options,
and an immutable feature has `A i = ∅`. The bin `c i` of the instance is always a possible end, because to keep it
is no change. So the counterfactual takes its minimum over `bins A i = insert (c i) (A i)`. The sufficient reason
does not use `A`: `worst` and `Sufficient` read only the plausible bins `P i`. -/

section Allowed

variable [∀ i, DecidableEq (B i)] (A : ∀ i, Finset (B i))

/-- The bins that feature `i` can end in: the allowed bins and the bin of the instance. -/
def bins (i : ι) : Finset (B i) := insert (m.c i) (A i)

omit [Fintype ι] [DecidableEq ι] in
theorem c_mem_bins (i : ι) : m.c i ∈ m.bins A i := Finset.mem_insert_self _ _

/-- The minimum of a score over the bins that feature `i` can end in. -/
noncomputable def amin (f : ∀ i, B i → ℝ) (i : ι) : ℝ :=
  (m.bins A i).inf' ⟨m.c i, m.c_mem_bins A i⟩ (f i)

/-- The drop over the allowed bins: `d_i(A) = t_i(c_i) - min_{v ∈ bins A i} p_i(v)`. -/
noncomputable def dropOn (i : ι) : ℝ := m.t i (m.c i) - m.amin A m.p i

/-- The bins `w` reach the minimum of `p_i` over the bins of each feature. The code picks one such bin with
`np.argmin`, and every theorem below holds for any such choice. -/
def IsLowest (w : ∀ i, B i) : Prop := ∀ i, w i ∈ m.bins A i ∧ m.p i (w i) = m.amin A m.p i

omit [DecidableEq ι] in
theorem exists_isLowest : ∃ w, m.IsLowest A w := by
  choose w hw using fun i => Finset.exists_mem_eq_inf' (s := m.bins A i) ⟨m.c i, m.c_mem_bins A i⟩ (m.p i)
  exact ⟨w, fun i => ⟨(hw i).1, (hw i).2.symm⟩⟩

/-- A change is admissible over `A` when each changed feature ends in one of its bins. -/
def AdmissibleOn (S : Finset ι) (w : ∀ i, B i) : Prop := ∀ i ∈ S, w i ∈ m.bins A i

/-- Some admissible change over `A` of the features in `S` flips the prediction. -/
def CanFlipOn (S : Finset ι) : Prop :=
  ∃ w, m.AdmissibleOn A S w ∧ m.robustMargin S w < 0

omit [Fintype ι] [DecidableEq ι] in
theorem amin_le (f : ∀ i, B i → ℝ) {i : ι} {v : B i} (hv : v ∈ m.bins A i) : m.amin A f i ≤ f i v :=
  Finset.inf'_le (f i) hv

/-- Lemma 4 for allowed bins, the lower bound: an admissible change cannot beat the drops. -/
theorem robustMargin_geOn {S : Finset ι} {w : ∀ i, B i} (hw : m.AdmissibleOn A S w) :
    m.margin - ∑ i ∈ S, m.dropOn A i ≤ m.robustMargin S w := by
  rw [robustMargin_eq]
  have : ∑ i ∈ S, (m.t i (m.c i) - m.p i (w i)) ≤ ∑ i ∈ S, m.dropOn A i := by
    refine Finset.sum_le_sum fun i hi => ?_
    have := m.amin_le A m.p (hw i hi)
    simp only [dropOn]
    linarith
  linarith

/-- Lemma 4 for allowed bins, equality: lowest bins reach the lower bound. -/
theorem robustMargin_lowest {w : ∀ i, B i} (hw : m.IsLowest A w) (S : Finset ι) :
    m.robustMargin S w = m.margin - ∑ i ∈ S, m.dropOn A i := by
  rw [robustMargin_eq]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [dropOn, (hw i).2]

omit [DecidableEq ι] in
theorem admissibleOn_lowest {w : ∀ i, B i} (hw : m.IsLowest A w) (S : Finset ι) : m.AdmissibleOn A S w :=
  fun i _ => (hw i).1

/-- Lemma 4 for allowed bins: `S` can flip the prediction iff its drops over `A` sum to more than the margin. -/
theorem canFlipOn_iff (S : Finset ι) : m.CanFlipOn A S ↔ m.margin < ∑ i ∈ S, m.dropOn A i := by
  constructor
  · rintro ⟨w, hw, hR⟩
    have := m.robustMargin_geOn A hw
    linarith
  · intro h
    obtain ⟨w, hw⟩ := m.exists_isLowest A
    refine ⟨w, m.admissibleOn_lowest A hw S, ?_⟩
    rw [m.robustMargin_lowest A hw]
    linarith

/-- Theorem 2 for allowed bins (minimum change). Let `e` sort the drops over `A` in descending order, and let `k`
be the first position where the sum of the first `k` drops passes the margin. Then the change of the first `k`
features to lowest bins `w` flips the prediction, and no admissible change of fewer features flips it. -/
theorem min_changeOn {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.dropOn A ∘ e)) {k : ℕ} (hk : k ≤ n)
    (hflip : m.margin < partialSum (m.dropOn A) e k)
    (hbefore : ∀ j, 1 ≤ j → j < k → partialSum (m.dropOn A) e j ≤ m.margin)
    {w : ∀ i, B i} (hw : m.IsLowest A w) :
    (m.AdmissibleOn A (prefixSet e hk) w ∧ m.robustMargin (prefixSet e hk) w < 0) ∧
      ∀ S : Finset ι, m.CanFlipOn A S → k ≤ S.card := by
  have hbefore' : ∀ j < k, partialSum (m.dropOn A) e j ≤ m.margin := by
    intro j hj
    rcases Nat.eq_zero_or_pos j with rfl | hpos
    · simpa [partialSum] using m.margin_nonneg
    · exact hbefore j hpos hj
  obtain ⟨ha, hb⟩ := threshold_minimal he hk hflip hbefore'
  refine ⟨⟨m.admissibleOn_lowest A hw _, ?_⟩, fun S hS => hb S ((m.canFlipOn_iff A S).1 hS)⟩
  rw [m.robustMargin_lowest A hw]
  linarith

/-- Theorem 2 for allowed bins, no answer: if no sum of the first `k` drops over `A` passes the margin, no
admissible change over `A` flips the prediction. -/
theorem min_changeOn_none {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.dropOn A ∘ e))
    (h : ∀ k ≤ n, partialSum (m.dropOn A) e k ≤ m.margin) (S : Finset ι) : ¬ m.CanFlipOn A S := by
  rw [canFlipOn_iff, not_lt]
  exact no_answer he h S

omit [DecidableEq ι] in
/-- Theorem 2 for allowed bins, the chosen bins: each of the first `k` features moves to a lowest bin other than
`c_i`. So the bin is in `A i`, and an immutable feature is never among the first `k`. -/
theorem chosen_bin_differsOn {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.dropOn A ∘ e)) {k : ℕ}
    (hk : k ≤ n) (hflip : m.margin < partialSum (m.dropOn A) e k)
    (hbefore : ∀ j < k, partialSum (m.dropOn A) e j ≤ m.margin) {w : ∀ i, B i} (hw : m.IsLowest A w)
    (j : Fin n) (hjk : (j : ℕ) < k) : w (e j) ≠ m.c (e j) ∧ w (e j) ∈ A (e j) := by
  have hpos := pos_of_threshold he hk hflip hbefore j hjk
  have hne : w (e j) ≠ m.c (e j) := by
    intro heq
    have h1 := (hw (e j)).2
    rw [heq] at h1
    have h2 : 0 ≤ m.z * m.sd (e j) (m.c (e j)) := mul_nonneg m.z_nonneg (m.sd_nonneg _ _)
    simp only [dropOn] at hpos
    simp only [p] at h1
    linarith
  refine ⟨hne, ?_⟩
  rcases Finset.mem_insert.1 (hw (e j)).1 with h | h
  · exact absurd h hne
  · exact h

/-! #### Fewer allowed bins never make a flip easier (contract 7) -/

omit [Fintype ι] [DecidableEq ι] in
theorem bins_mono {A A' : ∀ i, Finset (B i)} (hA : ∀ i, A i ⊆ A' i) (i : ι) : m.bins A i ⊆ m.bins A' i :=
  Finset.insert_subset_insert _ (hA i)

omit [DecidableEq ι] in
/-- A smaller allowed set never gives a larger drop. -/
theorem dropOn_mono {A A' : ∀ i, Finset (B i)} (hA : ∀ i, A i ⊆ A' i) (i : ι) : m.dropOn A i ≤ m.dropOn A' i := by
  have : m.amin A' m.p i ≤ m.amin A m.p i :=
    Finset.le_inf' _ _ fun v hv => m.amin_le A' m.p (m.bins_mono hA i hv)
  simp only [dropOn]
  linarith

/-- A smaller allowed set, for example one more immutable feature, never makes a set able to flip. -/
theorem canFlipOn_mono {A A' : ∀ i, Finset (B i)} (hA : ∀ i, A i ⊆ A' i) {S : Finset ι}
    (h : m.CanFlipOn A S) : m.CanFlipOn A' S := by
  obtain ⟨w, hw, hR⟩ := h
  exact ⟨w, fun i hi => m.bins_mono hA i (hw i hi), hR⟩

/-! #### The old model is the case `A = P` -/

omit [Fintype ι] [DecidableEq ι] in
theorem bins_P (i : ι) : m.bins m.P i = m.P i := Finset.insert_eq_of_mem (m.c_mem i)

omit [Fintype ι] [DecidableEq ι] in
theorem amin_P (f : ∀ i, B i → ℝ) (i : ι) : m.amin m.P f i = m.pmin f i :=
  le_antisymm
    (Finset.le_inf' _ _ fun v hv => m.amin_le m.P f (by rw [bins_P]; exact hv))
    (Finset.le_inf' _ _ fun v hv => m.pmin_le f (by rw [bins_P] at hv; exact hv))

omit [DecidableEq ι] in
/-- With `A = P`, the drop over the allowed bins is the drop `d_i`. -/
theorem dropOn_P : m.dropOn m.P = m.drop := by
  funext i
  rw [dropOn, drop, amin_P]

omit [Fintype ι] [DecidableEq ι] in
theorem admissibleOn_P (S : Finset ι) (w : ∀ i, B i) : m.AdmissibleOn m.P S w ↔ m.Admissible S w := by
  simp only [AdmissibleOn, Admissible, bins_P]

theorem canFlipOn_P (S : Finset ι) : m.CanFlipOn m.P S ↔ m.CanFlip S := by
  simp only [CanFlipOn, CanFlip, admissibleOn_P]

omit [DecidableEq ι] in
/-- With `A = P`, the bins `β` are lowest bins. -/
theorem isLowest_beta : m.IsLowest m.P m.beta := fun i => by
  rw [bins_P, amin_P]
  exact ⟨m.beta_mem i, m.p_beta i⟩

/-- Allowed bins are plausible bins, so a flip over `A` is a flip of the old model. -/
theorem canFlip_of_canFlipOn (hA : ∀ i, A i ⊆ m.P i) {S : Finset ι} (h : m.CanFlipOn A S) : m.CanFlip S :=
  (m.canFlipOn_P S).1 (m.canFlipOn_mono hA h)

end Allowed

/-! ### Lemma 4: the corollaries for `A = P` -/

/-- Lemma 4, the lower bound: an admissible change cannot beat the drops. -/
theorem robustMargin_ge {S : Finset ι} {w : ∀ i, B i} (hw : m.Admissible S w) :
    m.margin - ∑ i ∈ S, m.drop i ≤ m.robustMargin S w := by
  classical
  rw [← m.dropOn_P]
  exact m.robustMargin_geOn m.P ((m.admissibleOn_P S w).2 hw)

/-- Lemma 4, equality: the bins `β` reach the lower bound. -/
theorem robustMargin_beta (S : Finset ι) :
    m.robustMargin S m.beta = m.margin - ∑ i ∈ S, m.drop i := by
  classical
  rw [← m.dropOn_P]
  exact m.robustMargin_lowest m.P m.isLowest_beta S

omit [DecidableEq ι] in
theorem admissible_beta (S : Finset ι) : m.Admissible S m.beta := fun i _ => m.beta_mem i

/-- Lemma 4: `S` can flip the prediction iff the drops of `S` sum to more than the margin. -/
theorem canFlip_iff (S : Finset ι) : m.CanFlip S ↔ m.margin < ∑ i ∈ S, m.drop i := by
  classical
  rw [← m.canFlipOn_P, canFlipOn_iff, dropOn_P]

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

/-! ### Theorem 2: minimum change, the corollaries for `A = P` -/

/-- Theorem 2 (minimum change). Let `e` sort the drops in descending order, and let `k` be the first
position where the sum of the first `k` drops passes the margin. The code searches from `k = 1`,
and `margin_nonneg` shows that this loses nothing. Then the change of the first `k` features to the bins `β`
flips the prediction, and no change of fewer features flips it. This is `min_changeOn` with `A = P`. -/
theorem min_change {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e)) {k : ℕ} (hk : k ≤ n)
    (hflip : m.margin < partialSum m.drop e k)
    (hbefore : ∀ j, 1 ≤ j → j < k → partialSum m.drop e j ≤ m.margin) :
    (m.Admissible (prefixSet e hk) m.beta ∧ m.robustMargin (prefixSet e hk) m.beta < 0) ∧
      ∀ S : Finset ι, m.CanFlip S → k ≤ S.card := by
  classical
  rw [← m.dropOn_P] at he hflip hbefore
  obtain ⟨⟨ha, hb⟩, hc⟩ := m.min_changeOn m.P he hk hflip hbefore m.isLowest_beta
  exact ⟨⟨(m.admissibleOn_P _ _).1 ha, hb⟩, fun S hS => hc S ((m.canFlipOn_P S).2 hS)⟩

/-- Theorem 2, no answer: if no sum of the first `k` drops passes the margin, no change flips the
prediction. This is `min_changeOn_none` with `A = P`. -/
theorem min_change_none {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e))
    (h : ∀ k ≤ n, partialSum m.drop e k ≤ m.margin) (S : Finset ι) : ¬ m.CanFlip S := by
  classical
  rw [← m.dropOn_P] at he h
  rw [← m.canFlipOn_P]
  exact m.min_changeOn_none m.P he h S

omit [DecidableEq ι] in
/-- Theorem 2, the chosen bins: each of the first `k` features moves to a bin `β_i` other than `c_i`. This is
`chosen_bin_differsOn` with `A = P`. -/
theorem chosen_bin_differs {n : ℕ} {e : Fin n ≃ ι} (he : Antitone (m.drop ∘ e)) {k : ℕ}
    (hk : k ≤ n) (hflip : m.margin < partialSum m.drop e k)
    (hbefore : ∀ j < k, partialSum m.drop e j ≤ m.margin) (j : Fin n) (hjk : (j : ℕ) < k) :
    m.beta (e j) ≠ m.c (e j) := by
  classical
  rw [← m.dropOn_P] at he hflip hbefore
  exact (m.chosen_bin_differsOn m.P he hk hflip hbefore m.isLowest_beta j hjk).1

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
