import EBaMlProofs.LocalExplanations.Multiclass

/-!
# The sufficient reason of several margins contains Set Cover

This file proves the correctness of a reduction from Set Cover to the covering problem of `Covering.lean`. It
shows that the covering problem is as hard as Set Cover, so no sorting rule solves it. The membership in NP and
the polynomial size of the reduction are in the paper, and the NP-completeness of Set Cover is Karp's theorem.

* `SetCover` and `SetCover.IsCover` are the instance and its covers.
* `SetCover.problem` is the covering problem of the reduction, and `sufficient_iff_cover` is Theorem 8: a set of
  features is sufficient iff the matching sets cover every element.
* `min_size_iff` is the decision form of Theorem 8: a cover of at most `K` sets exists iff a sufficient set of at most
  `K` features exists.
* `multiclass` is a multiclass classifier with one rival class for each element.
  `sufficient_reason_iff_cover` is Theorem 9: its sufficient reason is the problem of the reduction.
-/

open Finset EBaMlProofs.LocalExplanations

namespace EBaMlProofs.LocalExplanations.Hardness

/-- An instance of Set Cover: the set `sets i` of the elements of `κ` for each `i`. -/
structure SetCover (ι κ : Type) where
  /-- The elements of each set. -/
  sets : ι → Finset κ

variable {ι κ : Type} [DecidableEq κ]

namespace SetCover

/-- A set `F` of indices is a cover when every element lies in one of its sets. -/
def IsCover (I : SetCover ι κ) (F : Finset ι) : Prop := ∀ k, ∃ i ∈ F, k ∈ I.sets i

/-- The gain of the reduction: `1` if the set `i` holds the element `k`, else `0`. -/
noncomputable def gain (I : SetCover ι κ) (k : κ) (i : ι) : ℝ := if k ∈ I.sets i then 1 else 0

theorem gain_nonneg (I : SetCover ι κ) (k : κ) (i : ι) : 0 ≤ I.gain k i := by
  unfold gain
  split_ifs <;> norm_num

/-- The covering problem of the reduction: a margin for each element, with the worst case `-1/2`, and a feature for
each set, with the gain `1` for the elements of the set and `0` for the others. -/
noncomputable def problem (I : SetCover ι κ) : Covering.Problem ι κ where
  worst _ := -1 / 2
  gain := I.gain
  gain_nonneg := I.gain_nonneg

/-- The gains of a set `F` pass `1/2` for the element `k` iff `F` has a set that holds `k`. -/
theorem half_lt_sum_iff (I : SetCover ι κ) (k : κ) (F : Finset ι) :
    (1 / 2 : ℝ) < ∑ i ∈ F, I.gain k i ↔ ∃ i ∈ F, k ∈ I.sets i := by
  have hsum : ∑ i ∈ F, I.gain k i = ((F.filter fun i => k ∈ I.sets i).card : ℝ) := by
    simp only [gain]
    rw [Finset.sum_boole]
  rw [hsum]
  constructor
  · intro h
    have hpos : 0 < (F.filter fun i => k ∈ I.sets i).card := by
      by_contra hzero
      have hcard : (F.filter fun i => k ∈ I.sets i).card = 0 := by omega
      rw [hcard] at h
      norm_num at h
    obtain ⟨i, hi⟩ := Finset.card_pos.1 hpos
    rw [Finset.mem_filter] at hi
    exact ⟨i, hi.1, hi.2⟩
  · rintro ⟨i, hi, hk⟩
    have hpos : 0 < (F.filter fun i => k ∈ I.sets i).card :=
      Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨hi, hk⟩⟩
    have hone : (1 : ℝ) ≤ (F.filter fun i => k ∈ I.sets i).card := by exact_mod_cast hpos
    linarith

/-- Theorem 8 (the reduction). A set of features is sufficient for the problem of the reduction iff the matching
sets cover every element. -/
theorem sufficient_iff_cover (I : SetCover ι κ) (F : Finset ι) : I.problem.Sufficient F ↔ I.IsCover F := by
  unfold Covering.Problem.Sufficient IsCover
  refine forall_congr' fun k => ?_
  rw [← half_lt_sum_iff]
  change (0 : ℝ) < -1 / 2 + ∑ i ∈ F, I.gain k i ↔ _
  constructor <;> intro h <;> linarith

/-- Theorem 8, decision form. A cover of at most `K` sets exists iff a sufficient set of at most `K` features
exists. So an algorithm for the minimum sufficient reason decides Set Cover. -/
theorem min_size_iff (I : SetCover ι κ) (K : ℕ) :
    (∃ F : Finset ι, F.card ≤ K ∧ I.problem.Sufficient F) ↔ ∃ F : Finset ι, F.card ≤ K ∧ I.IsCover F := by
  simp only [sufficient_iff_cover]

end SetCover

section Shape

variable [Fintype ι] [DecidableEq ι]

/-- A binary model of the shape of the reduction has a sufficient set `F` iff the gains of `F` pass `1/2`. The
model has two bins for each feature. The bin of the instance scores `a i`, the other bin scores `0`, and the
intercept is `-1/2`. -/
theorem sufficient_of_shape (m : Model ι fun _ => Bool) (a : ι → ℝ) (ha : ∀ i, 0 ≤ a i)
    (hb : m.b = -1 / 2) (hz : m.z = 0) (hP : ∀ i, m.P i = Finset.univ) (hc : ∀ i, m.c i = true)
    (hs1 : ∀ i, m.s i true = a i) (hs0 : ∀ i, m.s i false = 0) (hpos : 1 / 2 < ∑ i, a i) (F : Finset ι) :
    m.Sufficient F ↔ (1 / 2 : ℝ) < ∑ i ∈ F, a i := by
  have hlogit : 0 < m.logit := by
    unfold Model.logit
    simp only [hb, hc, hs1]
    linarith
  have heps : m.eps = 1 := by simp [Model.eps, hlogit]
  have hg : ∀ i v, m.g i v = m.s i v := by
    intro i v
    simp [Model.g, Model.t, heps, hz]
  have hpmin : ∀ i, m.pmin m.g i = 0 := by
    intro i
    refine le_antisymm ?_ ?_
    · have hfalse : false ∈ m.P i := by simp [hP]
      have := m.pmin_le m.g hfalse
      rw [hg, hs0] at this
      exact this
    · unfold Model.pmin
      refine Finset.le_inf' _ _ fun v _ => ?_
      rw [hg]
      cases v
      · simp [hs0]
      · simp [hs1, ha]
  have hgain : ∀ i, m.gain i = a i := by
    intro i
    simp [Model.gain, hpmin, hg, hc, hs1]
  have hworst : m.worst ∅ = -1 / 2 := by
    simp [Model.worst, hpmin, heps, hb]
  rw [Model.sufficient_iff, hworst]
  simp only [hgain]
  constructor <;> intro h <;> linarith

/-- The multiclass classifier of an instance of Set Cover. The classes are the predicted class `none` and one rival
`some k` for each element `k`. The margin of the predicted class against the rival `some k` has the intercept `-1/2`,
and the feature `i` raises it by `1` if the set `i` holds `k`. The deviations are zero. -/
noncomputable def multiclass (I : SetCover ι κ) : Multiclass ι (fun _ => Bool) (Option κ) where
  w := none
  b j := j.elim 0 fun _ => 1 / 2
  z := 0
  s j i v := j.elim 0 fun k => if v then -I.gain k i else 0
  sd _ _ _ := 0
  P _ := Finset.univ
  c _ := true
  z_nonneg := le_rfl
  sd_nonneg _ _ _ := le_rfl
  c_mem _ := Finset.mem_univ _

/-- Theorem 9. Let every element lie in a set, so that the predicted class `none` wins. Then a set of features is
sufficient for the predicted class of `multiclass I` iff the matching sets cover every element. So the sufficient reason
of a multiclass classifier contains Set Cover. -/
theorem sufficient_reason_iff_cover (I : SetCover ι κ) (hcov : ∀ k, ∃ i, k ∈ I.sets i) (F : Finset ι) :
    (multiclass I).problem.Sufficient F ↔ I.IsCover F := by
  rw [Multiclass.sufficient_class_iff]
  have hmargin : ∀ k, ((multiclass I).marginModel (some k)).Sufficient F ↔ ∃ i ∈ F, k ∈ I.sets i := by
    intro k
    rw [← SetCover.half_lt_sum_iff]
    refine sufficient_of_shape ((multiclass I).marginModel (some k)) (I.gain k) (I.gain_nonneg k)
      ?_ rfl (fun _ => rfl) (fun _ => rfl) ?_ ?_ ?_ F
    · simp [Multiclass.marginModel, multiclass]
      norm_num
    · intro i
      simp [Multiclass.marginModel, multiclass]
    · intro i
      simp [Multiclass.marginModel, multiclass]
    · obtain ⟨i, hi⟩ := hcov k
      have hone : I.gain k i = 1 := by simp [SetCover.gain, hi]
      have hle : I.gain k i ≤ ∑ j, I.gain k j :=
        Finset.single_le_sum (fun j _ => I.gain_nonneg k j) (Finset.mem_univ i)
      linarith
  constructor
  · intro h k
    exact (hmargin k).1 (h (some k) (Option.some_ne_none k))
  · intro h j hj
    cases j with
    | none => exact absurd rfl hj
    | some k => exact (hmargin k).2 (h k)

end Shape

end EBaMlProofs.LocalExplanations.Hardness
