import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.Linarith

/-!
# The covering problem of a decision with several margins

This file is the abstract part of the proof for `src/ebaml/local/reason_search.py`. It knows nothing about
EBMs. A decision with several margins is kept when every margin stays positive in the worst case. Fixing a
feature raises the worst case of each margin by a gain that is not negative.

* `Problem` holds the worst cases and the gains. `Problem.Sufficient` is the condition of the code.
* `sufficient_mono` is Lemma 6.
* `prune` is the removal pass of `Minimal`, and `prune_minimal` is Theorem 4.
* `exact_minimum` is Theorem 5, and `card_search_space` states the cost of `ExactOrMinimal`.
* `minimum_of_card_le_one` is the proof of the flag `minimum` of `Minimal` for a set of at most one feature.
-/

open Finset

namespace EBaMlProofs.LocalExplanations.Covering

/-- The numbers of the problem. `worst k` is the worst case of margin `k` when no feature is fixed, and
`gain k i` is how much fixing feature `i` raises it. -/
structure Problem (ι κ : Type*) where
  /-- The worst case `W_k(∅)` of each margin. -/
  worst : κ → ℝ
  /-- The gain `h_{k,i}` of each feature for each margin. -/
  gain : κ → ι → ℝ
  gain_nonneg : ∀ k i, 0 ≤ gain k i

namespace Problem

variable {ι κ : Type*} (P : Problem ι κ)

/-- A set `F` of features is sufficient when every margin is positive in the worst case. -/
def Sufficient (F : Finset ι) : Prop := ∀ k, 0 < P.worst k + ∑ i ∈ F, P.gain k i

/-- Lemma 6 (monotone). Fixing more features keeps a sufficient set sufficient. -/
theorem sufficient_mono {F G : Finset ι} (h : F ⊆ G) (hF : P.Sufficient F) : P.Sufficient G := by
  intro k
  have := Finset.sum_le_sum_of_subset_of_nonneg h (fun i _ _ => P.gain_nonneg k i)
  linarith [hF k]

open scoped Classical in
/-- The removal pass of `Minimal`. It visits the features of the list in order, and it removes a feature
when the set stays sufficient without it. -/
noncomputable def prune [DecidableEq ι] : List ι → Finset ι → Finset ι
  | [], cur => cur
  | f :: fs, cur => prune fs (if P.Sufficient (cur.erase f) then cur.erase f else cur)

/-- The invariant of the removal pass. A feature of the set was not visited yet, or it is needed. -/
theorem prune_invariant [DecidableEq ι] (l : List ι) :
    ∀ cur : Finset ι, P.Sufficient cur → (∀ f ∈ cur, f ∈ l ∨ ¬ P.Sufficient (cur.erase f)) →
      P.Sufficient (P.prune l cur) ∧ P.prune l cur ⊆ cur ∧
        ∀ f ∈ P.prune l cur, ¬ P.Sufficient ((P.prune l cur).erase f) := by
  induction l with
  | nil =>
    intro cur hs hinv
    refine ⟨hs, subset_rfl, fun f hf => ?_⟩
    simpa [prune] using (hinv f hf).resolve_left (by simp)
  | cons f0 l ih =>
    intro cur hs hinv
    classical
    set cur' : Finset ι := if P.Sufficient (cur.erase f0) then cur.erase f0 else cur with hcur'
    have hsub : cur' ⊆ cur := by
      by_cases h : P.Sufficient (cur.erase f0)
      · simp [hcur', h, Finset.erase_subset]
      · simp [hcur', h]
    have hs' : P.Sufficient cur' := by
      by_cases h : P.Sufficient (cur.erase f0)
      · simp [hcur', h]
      · simpa [hcur', h] using hs
    have hinv' : ∀ f ∈ cur', f ∈ l ∨ ¬ P.Sufficient (cur'.erase f) := by
      intro f hf
      have hfc : f ∈ cur := hsub hf
      rcases hinv f hfc with hmem | hneed
      · rcases List.mem_cons.1 hmem with rfl | hl
        · -- the visited feature stays only if the set is not sufficient without it
          by_cases h : P.Sufficient (cur.erase f)
          · simp [hcur', h] at hf
          · right
            simp [hcur', h]
        · exact Or.inl hl
      · -- an old fact about `cur` carries over, because `cur'.erase f ⊆ cur.erase f` and sufficiency is monotone
        exact Or.inr fun hy => hneed (P.sufficient_mono (Finset.erase_subset_erase f hsub) hy)
    obtain ⟨h1, h2, h3⟩ := ih cur' hs' hinv'
    exact ⟨h1, h2.trans hsub, h3⟩

/-- Theorem 4 (the removal pass). Let `cur` be a sufficient set, and let the list hold all its features.
Then the result of `prune` is a sufficient subset of `cur`, and it is minimal: no feature can be removed from it. -/
theorem prune_minimal [DecidableEq ι] {l : List ι} {cur : Finset ι} (hs : P.Sufficient cur)
    (hall : ∀ f ∈ cur, f ∈ l) :
    P.Sufficient (P.prune l cur) ∧ P.prune l cur ⊆ cur ∧
      ∀ f ∈ P.prune l cur, ¬ P.Sufficient ((P.prune l cur).erase f) :=
  P.prune_invariant l cur hs fun f hf => Or.inl (hall f hf)

/-- Theorem 5 (the exact search). `ExactOrMinimal` tries the sets by size and returns the first sufficient
set. If `F` is sufficient and no set with fewer features than `F` is sufficient, then `F` is a minimum: no
sufficient set has fewer features. -/
theorem exact_minimum {F : Finset ι} (hF : P.Sufficient F)
    (hnone : ∀ G : Finset ι, G.card < F.card → ¬ P.Sufficient G) :
    P.Sufficient F ∧ ∀ G : Finset ι, P.Sufficient G → F.card ≤ G.card := by
  refine ⟨hF, fun G hG => ?_⟩
  by_contra hlt
  exact hnone G (not_le.1 hlt) hG

/-- The cost of the exact search. The search visits the subsets of the features, so for `n` features it
visits at most `2 ^ n` sets. -/
theorem card_search_space [Fintype ι] [DecidableEq ι] :
    (Finset.univ : Finset ι).powerset.card = 2 ^ Fintype.card ι := by
  simp

/-- The flag `minimum` of `Minimal`. A set `R` of at most one feature is the minimum when the empty set is not
sufficient, or when `R` is empty. -/
theorem minimum_of_card_le_one {R : Finset ι} (hR : P.Sufficient R) (hcard : R.card ≤ 1)
    (hempty : R.Nonempty → ¬ P.Sufficient ∅) :
    P.Sufficient R ∧ ∀ G : Finset ι, P.Sufficient G → R.card ≤ G.card := by
  refine ⟨hR, fun G hG => ?_⟩
  by_cases hne : R.Nonempty
  · have hGne : G.Nonempty := by
      rcases G.eq_empty_or_nonempty with rfl | h
      · exact absurd hG (hempty hne)
      · exact h
    have := Finset.card_pos.2 hGne
    omega
  · rw [Finset.not_nonempty_iff_eq_empty] at hne
    simp [hne]

end Problem

end EBaMlProofs.LocalExplanations.Covering
