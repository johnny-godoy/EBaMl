import EBaMlProofs.Recourse.Plans
import Mathlib.Tactic.Ring

/-!
# The reduction of the solvers

This file checks the reduction of the section "Solvers, the greedy and the lower bound" of the local-explanations
paper, which that paper proves on paper only. The problem is `Recourse.Problem`: groups of alternatives with gains of
any sign, as `CoveringProblem` has them.

* `valid_erase` is Lemma 15: an alternative with no positive gain leaves a valid plan valid when it is removed, with
  one change fewer and a lower cost.
* `valid_swap` is Lemma 16: a dominated alternative can be swapped for the alternative that dominates it, with the
  same changed groups.
* `Useful`, `Dominates` and `Kept` are the rules of `reduce_problem`, with `pos` as the position of an alternative in
  its group. `exists_kept` shows that each useful alternative has a kept alternative with gains at least as large.
  `exists_kept_plan`, `exists_plan_iff_kept` and `smallest_plan_iff_kept` are Corollary 6: the reduction keeps the
  existence of a valid plan and its smallest size.
-/

open Finset

namespace EBaMlProofs.Recourse

variable {ι : Type} [Fintype ι] [DecidableEq ι] {α : ι → Type} [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)]
  {κ : Type}

namespace Problem

variable (P : Problem ι α κ)

/-! ### A plan with one group changed -/

omit [Fintype ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem pickGain_update (p : Plan ι α) (i : ι) (o : Option (α i)) (j : ι) (k : κ) :
    P.pickGain (Function.update p i o) j k =
      if j = i then (o.map fun a => P.gain i a k).getD 0 else P.pickGain p j k := by
  by_cases h : j = i
  · subst h
    simp [pickGain]
  · simp [pickGain, h]

omit [Fintype ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem pickCost_update (p : Plan ι α) (i : ι) (o : Option (α i)) (j : ι) :
    P.pickCost (Function.update p i o) j =
      if j = i then (o.map fun a => P.cost i a).getD 0 else P.pickCost p j := by
  by_cases h : j = i
  · subst h
    simp [pickCost]
  · simp [pickCost, h]

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- A sum over the groups with one term replaced. -/
theorem sum_ite_eq (f : ι → ℝ) (i : ι) (x : ℝ) :
    ∑ j, (if j = i then x else f j) = ∑ j, f j - f i + x := by
  rw [← Finset.add_sum_erase _ _ (mem_univ i), ← Finset.add_sum_erase _ f (mem_univ i)]
  have hrest : ∑ j ∈ univ.erase i, (if j = i then x else f j) = ∑ j ∈ univ.erase i, f j :=
    Finset.sum_congr rfl fun j hj => by simp [Finset.ne_of_mem_erase hj]
  rw [hrest]
  simp only [ite_true]
  ring

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem margin_update (p : Plan ι α) (i : ι) (o : Option (α i)) (k : κ) :
    P.margin (Function.update p i o) k = P.margin p k - P.pickGain p i k + (o.map fun a => P.gain i a k).getD 0 := by
  simp only [margin, pickGain_update, sum_ite_eq]
  ring

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem cost_update (p : Plan ι α) (i : ι) (o : Option (α i)) :
    P.cost' (Function.update p i o) = P.cost' p - P.pickCost p i + (o.map fun a => P.cost i a).getD 0 := by
  simp only [cost', pickCost_update, sum_ite_eq]

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem changed_erase (p : Plan ι α) (i : ι) : changed (Function.update p i none) = (changed p).erase i := by
  ext j
  by_cases h : j = i
  · subst h
    simp [changed]
  · simp [changed, h]

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem changed_swap {p : Plan ι α} {i : ι} {a : α i} (hpi : p i = some a) (b : α i) :
    changed (Function.update p i (some b)) = changed p := by
  ext j
  by_cases h : j = i
  · subst h
    simp [changed, hpi]
  · simp [changed, h]

/-! ### Lemma 15: an alternative with no positive gain -/

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- Lemma 15. If a valid plan picks an alternative with no positive gain, the plan without the pick is valid, changes
one group fewer and costs less. -/
theorem valid_erase {L : Limits ι α} {p : Plan ι α} {i : ι} {a : α i} (hp : P.Valid L p) (hpi : p i = some a)
    (hgain : ∀ k, P.gain i a k ≤ 0) :
    P.Valid L (Function.update p i none) ∧ (changed (Function.update p i none)).card + 1 = (changed p).card ∧
      P.cost' (Function.update p i none) < P.cost' p := by
  have hmem : i ∈ changed p := by simp [changed, hpi]
  have hcard : (changed (Function.update p i none)).card + 1 = (changed p).card := by
    rw [changed_erase, Finset.card_erase_add_one hmem]
  refine ⟨⟨fun k => ?_, fun n hn => ?_, fun j b hj => ?_⟩, hcard, ?_⟩
  · rw [margin_update]
    have := hp.1 k
    have hg := hgain k
    simp only [pickGain, hpi, Option.map_some, Option.getD_some, Option.map_none, Option.getD_none]
    linarith
  · have := hp.2.1 n hn
    omega
  · by_cases h : j = i
    · subst h
      simp at hj
    · rw [Function.update_of_ne h] at hj
      exact hp.2.2 j b hj
  · rw [cost_update]
    have := P.cost_pos i a
    simp only [pickCost, hpi, Option.map_some, Option.getD_some, Option.map_none, Option.getD_none]
    linarith

/-! ### Lemma 16: a dominated alternative -/

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- Lemma 16. If a valid plan picks `a`, and `b` of the same group has a gain at least as large for every margin and
is not excluded, the plan with `b` in place of `a` is valid and changes the same groups. Its cost changes by the
difference of the two costs. -/
theorem valid_swap {L : Limits ι α} {p : Plan ι α} {i : ι} {a b : α i} (hp : P.Valid L p) (hpi : p i = some a)
    (hdom : ∀ k, P.gain i a k ≤ P.gain i b k) (hb : ¬ L.excluded i b) :
    P.Valid L (Function.update p i (some b)) ∧ changed (Function.update p i (some b)) = changed p ∧
      P.cost' (Function.update p i (some b)) = P.cost' p - P.cost i a + P.cost i b := by
  refine ⟨⟨fun k => ?_, fun n hn => ?_, fun j c hj => ?_⟩, changed_swap hpi b, ?_⟩
  · rw [margin_update]
    have := hp.1 k
    have hg := hdom k
    simp only [pickGain, hpi, Option.map_some, Option.getD_some]
    linarith
  · rw [changed_swap hpi b]
    exact hp.2.1 n hn
  · by_cases h : j = i
    · subst h
      simp only [Function.update_self, Option.some.injEq] at hj
      exact hj ▸ hb
    · rw [Function.update_of_ne h] at hj
      exact hp.2.2 j c hj
  · rw [cost_update]
    simp [pickCost, hpi]

/-! ### Corollary 6: the reduction of the code -/

variable (pos : ∀ i, α i → ℕ)

/-- An alternative is useful when it has a positive gain for some margin (`any(gain > 0 ...)`). -/
def Useful (i : ι) (a : α i) : Prop := ∃ k, 0 < P.gain i a k

/-- `b` dominates `a` (`_dominates`): its gain is at least as large for every margin, and it is larger for one margin
or `b` comes first in the group. -/
def Dominates (i : ι) (b a : α i) : Prop :=
  (∀ k, P.gain i a k ≤ P.gain i b k) ∧ ((∃ k, P.gain i a k < P.gain i b k) ∨ pos i b < pos i a)

/-- An alternative that `reduce_problem` keeps: it is useful, and no useful alternative dominates it. -/
def Kept (i : ι) (a : α i) : Prop := P.Useful i a ∧ ¬ ∃ b, P.Useful i b ∧ P.Dominates pos i b a

omit [Fintype ι] [DecidableEq ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem dominates_irrefl {i : ι} (a : α i) : ¬ P.Dominates pos i a a := by
  rintro ⟨_, ⟨k, hk⟩ | h⟩
  · exact lt_irrefl _ hk
  · exact lt_irrefl _ h

omit [Fintype ι] [DecidableEq ι] [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
theorem dominates_trans {i : ι} {a b c : α i} (hcb : P.Dominates pos i c b) (hba : P.Dominates pos i b a) :
    P.Dominates pos i c a := by
  refine ⟨fun k => (hba.1 k).trans (hcb.1 k), ?_⟩
  rcases hba.2 with ⟨k, hk⟩ | hpos
  · exact Or.inl ⟨k, hk.trans_le (hcb.1 k)⟩
  · rcases hcb.2 with ⟨k, hk⟩ | hpos'
    · exact Or.inl ⟨k, (hba.1 k).trans_lt hk⟩
    · exact Or.inr (hpos'.trans hpos)

omit [Fintype ι] [DecidableEq ι] [∀ i, DecidableEq (α i)] in
/-- Each useful alternative has a kept alternative in its group with a gain at least as large for every margin. The
proof follows the useful alternatives that dominate it: each step has fewer dominating alternatives. -/
theorem exists_kept (i : ι) (a : α i) (ha : P.Useful i a) :
    ∃ b, P.Kept pos i b ∧ ∀ k, P.gain i a k ≤ P.gain i b k := by
  classical
  suffices H : ∀ n (a : α i), (univ.filter fun b => P.Useful i b ∧ P.Dominates pos i b a).card = n →
      P.Useful i a → ∃ b, P.Kept pos i b ∧ ∀ k, P.gain i a k ≤ P.gain i b k from H _ a rfl ha
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro a hn ha
    by_cases hk : ∃ b, P.Useful i b ∧ P.Dominates pos i b a
    · obtain ⟨b, hbu, hba⟩ := hk
      have hsub : (univ.filter fun c => P.Useful i c ∧ P.Dominates pos i c b) ⊂
          univ.filter fun c => P.Useful i c ∧ P.Dominates pos i c a := by
        refine (Finset.ssubset_iff_of_subset fun c hc => ?_).2 ⟨b, ?_, ?_⟩
        · simp only [mem_filter, mem_univ, true_and] at hc ⊢
          exact ⟨hc.1, P.dominates_trans pos hc.2 hba⟩
        · simp [hbu, hba]
        · simp [P.dominates_irrefl pos]
      obtain ⟨c, hc, hcb⟩ := ih _ (hn ▸ Finset.card_lt_card hsub) b rfl hbu
      exact ⟨c, hc, fun k => (hba.1 k).trans (hcb k)⟩
    · exact ⟨a, ⟨ha, hk⟩, fun _ => le_rfl⟩

/-- The pick of the reduced plan for the pick `a`: a kept alternative that is at least as good, or no pick when `a`
has no positive gain. -/
noncomputable def reduceAlt (i : ι) (a : α i) : Option (α i) := by
  classical
  exact if h : P.Useful i a then some (Classical.choose (P.exists_kept pos i a h)) else none

/-- The reduced plan: each pick replaced by `reduceAlt`. -/
noncomputable def reducePlan (p : Plan ι α) : Plan ι α := fun i => (p i).bind (P.reduceAlt pos i)

omit [Fintype ι] [DecidableEq ι] [∀ i, DecidableEq (α i)] in
theorem pickGain_le_reducePlan (p : Plan ι α) (i : ι) (k : κ) :
    P.pickGain p i k ≤ P.pickGain (P.reducePlan pos p) i k := by
  classical
  cases hpi : p i with
  | none => simp [pickGain, reducePlan, hpi]
  | some a =>
    by_cases ha : P.Useful i a
    · have hspec := Classical.choose_spec (P.exists_kept pos i a ha)
      simp only [pickGain, reducePlan, hpi, Option.bind_some, reduceAlt, ha, dite_true, Option.map_some,
        Option.getD_some]
      exact hspec.2 k
    · have hle : P.gain i a k ≤ 0 := by
        by_contra hpos
        exact ha ⟨k, lt_of_not_ge hpos⟩
      simp only [pickGain, reducePlan, hpi, Option.bind_some, reduceAlt, ha, dite_false, Option.map_some,
        Option.getD_some, Option.map_none, Option.getD_none]
      exact hle

omit [DecidableEq ι] [∀ i, DecidableEq (α i)] in
theorem changed_reducePlan_subset (p : Plan ι α) : changed (P.reducePlan pos p) ⊆ changed p := by
  intro i hi
  have hsome : ((p i).bind (P.reduceAlt pos i)).isSome = true := (Finset.mem_filter.1 hi).2
  refine Finset.mem_filter.2 ⟨mem_univ i, ?_⟩
  cases hpi : p i with
  | none => rw [hpi] at hsome; simp at hsome
  | some _ => simp

omit [Fintype ι] [DecidableEq ι] [∀ i, DecidableEq (α i)] in
theorem kept_of_reducePlan (p : Plan ι α) (i : ι) (b : α i) (hb : P.reducePlan pos p i = some b) :
    P.Kept pos i b := by
  classical
  cases hpi : p i with
  | none => simp [reducePlan, hpi] at hb
  | some a =>
    by_cases ha : P.Useful i a
    · simp only [reducePlan, hpi, Option.bind_some, reduceAlt, ha, dite_true, Option.some.injEq] at hb
      exact hb ▸ (Classical.choose_spec (P.exists_kept pos i a ha)).1
    · simp [reducePlan, hpi, reduceAlt, ha] at hb

omit [DecidableEq ι] [∀ i, DecidableEq (α i)] in
/-- Corollary 6, the construction. For a problem with a cap and no exclusion, as `reduce_problem` receives it, each
valid plan gives a valid plan that picks only kept alternatives and changes no more groups. -/
theorem exists_kept_plan (cap : Option ℕ) {p : Plan ι α} (hp : P.Valid ⟨cap, fun _ _ => False⟩ p) :
    ∃ q, P.Valid ⟨cap, fun _ _ => False⟩ q ∧ (∀ i b, q i = some b → P.Kept pos i b) ∧
      (changed q).card ≤ (changed p).card := by
  have hcard := Finset.card_le_card (P.changed_reducePlan_subset pos p)
  refine ⟨P.reducePlan pos p, ⟨fun k => ?_, fun n hn => (hcard.trans (hp.2.1 n hn)), fun _ _ _ h => h⟩,
    P.kept_of_reducePlan pos p, hcard⟩
  exact lt_of_lt_of_le (hp.1 k) (add_le_add le_rfl (Finset.sum_le_sum fun i _ => P.pickGain_le_reducePlan pos p i k))

omit [DecidableEq ι] [∀ i, DecidableEq (α i)] in
/-- Corollary 6, existence: a valid plan exists iff a valid plan of kept alternatives exists. -/
theorem exists_plan_iff_kept (cap : Option ℕ) :
    (∃ p, P.Valid ⟨cap, fun _ _ => False⟩ p) ↔
      ∃ q, P.Valid ⟨cap, fun _ _ => False⟩ q ∧ ∀ i b, q i = some b → P.Kept pos i b := by
  constructor
  · rintro ⟨p, hp⟩
    obtain ⟨q, hq, hkept, _⟩ := P.exists_kept_plan pos cap hp
    exact ⟨q, hq, hkept⟩
  · rintro ⟨q, hq, _⟩
    exact ⟨q, hq⟩

omit [DecidableEq ι] [∀ i, DecidableEq (α i)] in
/-- Corollary 6, size: a valid plan of at most `n` changes exists iff a valid plan of kept alternatives with at most
`n` changes exists. So the smallest size is the same before and after the reduction. -/
theorem smallest_plan_iff_kept (cap : Option ℕ) (n : ℕ) :
    (∃ p, P.Valid ⟨cap, fun _ _ => False⟩ p ∧ (changed p).card ≤ n) ↔
      ∃ q, P.Valid ⟨cap, fun _ _ => False⟩ q ∧ (∀ i b, q i = some b → P.Kept pos i b) ∧ (changed q).card ≤ n := by
  constructor
  · rintro ⟨p, hp, hn⟩
    obtain ⟨q, hq, hkept, hcard⟩ := P.exists_kept_plan pos cap hp
    exact ⟨q, hq, hkept, hcard.trans hn⟩
  · rintro ⟨q, hq, _, hn⟩
    exact ⟨q, hq, hn⟩

end Problem

end EBaMlProofs.Recourse
