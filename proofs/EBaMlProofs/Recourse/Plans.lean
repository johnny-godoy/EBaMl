import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith

/-!
# Recourse plans with costs: the abstract problem and its integer program

This file knows nothing about EBMs. A plan picks at most one alternative in each group (one group for each feature,
one alternative for each bin that the feature may move to). Each pick raises each margin by a gain and costs a
positive amount. A plan is valid when every margin is positive, it changes at most `cap` groups, and it picks no
excluded alternative. The exclusions model the allowed values, the immutable features and the earlier plans.

* `Problem`, `Plan`, `Limits` and `Valid` are the definitions of the paper (Section 2).
* `ofPlan_feasible_iff`, `objective_ofPlan`, `toPlan_feasible` and `objective_toPlan` are Theorem 1: the feasible
  points of the integer program of `build_model` are exactly the valid plans, and the objective is the cost.
* `cheapest_of_optimum` is Corollary 1: an optimum of the integer program is a cheapest plan.
* `cost_le_of_valid_imp`, `valid_of_gain_le` and `valid_of_limits_le` are Theorem 2 (contract 7).
* `valid_excludeUsed_new`, `cost_mono_excludeUsed` are Theorem 3: diverse plans.
-/

open Finset

namespace EBaMlProofs.Recourse

variable {ι : Type} [Fintype ι] {α : ι → Type} [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)]
  {κ : Type}

/-- The numbers of the problem: the worst case of each margin, and the gain and the cost of each alternative. -/
structure Problem (ι : Type) (α : ι → Type) (κ : Type) where
  /-- The margin `W_k` when no alternative is picked. -/
  worst : κ → ℝ
  /-- The gain `g_{i,a,k}` of alternative `a` of group `i` for margin `k`. -/
  gain : ∀ i, α i → κ → ℝ
  /-- The cost `w_{i,a}` of alternative `a` of group `i`. -/
  cost : ∀ i, α i → ℝ
  cost_pos : ∀ i a, 0 < cost i a

/-- A plan picks at most one alternative in each group. -/
abbrev Plan (ι : Type) (α : ι → Type) := ∀ i, Option (α i)

/-- The cap on the number of changed groups, and the excluded alternatives. -/
structure Limits (ι : Type) (α : ι → Type) where
  /-- The most groups that a plan may change, or `none` for no cap. -/
  cap : Option ℕ
  /-- The alternatives that a plan must not pick. -/
  excluded : ∀ i, α i → Prop

namespace Problem

variable (P : Problem ι α κ)

/-- The gain of the pick of group `i` for margin `k`, or `0` with no pick. -/
noncomputable def pickGain (p : Plan ι α) (i : ι) (k : κ) : ℝ := ((p i).map fun a => P.gain i a k).getD 0

/-- The cost of the pick of group `i`, or `0` with no pick. -/
noncomputable def pickCost (p : Plan ι α) (i : ι) : ℝ := ((p i).map fun a => P.cost i a).getD 0

/-- The margin `W_k + Σ g` of a plan. -/
noncomputable def margin (p : Plan ι α) (k : κ) : ℝ := P.worst k + ∑ i, P.pickGain p i k

/-- The cost of a plan: the sum of the costs of its picks. -/
noncomputable def cost' (p : Plan ι α) : ℝ := ∑ i, P.pickCost p i

/-- The groups that a plan changes. -/
def changed (p : Plan ι α) : Finset ι := univ.filter fun i => (p i).isSome

/-- A plan is valid when every margin is positive, it keeps the cap, and it picks no excluded alternative. -/
def Valid (L : Limits ι α) (p : Plan ι α) : Prop :=
  (∀ k, 0 < P.margin p k) ∧ (∀ n, L.cap = some n → (changed p).card ≤ n) ∧
    ∀ i a, p i = some a → ¬ L.excluded i a

/-- A cheapest plan: valid, and no valid plan costs less. -/
def Cheapest (L : Limits ι α) (p : Plan ι α) : Prop := P.Valid L p ∧ ∀ q, P.Valid L q → P.cost' p ≤ P.cost' q

/-! ### The integer program of the code -/

/-- A point of the integer program: one 0/1 variable for each alternative of each group. -/
abbrev Point (ι : Type) (α : ι → Type) := ∀ i, α i → Bool

/-- The picks of group `i` in a point. -/
def picks (x : Point ι α) (i : ι) : Finset (α i) := univ.filter fun a => x i a = true

/-- The feasible points: at most one pick in each group, every margin positive, the cap on the number of picks,
and no excluded pick. These are the constraints that `build_model` writes. -/
def Feasible (L : Limits ι α) (x : Point ι α) : Prop :=
  (∀ i, (picks x i).card ≤ 1) ∧
    (∀ k, 0 < P.worst k + ∑ i, ∑ a, if x i a then P.gain i a k else 0) ∧
    (∀ n, L.cap = some n → ∑ i, (picks x i).card ≤ n) ∧
    ∀ i a, x i a = true → ¬ L.excluded i a

/-- The objective of the integer program: the sum of the costs of the picks. -/
noncomputable def objective (x : Point ι α) : ℝ := ∑ i, ∑ a, if x i a then P.cost i a else 0

/-- The point of a plan. -/
def ofPlan (p : Plan ι α) : Point ι α := fun i a => decide (p i = some a)

/-- The plan of a point: the pick of each group, if it has one. -/
noncomputable def toPlan (x : Point ι α) : Plan ι α := fun i =>
  if h : ∃ a, x i a = true then some (Classical.choose h) else none

omit [Fintype ι] in
/-- In a group, the sum over the picks of a plan is the value of its pick. -/
theorem sum_ofPlan (p : Plan ι α) (i : ι) (f : α i → ℝ) :
    (∑ a, if ofPlan p i a then f a else 0) = ((p i).map f).getD 0 := by
  cases h : p i with
  | none => simp [ofPlan, h]
  | some b =>
    rw [Finset.sum_eq_single b]
    · simp [ofPlan, h]
    · intro a _ hab
      simp [ofPlan, h, Ne.symm hab]
    · simp

omit [Fintype ι] in
theorem card_picks_ofPlan (p : Plan ι α) (i : ι) :
    (picks (ofPlan p) i).card = if (p i).isSome then 1 else 0 := by
  cases h : p i with
  | none => simp [picks, ofPlan, h]
  | some b =>
    have : picks (ofPlan p) i = {b} := by
      ext a
      simp [picks, ofPlan, h, eq_comm]
    simp [this]

theorem sum_card_picks_ofPlan (p : Plan ι α) : ∑ i, (picks (ofPlan p) i).card = (changed p).card := by
  simp only [card_picks_ofPlan, changed, Finset.card_filter]

/-- Theorem 1, the plans: a plan is valid iff its point is feasible. -/
theorem ofPlan_feasible_iff (L : Limits ι α) (p : Plan ι α) : P.Feasible L (ofPlan p) ↔ P.Valid L p := by
  simp only [Feasible, Valid, margin, pickGain, sum_ofPlan, sum_card_picks_ofPlan]
  constructor
  · rintro ⟨_, hm, hc, he⟩
    exact ⟨hm, hc, fun i a h => he i a (by simp [ofPlan, h])⟩
  · rintro ⟨hm, hc, he⟩
    refine ⟨fun i => ?_, hm, hc, fun i a h => he i a (by simpa [ofPlan] using h)⟩
    rw [card_picks_ofPlan]
    split_ifs <;> simp

/-- Theorem 1, the plans: the objective of the point of a plan is its cost. -/
theorem objective_ofPlan (p : Plan ι α) : P.objective (ofPlan p) = P.cost' p := by
  simp only [objective, cost', pickCost, sum_ofPlan]

omit [Fintype ι] [∀ i, DecidableEq (α i)] in
theorem eq_of_picks {x : Point ι α} {i : ι} (h1 : (picks x i).card ≤ 1) {a b : α i} (ha : x i a = true)
    (hb : x i b = true) : a = b :=
  Finset.card_le_one.1 h1 a (by simp [picks, ha]) b (by simp [picks, hb])

omit [Fintype ι] in
/-- A point with at most one pick in each group is the point of its plan. -/
theorem ofPlan_toPlan {x : Point ι α} (h1 : ∀ i, (picks x i).card ≤ 1) : ofPlan (toPlan x) = x := by
  classical
  funext i a
  by_cases h : ∃ a, x i a = true
  · have hc := Classical.choose_spec h
    simp only [ofPlan, toPlan, h, dite_true, Option.some.injEq]
    by_cases hxa : x i a = true
    · simp [hxa, eq_of_picks (h1 i) hc hxa]
    · have : Classical.choose h ≠ a := fun he => hxa (he ▸ hc)
      simp [this, hxa]
  · push Not at h
    simp [ofPlan, toPlan, h]

/-- Theorem 1, the points: the plan of a feasible point is valid. -/
theorem toPlan_valid {L : Limits ι α} {x : Point ι α} (hx : P.Feasible L x) : P.Valid L (toPlan x) := by
  rw [← ofPlan_feasible_iff, ofPlan_toPlan hx.1]
  exact hx

/-- Theorem 1, the points: the cost of the plan of a feasible point is its objective. -/
theorem objective_toPlan {L : Limits ι α} {x : Point ι α} (hx : P.Feasible L x) :
    P.cost' (toPlan x) = P.objective x := by
  rw [← objective_ofPlan, ofPlan_toPlan hx.1]

/-- Corollary 1: an optimum of the integer program is a cheapest plan. -/
theorem cheapest_of_optimum {L : Limits ι α} {x : Point ι α} (hx : P.Feasible L x)
    (hopt : ∀ y, P.Feasible L y → P.objective x ≤ P.objective y) : P.Cheapest L (toPlan x) := by
  refine ⟨P.toPlan_valid hx, fun q hq => ?_⟩
  rw [P.objective_toPlan hx, ← objective_ofPlan]
  exact hopt _ ((ofPlan_feasible_iff P L q).2 hq)

/-! ### Theorem 2: a harder problem never has a cheaper plan (contract 7) -/

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- If each valid plan of `(P', L')` is a valid plan of `(P, L)`, a cheapest plan of `(P, L)` costs at most a cheapest
plan of `(P', L')`, and `(P', L')` has no valid plan when `(P, L)` has none. -/
theorem cost_le_of_valid_imp {P' : Problem ι α κ} {L L' : Limits ι α} (hcost : P'.cost = P.cost)
    (himp : ∀ q, P'.Valid L' q → P.Valid L q) {p p' : Plan ι α} (hp : P.Cheapest L p) (hp' : P'.Cheapest L' p') :
    P.cost' p ≤ P'.cost' p' := by
  have h := hp.2 p' (himp p' hp'.1)
  have : P'.cost' p' = P.cost' p' := by simp only [cost', pickCost, hcost]
  linarith

omit [∀ i, DecidableEq (α i)] [∀ i, Fintype (α i)] in
/-- Smaller gains never make a plan valid: a plan of `P'` with gains at most those of `P` and the same worst cases
is a plan of `P`. In the counterfactual, a higher confidence level lowers each gain. -/
theorem valid_of_gain_le {P' : Problem ι α κ} (hworst : P'.worst = P.worst)
    (hgain : ∀ i a k, P'.gain i a k ≤ P.gain i a k) {L : Limits ι α} {p : Plan ι α} (h : P'.Valid L p) :
    P.Valid L p := by
  refine ⟨fun k => ?_, h.2.1, h.2.2⟩
  have hle : ∑ i, P'.pickGain p i k ≤ ∑ i, P.pickGain p i k := by
    refine Finset.sum_le_sum fun i _ => ?_
    cases hp : p i <;> simp [pickGain, hp, hgain]
  have := h.1 k
  simp only [margin, hworst] at this ⊢
  linarith

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- A lower cap and more excluded alternatives never make a plan valid. A smaller allowed set and a new immutable
feature exclude more alternatives. -/
theorem valid_of_limits_le {L L' : Limits ι α} (hcap : ∀ n, L.cap = some n → ∃ n', L'.cap = some n' ∧ n' ≤ n)
    (hexc : ∀ i a, L.excluded i a → L'.excluded i a) {p : Plan ι α} (h : P.Valid L' p) : P.Valid L p := by
  refine ⟨h.1, fun n hn => ?_, fun i a hp he => h.2.2 i a hp (hexc i a he)⟩
  obtain ⟨n', hn', hle⟩ := hcap n hn
  exact (h.2.1 n' hn').trans hle

/-! ### Theorem 3: diverse plans -/

/-- The limits after the plans `ps`: each pick (group, alternative) of an earlier plan is excluded, as the code of
GAM Coach removes the variable of each such pick. -/
def excludeUsed (L : Limits ι α) (ps : List (Plan ι α)) : Limits ι α where
  cap := L.cap
  excluded i a := L.excluded i a ∨ ∃ q ∈ ps, q i = some a

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- Theorem 3 (a): a valid plan after the plans `ps` keeps the limits, and it uses no pick of an earlier plan. -/
theorem valid_excludeUsed_new {L : Limits ι α} {ps : List (Plan ι α)} {p : Plan ι α}
    (h : P.Valid (excludeUsed L ps) p) :
    P.Valid L p ∧ ∀ q ∈ ps, ∀ i a, p i = some a → q i ≠ some a := by
  refine ⟨⟨h.1, h.2.1, fun i a hp he => h.2.2 i a hp (Or.inl he)⟩, fun q hq i a hp hqa => ?_⟩
  exact h.2.2 i a hp (Or.inr ⟨q, hq, hqa⟩)

omit [∀ i, Fintype (α i)] [∀ i, DecidableEq (α i)] in
/-- Theorem 3 (b): the costs of the diverse plans do not go down. Plan `k + 1` is a cheapest plan after the plans up
to `k`, and plan `k` is a cheapest plan after the plans before it. -/
theorem cost_mono_excludeUsed {L : Limits ι α} {ps : List (Plan ι α)} {p p' : Plan ι α}
    (hp : P.Cheapest (excludeUsed L ps) p) (hp' : P.Cheapest (excludeUsed L (ps ++ [p])) p') :
    P.cost' p ≤ P.cost' p' :=
  P.cost_le_of_valid_imp (L := excludeUsed L ps) (L' := excludeUsed L (ps ++ [p])) rfl
    (fun q hq => P.valid_of_limits_le (L := excludeUsed L ps) (L' := excludeUsed L (ps ++ [p]))
      (fun n hn => ⟨n, hn, le_rfl⟩)
      (fun i a he => by
        rcases he with he | ⟨r, hr, hra⟩
        · exact Or.inl he
        · exact Or.inr ⟨r, List.mem_append_left [p] hr, hra⟩) hq) hp hp'

end Problem

end EBaMlProofs.Recourse
