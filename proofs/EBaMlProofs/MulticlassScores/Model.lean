import EBaMlProofs.MulticlassScores.Softmax
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# The graphs of a multiclass EBM keep its predictions

This file formalizes the scores of a multiclass graph of `EBMGraph.from_ebm` in `src/ebaml/graphs/ebm_graph.py`
(the centring of `_select_outputs`, then the normalization) and the prediction of `MulticlassClassifier.predict` in
`src/ebaml/graphs/model_kinds.py`. It applies the abstract result of `EBaMlProofs.MulticlassScores.Softmax` to them.

* `Model` and the definitions that follow it are Section 2 of the paper.
* `score_shift` is Lemma 2, and `predict_shift` is Theorem 3.
* `graph_mean`, `graph_sub` and `predict_graph` are Theorem 4.
-/

open Finset

namespace EBaMlProofs.MulticlassScores

/-- A multiclass EBM and the instance to explain. The model has the classes `κ` and the terms `τ`.
Term `t` has the bins `B t`, and the instance falls in the bin `c t`. A term may be a feature or an interaction:
the proof needs only that each term adds one score per class. -/
structure Model (τ κ : Type) (B : τ → Type) where
  /-- The intercept of each class. -/
  b : κ → ℝ
  /-- The score of each bin of each term, for each class. -/
  s : ∀ t, B t → κ → ℝ
  /-- The bin of the instance in each term. -/
  c : ∀ t, B t

namespace Model

variable {τ κ : Type} [Fintype τ] [Fintype κ] {B : τ → Type} (m : Model τ κ B)

/-- The score of class `k` for the instance: the intercept plus the score of the bin of each term. -/
noncomputable def score (k : κ) : ℝ := m.b k + ∑ t, m.s t (m.c t) k

/-- The predicted probabilities: softmax of the scores. -/
noncomputable def predict : κ → ℝ := Softmax.softmax m.score

/-- The model whose scores are those of `m` minus `d t v` for every class of bin `v` of term `t`. -/
def shift (d : ∀ t, B t → ℝ) : Model τ κ B := { b := m.b, s := fun t v k => m.s t v k - d t v, c := m.c }

omit [Fintype κ] in
/-- Lemma 2. A shift of each bin moves the score of every class by the same constant. -/
theorem score_shift (d : ∀ t, B t → ℝ) (k : κ) : (m.shift d).score k = m.score k + -∑ t, d t (m.c t) := by
  simp only [score, shift, Finset.sum_sub_distrib]
  ring

/-- Theorem 3. A shift of each bin by one constant for all the classes keeps the predicted probabilities. -/
theorem predict_shift (d : ∀ t, B t → ℝ) : (m.shift d).predict = m.predict := by
  rw [predict, predict, show (m.shift d).score = fun k => m.score k + -∑ t, d t (m.c t) from
    funext (m.score_shift d), Softmax.softmax_add_const]

/-- The mean of the scores of bin `v` of term `t` over all the classes of the model. -/
noncomputable def mean (t : τ) (v : B t) : ℝ := (∑ j, m.s t v j) / Fintype.card κ

/-- The scores of the graphs of `m` with the normalization constant `a`: each bin is centred over all the classes,
and then every score is lowered by `a`. -/
noncomputable def graph (a : ℝ) : Model τ κ B := m.shift fun t v => m.mean t v + a

omit [Fintype τ] in
/-- Theorem 4 (a). In every bin of every graph, the mean of the scores over all the classes is `-a`. -/
theorem graph_mean [Nonempty κ] (a : ℝ) (t : τ) (v : B t) : (m.graph a).mean t v = -a := by
  have hn : (Fintype.card κ : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [mean, graph, shift, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

omit [Fintype τ] in
/-- Theorem 4 (b). The graphs keep the difference between the scores of two classes in every bin. -/
theorem graph_sub (a : ℝ) (t : τ) (v : B t) (k j : κ) :
    (m.graph a).s t v k - (m.graph a).s t v j = m.s t v k - m.s t v j := by
  simp only [graph, shift]
  ring

/-- Theorem 4 (c). The graphs predict the same probabilities as the model. -/
theorem predict_graph (a : ℝ) : (m.graph a).predict = m.predict :=
  m.predict_shift _

end Model

end EBaMlProofs.MulticlassScores
