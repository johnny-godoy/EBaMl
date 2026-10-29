import EBaMlProofs.MulticlassScores.Model

/-!
# Axiom check

`lake build` prints the axioms of the main theorems.
The expected output for each is only `propext`, `Classical.choice` and `Quot.sound`.
-/

#print axioms EBaMlProofs.MulticlassScores.Softmax.softmax_add_const
#print axioms EBaMlProofs.MulticlassScores.Softmax.softmax_eq_iff
#print axioms EBaMlProofs.MulticlassScores.Model.predict_shift
#print axioms EBaMlProofs.MulticlassScores.Model.graph_mean
#print axioms EBaMlProofs.MulticlassScores.Model.graph_sub
#print axioms EBaMlProofs.MulticlassScores.Model.predict_graph
