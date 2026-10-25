import EBaMlProofs.LocalExplanations.Model

/-!
# Axiom check

`lake build` prints the axioms of the main theorems.
The expected output for each is only `propext`, `Classical.choice` and `Quot.sound`.
-/

#print axioms EBaMlProofs.LocalExplanations.Greedy.threshold_minimal
#print axioms EBaMlProofs.LocalExplanations.Model.min_change
#print axioms EBaMlProofs.LocalExplanations.Model.min_change_none
#print axioms EBaMlProofs.LocalExplanations.Model.chosen_bin_differs
#print axioms EBaMlProofs.LocalExplanations.Model.min_sufficient_reason
#print axioms EBaMlProofs.LocalExplanations.Model.min_sufficient_reason_none
