import EBaMlProofs.LocalExplanations.Model
import EBaMlProofs.LocalExplanations.Regressor
import EBaMlProofs.LocalExplanations.Multiclass
import EBaMlProofs.LocalExplanations.Hardness

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
#print axioms EBaMlProofs.LocalExplanations.Model.shift_eps
#print axioms EBaMlProofs.LocalExplanations.Model.shift_margin_eq
#print axioms EBaMlProofs.LocalExplanations.Model.shift_robustMargin_neg_iff
#print axioms EBaMlProofs.LocalExplanations.Model.canFlipInside_iff
#print axioms EBaMlProofs.LocalExplanations.Model.min_change_inside
#print axioms EBaMlProofs.LocalExplanations.Model.min_change_inside_none
#print axioms EBaMlProofs.LocalExplanations.problemOfModels_sufficient
#print axioms EBaMlProofs.LocalExplanations.Covering.Problem.sufficient_mono
#print axioms EBaMlProofs.LocalExplanations.Covering.Problem.prune_minimal
#print axioms EBaMlProofs.LocalExplanations.Covering.Problem.exact_minimum
#print axioms EBaMlProofs.LocalExplanations.Covering.Problem.card_search_space
#print axioms EBaMlProofs.LocalExplanations.Covering.Problem.minimum_of_card_le_one
#print axioms EBaMlProofs.LocalExplanations.Multiclass.logit_marginModel
#print axioms EBaMlProofs.LocalExplanations.Multiclass.winner_iff
#print axioms EBaMlProofs.LocalExplanations.Multiclass.marginModel_eps
#print axioms EBaMlProofs.LocalExplanations.Multiclass.canFlipClass_iff
#print axioms EBaMlProofs.LocalExplanations.Multiclass.min_change_class
#print axioms EBaMlProofs.LocalExplanations.Multiclass.min_change_class_none
#print axioms EBaMlProofs.LocalExplanations.Multiclass.sufficient_class_iff
#print axioms EBaMlProofs.LocalExplanations.sd_sum_bound
#print axioms EBaMlProofs.LocalExplanations.Hardness.SetCover.half_lt_sum_iff
#print axioms EBaMlProofs.LocalExplanations.Hardness.SetCover.sufficient_iff_cover
#print axioms EBaMlProofs.LocalExplanations.Hardness.SetCover.min_size_iff
#print axioms EBaMlProofs.LocalExplanations.Hardness.sufficient_of_shape
#print axioms EBaMlProofs.LocalExplanations.Hardness.sufficient_reason_iff_cover
