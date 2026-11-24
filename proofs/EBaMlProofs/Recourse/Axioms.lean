import EBaMlProofs.Recourse.Plans
import EBaMlProofs.Recourse.Counterfactual
import EBaMlProofs.Recourse.Target
import EBaMlProofs.Recourse.Reduction
import EBaMlProofs.Recourse.Bounds

/-!
# Axiom check

`lake build` prints the axioms of the main theorems of the recourse plans.
The expected output for each is only `propext`, `Classical.choice` and `Quot.sound`.
-/

#print axioms EBaMlProofs.Recourse.Problem.ofPlan_feasible_iff
#print axioms EBaMlProofs.Recourse.Problem.objective_ofPlan
#print axioms EBaMlProofs.Recourse.Problem.toPlan_valid
#print axioms EBaMlProofs.Recourse.Problem.objective_toPlan
#print axioms EBaMlProofs.Recourse.Problem.cheapest_of_optimum
#print axioms EBaMlProofs.Recourse.Problem.cost_le_of_valid_imp
#print axioms EBaMlProofs.Recourse.Problem.valid_of_gain_le
#print axioms EBaMlProofs.Recourse.Problem.valid_of_limits_le
#print axioms EBaMlProofs.Recourse.Problem.valid_excludeUsed_new
#print axioms EBaMlProofs.Recourse.Problem.cost_mono_excludeUsed
#print axioms EBaMlProofs.Recourse.margin_cfProblem
#print axioms EBaMlProofs.Recourse.valid_cfProblem_iff
#print axioms EBaMlProofs.Recourse.sorting_cheapest
#print axioms EBaMlProofs.Recourse.valid_of_higher_confidence
#print axioms EBaMlProofs.Recourse.valid_embed
#print axioms EBaMlProofs.Recourse.cost_embed
#print axioms EBaMlProofs.Recourse.sorting_not_cheapest
#print axioms EBaMlProofs.Recourse.knapsack_canFlipOn_iff
#print axioms EBaMlProofs.Recourse.Multiclass.margin_targetProblem
#print axioms EBaMlProofs.Recourse.Multiclass.valid_targetProblem_iff
#print axioms EBaMlProofs.Recourse.Multiclass.reaches_of_valid
#print axioms EBaMlProofs.Recourse.Multiclass.plan_of_reaching
#print axioms EBaMlProofs.Recourse.Multiclass.exists_plan_iff
#print axioms EBaMlProofs.Recourse.Multiclass.smallest_plan_iff
#print axioms EBaMlProofs.Recourse.Problem.valid_erase
#print axioms EBaMlProofs.Recourse.Problem.valid_swap
#print axioms EBaMlProofs.Recourse.Problem.exists_kept
#print axioms EBaMlProofs.Recourse.Problem.exists_kept_plan
#print axioms EBaMlProofs.Recourse.Problem.exists_plan_iff_kept
#print axioms EBaMlProofs.Recourse.Problem.smallest_plan_iff_kept
#print axioms EBaMlProofs.Recourse.Problem.lp_lower_bound
#print axioms EBaMlProofs.Recourse.Problem.no_plan_of_relaxation_infeasible
#print axioms EBaMlProofs.Recourse.Problem.card_mul_le_cost
#print axioms EBaMlProofs.Recourse.Problem.cost_lower_bound
#print axioms EBaMlProofs.Recourse.Problem.one_le_card
