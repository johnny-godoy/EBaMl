# Plan: actionable recourse (0.14.0)

You are a senior engineer in optimization and formal proofs, with experience in Lean 4, mixed-integer programming and
algorithmic recourse. You will implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and
every file in `.claude/rules/` first, with special care for `.claude/rules/semantic-contracts.md`,
`.claude/rules/proofs.md`, `.claude/rules/solid.md` and `.claude/rules/reuse.md`, because the project's hooks enforce
those rules and `prek run --all-files` must pass before you finish.

<goal>
The counterfactual of a local explanation becomes a recourse plan that a person can carry out. Today it is the smallest
set of changes, with every feature equally easy to change and every plausible bin allowed. After this milestone, the
caller can say which features must not change, which values each feature may take, how hard each change is, and how many
different plans to return. Python still computes every fact exactly under the stated assumptions, and a Lean proof
backs every exact claim, including the claim that each old theorem still holds in the general setting.

The ideas come from GAM Coach (Wang et al., "GAM Coach: Towards Interactive and User-centered Algorithmic Recourse",
CHI 2023, arXiv:2302.14165, MIT license). Reimplement them on the covering problem that EBaMl has. Do not copy its
code, and cite the paper in the docs and in the paper's `references.yml`.
</goal>

<decisions>
The user made these choices. Do not reopen them.

1. The milestone lands before 1.0.0, so 1.0.0 freezes the result types that this milestone settles.
2. The scope has four features: immutable features, allowed values, weighted costs, and several diverse plans.
3. Every way to get the costs is a strategy that the caller picks: unit costs (the fewest changes, as today), costs from
   the training data, costs from the user, costs that a BAML helper proposes, and costs that the MCP agent passes.
4. The minimum-cost claim gets a Lean proof in this milestone.
5. No proof is assumed to carry over to the general setting. Each theorem of `proofs/EBaMlProofs/LocalExplanations/`
   that the change touches is restated for the general model and proved again, and the old statement is derived from
   the new one as a corollary in Lean. A comment such as "the same argument applies" is not a proof.
</decisions>

<context>
- `src/ebaml/local/counterfactuals.py` has two searches. `_counterfactual` handles a decision with one margin: it
  sorts the features by their largest robust drop and takes the shortest prefix that flips the decision. The proof is
  `min_change` in `Model.lean`, and the result has `MinimalityStatus.SORTING_PROOF`. `_target_counterfactual` handles a
  target class of a multiclass classifier: it builds a `CoveringProblem` (`covering_problem.py`) and solves it with
  `options.search` (`MipCover`, `GreedyCover`, `LocalSearchCover` in `covering_solvers.py`, OR-Tools through
  `covering_backend.py` and `ortools_backend.py`).
- The proofs model "the realistic values" of a feature as a set `P i` (`Model.lean`), which the code builds from
  `min_support` (`_plausible_bins` in `fact_features.py`). The sufficient reason and the counterfactual share this set.
- The paper `proofs/papers/local-explanations/paper.typ` lists the counterfactual of a chosen class as outside the Lean
  model (`docs/plans/hardness-proof-completion.md`, limit 4). This milestone must close that gap, because the weighted
  search runs on the same covering problem. Update limit 4 of that plan when you close it.
- `MinimalityStatus` (`minimality.py`) and `Counterfactual.lower_bound: int` speak of the smallest set. The prompts read
  only `MinimalityStatus.is_smallest`. `LocalFacts` holds one counterfactual. The MCP tool `explain_prediction`
  (`mcp/explain_tools.py`, `PredictionRequest`) has no recourse options.
- With weighted costs and one margin, the problem is a multiple-choice knapsack problem, which is NP-hard. Sorting then
  proves nothing, so the one-margin case goes through the MIP. Unit costs keep the sorting search and its proof.
- The local explanations reject a model with interaction terms. This milestone does not change that.
- Before 1.0.0, a breaking change is allowed. Mark it **BREAKING** in the fragment.
</context>

<design>
Settle these points in the first pull request, and ask the user only where the text says so.

- **Two sets of values, not one.** Keep `P i` (the realistic values) for the sufficient reason and the completions,
  because they describe what the world can be. Add the allowed values `A i ⊆ P i` for the counterfactual only, because
  they describe what the person can do. An immutable feature has `A i = ∅`. Allowed values never change a sufficient
  reason, and a test shows it.
- **Options.** Put the recourse options in one `@attrs.frozen` class `RecourseOptions`, a field of `LocalOptions`, so
  that `LocalOptions` stays small (`solid.md`, I). Its fields:
  - `immutable: Collection[str]`
  - `allowed: Mapping[str, AllowedValues]` (an interval for a continuous feature, a set of levels for a categorical one)
  - `costs: CostStrategy`, by default `UnitCost()`
  - `max_changes: int | None`, a cap on the number of changed features. GAM Coach has it, and a cheapest plan without a
    cap can change many cheap features.
  - `plans: int = 1`
  Reject a name that is not a feature of the model, and an interval that leaves no allowed bin, before any search.
- **Cost strategies.** Add a `Protocol` `CostStrategy` in `ebaml.local`, with one method that returns a cost for each
  feature and each allowed bin from the `Feature` views. Search the registry first (`reuse.md`), then register it in
  `architecture/abstractions.yml`. Each cost is finite and strictly positive, so a free change cannot pad a plan.
  Validate this in Python. One class for each strategy (`solid.md`, O):
  - `UnitCost`: every change costs 1. This is the fewest-changes counterfactual that exists today.
  - `DataCost`: for a continuous feature, the distance from the value of the instance to the nearest value of the bin,
    divided by the median absolute deviation of the feature. For a categorical feature, `1 - share` of the level.
    Then scale the categorical costs so that their mean equals the mean of the continuous costs, as GAM Coach does.
    Compute the median and the deviation from the training histogram (`bin_weights_`) that the EBM holds, and say in the
    docstring and in the facts that they are estimates from binned data.
  - `UserCost`: a cost for each feature, or for each feature and bin, from the caller.
  - `ScaledCost`: a difficulty multiplier for each feature on top of another strategy. GAM Coach uses this for "easy" or
    "hard". "Impossible" is `immutable`, not an infinite cost.
  - The BAML helper: a BAML function `ProposeFeatureDifficulty` in `baml_src/local_explanations.baml` (or a new
    source file with its own test file), called only from `llm/descriptors.py`. It returns, for each feature, a
    difficulty from a fixed scale and a reason. A public function in `ebaml.llm` turns the reply into a `ScaledCost`
    and returns it with the reasons. It never runs a search: the caller reads the proposal and passes it on. Write
    the prompt to Anthropic's guide (`writing-style.md`).
  - The MCP agent: `explain_prediction` accepts `immutable`, `allowed`, `difficulty` (feature to multiplier),
    `cost_strategy` (`unit` or `data`, the base under the multipliers), `max_changes` and `plans`. Add a tool
    `propose_difficulties` that calls the BAML helper, so an agent without its own judgement can ask for a proposal
    and show it to the user.
- **Ties.** Two plans of equal cost: prefer fewer changes, then the lower feature positions, so a run is reproducible.
  Solve the minimum cost first, then the fewest changes at that cost. Write the float tolerance next to the code, and
  write it in the paper's limits.
- **Diverse plans.** The user chose the precedent of GAM Coach, as its code implements it (`generate_cfs` and
  `create_milp` in `gamcoach/gamcoach.py`): after each plan, remove from the MIP the variable of every pair (feature,
  bin) that the plan uses. Plan `k + 1` is then the cheapest plan that uses no pair (feature, bin) of an earlier plan.
  It may change a feature of an earlier plan to another bin, and it may share features with an earlier plan, but never
  the same bin of the same feature. For example, if plan 1 moves income to `[40k, 50k)`, plan 2 may move income to
  `[50k, 60k)`, but no later plan moves income to `[40k, 50k)`. The paper of GAM Coach (appendix A.8) describes a
  weaker rule, which forbids only the whole earlier solution. Follow the code, and write the difference in the paper
  of the milestone. In the MIP, fix each removed variable to 0, so the problem keeps one variable for each allowed bin.
  A search that proves no further plan ends the list early. GAM Coach skips an iteration with no optimal solution, so
  it can return fewer plans: report the status of each plan instead (`<design>`, "Guarantee names").
- **Guarantee names.** Rename the claim from "smallest" to "optimal" where costs apply:
  - `MinimalityStatus.is_smallest` becomes `is_optimal`.
  - `lower_bound: int` becomes `cost_lower_bound: float`.
  - `Counterfactual` gains `cost: float` and the costs it used.
  - `LocalFacts.counterfactual` becomes `counterfactuals: list[Counterfactual]`, cheapest first. An empty list means
    that the search proves that no plan exists under the stated options.
  - Add the status `SOLVER_PROOF` for a weighted one-margin search, and keep `SORTING_PROOF` for unit costs only.
  - Each plan carries its own status, because the time limit of the MIP can end during any plan.
  Mark each rename **BREAKING**.
- **The text.** The facts name the strategy, the immutable features, the allowed values and the cap. The prompts say
  "the cheapest plans under the difficulties that were given" for weighted costs, and "the fewest changes" for unit
  costs. A difficulty that an LLM proposed is an assumption, and the text says so (contracts 3 and 7).
</design>

<proofs>
Do the proof work before the code that relies on it, and commit the Lean files before the paper (`proofs.md`). Add
the general model to `LocalExplanations`, and put the cost results in a new topic `Recourse`, with the paper
`proofs/papers/recourse/paper.typ`.

1. **Restate the old model with allowed values.** In `Model.lean`, add `A i ⊆ P i` and define the robust drop over
   `A i`. Prove again, for the general model, every theorem that uses the drop or `CanFlip`: `canFlip_iff`,
   `min_change`, `min_change_none`, `chosen_bin_differs`, and the versions in `Multiclass.lean` and `Regressor.lean`.
   Then prove in Lean that `A = P` gives the old definitions, and derive each old theorem from the new one, so that
   the reduction is checked, not assumed. Prove that the sufficient reason theorems do not depend on `A`.
2. **Model the counterfactual of a chosen class.** Add to the Lean model the covering problem that
   `_target_counterfactual` builds: groups of alternatives, at most one alternative per group, every margin positive.
   Prove that its solutions are exactly the sets of changes that reach the target class under the robust scores.
   Stage 4 of `docs/plans/hardness-proof-completion.md` models the same problem. If it is on `main`, build on its
   definitions. Do not add a second model of the same problem.
3. **Weighted costs.** Define the cost of a plan as the sum of the costs of the chosen bins, with costs strictly
   positive. Prove:
   - The integer program of the code (one 0/1 variable for each allowed bin, at most one bin for each feature, the
     margin constraints, the cap and the exclusions of the earlier plans) has as its feasible points exactly the valid
     plans, and as its objective exactly the cost. Its optimum is then the cheapest plan.
   - With unit costs, the sorting search returns a cheapest plan. This is the theorem that keeps `SORTING_PROOF`.
     Derive it from the weighted model. Do not reuse `min_change` without the link.
   - With weighted costs and one margin, the bin with the largest drop is not always a bin of a cheapest plan. Give a
     counterexample in Lean, so the paper shows why the sorting search is not used.
   - The weighted problem with one margin is NP-hard, by a reduction from the knapsack problem. Karp's theorem enters
     as a named premise, as in `Hardness.lean`.
4. **Monotonicity (contract 7).** Prove that a higher confidence level never lowers the cheapest cost and never makes an
   infeasible plan feasible. Prove the same for a smaller allowed set, a new immutable feature and a lower cap.
5. **Diverse plans.** Prove that each plan is valid, that plan `k + 1` is the cheapest plan that uses no pair
   (feature, bin) of an earlier plan, and that the costs of the plans do not go down.
6. **Lower bound.** If the code reports `AT_LOWER_BOUND` from an LP relaxation, prove that the relaxation bounds the
   cheapest cost from below. The bound itself rests on the LP solver, as in the paper today.
7. Run `#print axioms` on each main theorem in `Axioms.lean`. Update the correspondence table of the
   local-explanations paper for every changed definition, write the recourse paper from the template, and render
   every page (`proofs.md`, step 7).
</proofs>

<steps>
Split the work into pull requests that pass every check alone. A suitable order:

1. **The oracle.** Extend the oracle in `tests/unit/ebaml/local/conftest.py` with an exhaustive search over the
   allowed bins on models with up to four features, for weighted costs, the cap and the exclusions. Add
   `test_each_plan_matches_the_oracle` to `test_counterfactuals.py`. It fails until the code exists, so land it with
   step 3. The oracle is a contract change, so show it to the user first (`semantic-contracts.md`, evidence step 4).
2. **The proofs**, as in `<proofs>`, one stage per pull request where a stage is large.
3. **Allowed values and immutable features.** Add `RecourseOptions` with `immutable` and `allowed`, and filter the
   alternatives of both searches. Add a test that the sufficient reason is the same with and without the options.
4. **Weighted costs.** Add `CostStrategy`, `UnitCost`, `DataCost`, `UserCost` and `ScaledCost`. Add the cost
   objective to `CoveringProblem`, to `build_model` in `covering_backend.py` and to the solvers. `GreedyCover` and
   `LocalSearchCover` must report `UNPROVED` for a weighted problem unless they prove more. Send a weighted one-margin
   problem to `options.search`, and keep the sorting search for unit costs. Add the cap.
5. **Diverse plans.** Fix the variables of the pairs (feature, bin) of the earlier plans to 0, and add `plans`. Return the list cheapest first, and stop early with
   the plans found when the search proves that no further plan exists.
6. **Result types and prompts.** Rename the guarantee fields, change `LocalFacts`, and update `fact_text.py` and the
   counterfactual prompt in `baml_src/local_explanations.baml`. Add offline BAML tests that fail on the old wording:
   the text names the costs as given, names the immutable features, and claims "cheapest" only when `is_optimal` holds.
   Run `baml generate`, `baml fmt`, `baml check`, the offline tests and the live tests.
7. **The BAML helper.** Add `ProposeFeatureDifficulty`, its row in `CALLED` of `tests/contract/test_sdk_calls.py`, its
   offline tests, a live test, and the public function in `ebaml.llm` with its `model=` argument.
8. **The MCP tools.** Extend `PredictionRequest`, add `propose_difficulties`, and update the tool snapshot,
   `tests/api/test_mcp_tools.py` and `test_the_tools_name_or_reject_a_model_that_they_cover_in_part`.
9. **Docs and notebook.** Add a section "Recourse plans" to `docs/local-explanations.md`, a proof page
   `docs/recourse-proof.md` with its iframe, and a marimo notebook `notebooks/recourse.py` (`notebooks.md`). The
   notebook shows one section for each cost strategy with the same instance, so the reader compares the plans, and it
   puts the BAML helper behind a button. Register both pages in `zensical.toml` and the notebook in
   `scripts/docs/build.py`. Update the README table "Supported models".
</steps>

<evidence>
For each contract that the change touches, name the guard and add a test (`semantic-contracts.md`):

| Contract | Guard to add or extend |
| --- | --- |
| 1 | `test_each_plan_matches_the_oracle`, and the oracle test of the facts with recourse options. |
| 2 | A test that each plan reports the status of its own search, and that a weighted one-margin plan never reports `SORTING_PROOF`. |
| 3 | Offline BAML tests of the counterfactual prompt, and a test of `fact_text.py` that the text names the costs and the immutable features. |
| 5 | A test that each plan flips the decision of the fitted EBM and changes no immutable feature and no value outside `allowed`. |
| 7 | A property test that a higher confidence level, a smaller allowed set and a lower cap never lower the cheapest cost. |

Report each guarantee that no test or proof shows.
</evidence>

<release>
This PR completes the milestone: it sets `version = "0.14.0"` in `pyproject.toml` and runs `uv lock`. Earlier pull
requests of the milestone each add their own fragment. The fragments list the recourse options, the cost strategies,
the BAML helper and the MCP tools under `Added`, and each renamed field under `Changed` with **BREAKING**. Mark the 0.14.0
row of `ROADMAP.md` as done, and delete this plan.
</release>

<done>
- A caller can pass immutable features, allowed values, a cost strategy, a cap and a number of plans, from Python and
  from the MCP server, and every cost strategy of `<decisions>` exists.
- Each plan agrees with the exhaustive oracle on small models, and each reports the guarantee that its search proved.
- Every theorem that the change touches is proved again in the general model, each old statement is a corollary in
  Lean, the counterfactual of a chosen class is in the Lean model, and `lake build` passes with the expected axioms.
- The texts name the costs, the immutable features and the allowed values, and claim "cheapest" only with a proof.
- `prek run --all-files`, `uv run pytest`, mutation testing of the changed lines, `(cd proofs && lake build)` and the
  strict docs build pass.
</done>
