# Plan: complete the proof of the hardness of the sufficient reason

You are a senior researcher in formal proofs with Lean 4 and in computational complexity. You will close every gap that
the paper `proofs/papers/local-explanations/paper.typ` lists for its section "The sufficient reason of several margins
is NP-hard". Read `CLAUDE.md` and every file in `.claude/rules/` first, with special care for
`.claude/rules/proofs.md`, because the project's hooks enforce those rules and `prek run --all-files` must pass before
you finish.

This plan is a follow-up that belongs to no milestone of `ROADMAP.md`. The user asked for it so that the proof can
reach full detail later. Each stage below stands alone, so the user can ask for one stage at a time.

<goal>
Every statement of the hardness section is checked in Lean or is a named premise of a Lean theorem. A reader can then
see exactly which published results the conclusion trusts, and nothing else is assumed. The published results that
stay as premises are Karp's theorem, Corollary 1.5 of Dinur and Steurer, and the algorithms of Chvátal and of
Kolliopoulos and Young.
</goal>

<context>
- `proofs/EBaMlProofs/LocalExplanations/Hardness.lean` holds the reduction from Set Cover. Theorem 8
  (`SetCover.sufficient_iff_cover`, `SetCover.min_size_iff`) says that the sufficient sets of the reduction are the
  covers. Theorem 9 (`sufficient_reason_iff_cover`) builds a multiclass classifier with this problem as its sufficient
  reason. Theorem 10 (`SetCover.isOptimum_iff`, `SetCover.isApprox_iff`, and the `multiclass_` versions) says that the
  two problems have the same optimum and the same `ρ`-approximate sets.
- `proofs/EBaMlProofs/LocalExplanations/Covering.lean` holds the covering problem (`Problem`, `Sufficient`), and
  `Multiclass.lean` holds the margin models of a multiclass classifier.
- The paper argues these points in prose, and Lean does not check them. Its section "Limits" lists each one:
  1. Corollary 4: the decision problem is in NP, and the reduction runs in polynomial time.
  2. Corollary 5: the logic that carries the hardness of approximation of Dinur and Steurer to the sufficient reason.
  3. The reading of the size `n` in Corollary 1.5 of Dinur and Steurer as the number of elements.
  4. The counterfactual of a chosen class, which the code computes in `_target_counterfactual` and which the Lean model
     does not contain.
  5. A fixed number of margins that is larger than one, about which the paper proves nothing.
  6. The upper bound of the approximation ratio, which the paper cites and the code does not implement.
- Mathlib has `TM2ComputableInPolyTime` in `Mathlib/Computability/TuringMachine/Computable.lean` and `Encoding` in
  `Mathlib/Computability/Encoding.lean`. It proves only that the identity is computable in polynomial time
  (`idComputableInPolyTime`). It has no class NP, no polynomial-time reduction, no composition lemma for
  polynomial-time functions and no Cook-Levin theorem.
- `.claude/rules/proofs.md` forbids a new `axiom`, so each published result enters as a hypothesis of a theorem.
</context>

<stages>
Do the stages in this order, because each one uses the one before it. Ask the user before you start stage 3, because
it is large.

1. **The logic of Corollaries 4 and 5, with an abstract class of efficient functions.**
   - Define a structure for a class of functions between encoded types: it holds the identity and is closed under
     composition. Do not fix the model of computation, so that the stage is small.
   - Define a `ρ`-approximation algorithm for a property of sets as a function in the class that returns a
     `ρ`-approximate set (`IsApprox`) whenever a set with the property exists.
   - Prove: if the reduction map is in the class, and no function in the class is a `ρ`-approximation for Set Cover,
     then no function in the class is a `ρ`-approximation for the sufficient reason. Prove the same for the classifier
     of Theorem 9. The proof composes the reduction with the algorithm and uses Theorem 10.
   - Prove the same pattern for Corollary 4: with an abstract notion of reduction, NP-hardness of Set Cover and a
     reduction in the class give NP-hardness of the covering problem.
   - Name each premise after its source (`dinur_steurer`, `karp_set_cover`), so that the signature shows what the
     conclusion trusts. The reason: a premise that is stated wrongly, for example without a time bound, can be false,
     and then the theorem says nothing. A reader must be able to find and check each premise.
2. **The meaning of the size in Dinur and Steurer.** Read Corollary 1.5 and its proof in Section 6.2 of the paper
   (arXiv:1305.1979). Fix one encoding of a Set Cover instance, and write in the paper how `ln n` for that encoding
   relates to `ln` of the number of elements. If the two differ by more than the factor `1 - α` absorbs, state the
   hardness of Corollary 5 in the size that Dinur and Steurer use, and correct Claim 7 and the docs page.
3. **Instantiate the class with polynomial time.** This stage is large: a proof of running time on a Turing machine
   usually takes far more lines than the idea.
   - Prove that the functions of `TM2ComputableInPolyTime` are closed under composition, if Mathlib still lacks it.
   - Fix the encodings of a Set Cover instance and of a covering problem with rational numbers.
   - Prove that the reduction of Theorem 8 is computable in polynomial time, and that the check of a certificate (a set
     of features) is computable in polynomial time. The second fact is the membership in NP.
   - If a library of polynomial-time reductions exists for Lean 4 and Mathlib at the pinned version, prefer it, and
     ask the user before you add a dependency.
4. **The counterfactual of a chosen class.** Model `_target_counterfactual` in Lean: the margins of the target class,
   the choice of one bin for each feature, and the covering problem after the choice. Prove two facts:
   - The problem contains the covering problem, so it is NP-hard and has the hardness of Corollary 5. Use the case in
     which each feature has one bin besides the bin of the instance.
   - The choice of the bin with the largest sum of gains (`_best_option`) can miss the smallest set. State the
     counterexample of `test_a_target_counterfactual_is_not_always_the_smallest_set` with its numbers, and prove it
     with `norm_num` or `decide`.
5. **A fixed number of margins.** This is an open question for this project. Find out whether the covering problem
   with two margins is NP-hard, or whether an algorithm in polynomial time solves it. Search the literature on
   covering integer programs with a fixed number of constraints and a cardinality objective. Report the answer to
   the user before you prove anything, because the answer decides whether a proof or an algorithm follows.
6. **The upper bound.** Do this stage only if the code implements an algorithm with a guarantee, because a proof of an
   algorithm that the code does not run proves nothing about the code. If the user chooses the greedy rule by marginal
   coverage for `Minimal`, prove its ratio (the bound of Wolsey for submodular cover) for gains that are integers after
   a scaling. Keep the result of Kolliopoulos and Young as a citation.
</stages>

<tests>
- `lake build` in `proofs/` passes with no `sorry`, and `Axioms.lean` shows only `propext`, `Classical.choice` and
  `Quot.sound` for each theorem that it prints. Add a `#print axioms` line for each new main theorem.
- Each published result is a hypothesis, never an `axiom`. A search for `axiom` in `proofs/EBaMlProofs` finds only the
  `#print axioms` lines.
- For stage 4, a unit test runs the counterexample of the Lean theorem on `_target_counterfactual`. The test
  `test_a_target_counterfactual_is_not_always_the_smallest_set` already does this, so keep its numbers and the Lean
  theorem the same.
- For stage 6, a unit test compares the greedy rule with `ExactOrMinimal` on small instances and checks the ratio.
</tests>

<documentation>
- Update the section "The sufficient reason of several margins is NP-hard" of the paper for each stage. Name the Lean
  theorem next to each statement, and remove each item of "Limits" that a stage closes.
- Update the commit in the appendix of the paper to the commit at which the Lean files were checked, as
  `.claude/rules/proofs.md` says, and add a row for each new statement to the table of the appendix.
- Update the sections "What is proved" and "What is not proved" of `docs/local-explanations-proof.md`.
- Render the changed pages of the paper and look at them, as step 7 of `.claude/rules/proofs.md` says.
- Write all prose in the ASD-STE100 style of `.claude/rules/writing-style.md`.
</documentation>

<release>
A stage that changes only `proofs/` and the docs has no user-visible change, so it needs no changelog fragment. A stage
that changes `src/ebaml/` (stage 6) adds a fragment if the behavior of `Minimal` changes. Give each stage its own pull
request, because each one is reviewed on its own.
</release>

<done>
- The section "Limits" of the paper lists no gap of the hardness section, except the published results that stay as
  named premises and the questions that the user chose to leave open.
- `prek run --all-files`, `uv run pytest`, the strict docs build and the proof build pass.
</done>
