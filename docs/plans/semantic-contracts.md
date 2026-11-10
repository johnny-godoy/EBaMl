# Plans: semantic contracts

`.claude/rules/semantic-contracts.md` lists seven contracts that keep the code, the proofs, the result types, the
prompts and the docs in agreement about each explanation. The rule states them. These plans add the checks that
enforce them. Each plan is a handoff prompt for one pull request, and none of them is a `ROADMAP.md` milestone.

The plans come from an outside review, checked against the code on 2026-11-09. Each plan starts with what the code
does now, so the implementing agent knows which parts of the review were already done and which are real gaps.

| Order | Plan | Contract | Main gap found |
| --- | --- | --- | --- |
| 2 | [confidence-semantics](confidence-semantics.md) | 7 | The module docstring of `facts.py` says that every score moves by `z`. The paper records that this is false. |
| 4 | [guarantee-strength](guarantee-strength.md) | 2 | The docstrings of `Counterfactual` and `compute_local_facts` describe guarantees and model kinds that the code does not have. |
| 5 | [model-consistency](model-consistency.md) | 5 | The checks against `ebm.predict` use a few fixed rows of two fitted models. |
| 6 | [completion-sampling](completion-sampling.md) | 6 | A test asserts that the share grows, and no test compares the estimate with an exact count. |
| 7 | [omitted-terms](omitted-terms.md) | 4 | `describe_ebm` leaves out interaction terms with only a log warning. The summary prompt does not know about them. |

The order puts the cheap fixes of wrong claims first. Plans 3 and 5 can share one Hypothesis strategy of small models,
so do 3 first and reuse its strategy in 5.
