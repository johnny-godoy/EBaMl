# Proofs

This directory holds the Lean 4 proofs of the exact mathematical claims of EBaMl, and one paper for each topic. A
claim in the code or the docs that says "optimal", "minimal" or "exact" must have a proof here. The rules for this
directory are in `.claude/rules/proofs.md`.

## Topics

| Topic | Claim | Lean | Paper |
| --- | --- | --- | --- |
| `LocalExplanations` | The sorting rules of `src/ebaml/local/facts.py` solve the minimum change required and the minimum sufficient reason exactly. | `EBaMlProofs/LocalExplanations/` | `papers/local-explanations/paper.typ` |

## Layout

- `lakefile.toml`, `lean-toolchain` and `lake-manifest.json` pin one Lean and Mathlib release for all topics.
- `EBaMlProofs/<Topic>/` holds the Lean files of a topic. `Greedy.lean` is abstract, `Model.lean` is the model of the
  code, and `Axioms.lean` prints the axioms of the main theorems.
- `EBaMlProofs.lean` imports every topic file.
- `papers/style.typ` is the Typst style that all papers share. `papers/<topic>/paper.typ` is the paper of a topic.

CI does not run this check. Run it by hand with the commands below.

## Check the proofs

Install [elan](https://github.com/leanprover/elan), the Lean version manager. Then run these commands from a clean
clone:

```sh
cd proofs
lake exe cache get   # download the compiled Mathlib
lake build           # check every proof of every topic
```

The build must end with `Build completed successfully` and no line that starts with `error`. Each `#print axioms` line
must list only `propext`, `Classical.choice` and `Quot.sound`. If the cache server is not reachable, `lake build`
compiles the imported part of Mathlib from source. This takes tens of minutes.

## Build a paper

Install [Typst](https://github.com/typst/typst) 0.15 or later. Then run, for example:

```sh
cd proofs
typst compile --root papers papers/local-explanations/paper.typ   # writes paper.pdf next to the source
```

The appendix of each paper has the versions, the expected output and a table from each statement to its Lean theorem.

## Add a topic

1. Add `EBaMlProofs/<Topic>/` with the Lean files, and one `import` for each file in `EBaMlProofs.lean`.
2. Add `papers/<topic>/paper.typ` and `references.yml`. Start with `#import "/style.typ": *`.
3. Add a row to the table above.
