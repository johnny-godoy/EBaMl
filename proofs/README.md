# Proofs

This directory holds the Lean 4 proof that the sorting rules of `src/ebaml/local/facts.py` are optimal, and the paper
that presents it. The paper is `paper.typ`. It states each claim, gives the proof in numbered steps, and names the
Lean theorem for each statement.

The Lean library `EBaMlProofs` has three files in `EBaMlProofs/`:

- `Greedy.lean` holds the abstract result. It knows nothing about EBMs.
- `Model.lean` holds the EBM model and the proofs of the two optimality theorems.
- `Axioms.lean` prints the axioms of the main theorems.

CI does not run this check. Run it by hand with the commands below.

## Check the proof

Install [elan](https://github.com/leanprover/elan), the Lean version manager. Then run these commands from a clean
clone:

```sh
cd proofs
lake exe cache get   # download the compiled Mathlib
lake build           # check every proof
```

The build must end with `Build completed successfully` and no line that starts with `error`. The six `#print axioms`
lines must list only `propext`, `Classical.choice` and `Quot.sound`. If the cache server is not reachable,
`lake build` compiles the imported part of Mathlib from source. This takes tens of minutes.

## Build the paper

Install [Typst](https://github.com/typst/typst) 0.15 or later. Then run:

```sh
cd proofs
typst compile paper.typ   # writes paper.pdf
```

The appendix of the paper has the versions, the expected output and a table from each statement to its Lean theorem.
