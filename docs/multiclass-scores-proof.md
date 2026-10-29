# Proof of the multiclass scores

The graph of a multiclass classifier has one series of scores for each class. The text of the graph and the prompt
say that only the differences between the classes of a bin change the predicted probabilities. This page states what
a proof shows, and where to read it.

<p><a href="proofs/multiclass-scores.pdf">Read the paper (PDF)</a></p>

## What is proved

The proof covers a multiclass EBM: the score of each class is the intercept plus one score for each term, and softmax
turns the scores into probabilities. A term is a feature or an interaction.

- **Only the differences count.** Two score vectors give the same probabilities if and only if they differ by one
  constant for all the classes.
- **Same mean.** In every bin of every graph that `EBMGraph.from_ebm` reads, the scores have the same mean over all
  the classes of the model. The mean is zero when the normalization is `NONE`.
- **Same differences.** In every bin, the graph keeps the difference between the scores of two classes.
- **Same predictions.** A model with the scores of the graphs predicts the same probabilities as the EBM.

The last three claims come from one result: if you lower the scores of all the classes of a bin by one number, the
predictions do not change. The centring and the normalization of `EBMGraph.from_ebm` both lower the scores in this way.

## What is not proved

- Floating point. The proof works over the real numbers.
- The standard deviations. The graph keeps the standard deviations of the EBM, and the claims say nothing about them.
- The prediction of `interpret`. The proof assumes that the EBM predicts the softmax of its scores. It does not check
  the code of `interpret`.
- The text. The LLM reads rounded scores. The proof is about the numbers before the rounding.
