"""Deterministic local explanations of one EBM prediction.

The three explanations are the explainability queries of Barceló, Monet, Pérez and Subercaseaux, "Model
Interpretability through the Lens of Computational Complexity" (arXiv:2010.12265):

- *Minimum change required* (counterfactual): the fewest inputs to change so that the prediction flips.
- *Minimum sufficient reason*: the fewest inputs that fix the prediction whatever the other inputs are.
- *Count completions*: the share of completions of a partially known input that get the same prediction.

An EBM without interactions is a sum of one score per feature. The decision of a prediction is a margin: a sum of one
term per feature, positive while the decision holds (``ebaml.local.local_kinds``). The first two explanations are solved
exactly by sorting the features by how much they move the margin, and the third by sampling the unknown features from
their training distribution with a fixed seed. A decision with several margins, and the counterfactual of a chosen
class, are covering problems that sorting does not solve. A ``CoveringSolver`` of ``ebaml.local.covering_solvers``
solves them. Nothing here calls an LLM, and nothing here renders text: ``ebaml.local.fact_text`` puts the facts in words
for the prompts in ``baml_src/local_explanations.baml``.

The proof that the sorting algorithms are exact is in ``proofs/papers/local-explanations/paper.typ``. It covers a binary
classifier, a multiclass classifier and a regressor, which use the same algorithms on the margin of their decision. For
a covering problem, the paper states what a result of a solver proves and what it does not.

The confidence of the model enters through ``confidence_level``, as a number ``z`` of standard deviations. Each claim
names the scores that move by ``z`` deviations, always in the direction that makes the claim harder to prove:

- A counterfactual moves the score of the new bin of each changed feature against the flip. The features that do not
  change keep their fitted scores.
- A sufficient reason moves every score against the decision: a fixed feature at its bin, and a free feature at its
  worst plausible bin.
- The interval of the output moves the total score. Its deviation is that of a sum of independent errors.
- A margin of a multiclass classifier is the difference of two class scores. Its deviation is the sum of the two
  deviations, because an EBM stores no covariance between two classes.

A claim holds under its own model only. ``confidence_level=0`` gives ``z = 0``, so no score moves.
"""

import statistics
from collections.abc import Mapping

import numpy as np

from ebaml.core.custom_types import ExplainableBoostingModel, FeatureValue
from ebaml.core.exceptions import OptionRangeError, UnsupportedModelError
from ebaml.graphs.model_kinds import model_kind
from ebaml.local.completions import reveal_order, sample_completions
from ebaml.local.counterfactuals import Totals, choose_counterfactual
from ebaml.local.fact_features import read_features
from ebaml.local.fact_types import Contribution, LocalFacts, LocalOptions, Prediction
from ebaml.local.local_kinds import LocalKind, local_kind
from ebaml.local.sufficient_reasons import Bound, find_reason


def _local_kind(ebm: ExplainableBoostingModel) -> LocalKind:
    """Read the kind of an EBM for the local explanations.

    The model kind is the one place that reads ``link_`` and ``classes_``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The EBM.

    Returns
    -------
    LocalKind
        The class of the kind.

    Raises
    ------
    UnsupportedModelError
        If the model is not a classifier or a regressor, or its kind is unknown.

    """
    message = "Local explanations need a classifier or a regressor."
    try:
        return local_kind(model_kind(ebm))
    except UnsupportedModelError as error:
        if str(error).startswith(message):
            raise
        msg = f"{message} {error}"
        raise UnsupportedModelError(msg) from error


def compute_local_facts(
    ebm: ExplainableBoostingModel,
    instance: Mapping[str, FeatureValue],
    options: LocalOptions | None = None,
) -> LocalFacts:
    """Compute the three local explanations of one prediction of a classifier or a regressor.

    The classifier is binary or multiclass. The function raises an ``UnsupportedModelError`` for an EBM that is not a
    classifier or a regressor, and for an EBM with interactions. It raises an ``InvalidInstanceError`` for an instance
    that lacks a value or has an unknown category. A regressor explains the answer to ``LocalOptions.decision``.

    A counterfactual and a sufficient reason are the smallest sets only if their ``minimality`` says so. Sorting proves
    it for a decision with one margin. A decision with several margins and the counterfactual of a target class
    depend on ``LocalOptions.search``.

    Parameters
    ----------
    ebm: ExplainableBoostingModel
        The fitted EBM.
    instance: Mapping[str, FeatureValue]
        The value of every feature, by feature name.
    options: LocalOptions, optional
        Confidence level, sampling and support options.

    Returns
    -------
    LocalFacts
        The prediction, the contributions and the three explanations.

    Raises
    ------
    OptionRangeError
        If the confidence level is not in [0, 1).

    """
    options = options or LocalOptions()
    if not 0 <= options.confidence_level < 1:
        msg = f"confidence_level must be in [0, 1). Got {options.confidence_level}."
        raise OptionRangeError(msg)
    z = statistics.NormalDist().inv_cdf((1 + options.confidence_level) / 2)
    kind = _local_kind(ebm)
    features = read_features(ebm, instance, options.min_support)
    intercepts = np.ravel(ebm.intercept_).tolist()
    scores = (np.array(intercepts) + sum(f.outputs(f.bin_index) for f in features)).tolist()
    score_stds = np.sqrt(sum(f.output_stds(f.bin_index) ** 2 for f in features)).tolist()
    frame = kind.frame(
        scores,
        score_stds,
        intercepts=intercepts,
        confidence_level=options.confidence_level,
        decision=options.decision,
    )
    margins = frame.side.margins
    binding = frame.side.binding(scores)
    # every algorithm sees the margin sign * (intercept - threshold + the scores), which is positive while it holds
    bounds = [Bound(margin.offset(intercepts), margin.sign, margin.columns) for margin in margins]
    prediction = Prediction(
        decision=frame.side.label,
        output=frame.output,
        lower=frame.lower,
        upper=frame.upper,
        score=scores[frame.column],
        score_std=score_stds[frame.column],
        margin=binding.value(scores),
        threshold=binding.limit(scores),
        output_kind=kind.output_kind,
    )
    order = reveal_order(features, bounds, z)
    counterfactual = choose_counterfactual(
        features,
        frame,
        Totals(scores, intercepts, z),
        kind=kind,
        options=options,
    )
    return LocalFacts(
        prediction=prediction,
        contributions=sorted(
            (
                Contribution(
                    f.name,
                    f.label,
                    float(f.outputs(f.bin_index)[frame.column]),
                    float(f.output_stds(f.bin_index)[frame.column]),
                )
                for f in features
            ),
            key=lambda c: abs(c.score),
            reverse=True,
        ),
        counterfactual=counterfactual,
        sufficient_reason=find_reason(features, bounds, z, options),
        completions=sample_completions(features, order, bounds, options),
        confidence_level=options.confidence_level,
        min_support=options.min_support,
        n_samples=options.n_samples,
    )
