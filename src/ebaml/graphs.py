"""Graphs for EBM models."""

from typing import Self

import attrs
import numpy as np
from interpret.glassbox._ebm._utils import convert_to_intervals  # ruff: ignore[import-private-name] -- interpret has no public convert_to_intervals

from ebaml.custom_types import ExplainableBoostingModel, FeatureType, Normalization, normalize_feature_type
from ebaml.exceptions import InvalidOptionError
from ebaml.graph_text import graph_to_text, parse_graph_text


@attrs.define
class EBMGraph:
    """A datastructure for the graphs of Explainable Boosting Machines."""

    feature_name: str = attrs.field()
    feature_type: str = attrs.field()
    # x_vals does not admit categorical features yet
    x_vals: list[tuple[float, float]] = attrs.field()
    scores: list[float] = attrs.field()
    stds: list[float] = attrs.field()

    @classmethod
    def from_ebm(
        cls,
        ebm: ExplainableBoostingModel,
        feature_index: int,
        normalization: Normalization = Normalization.NONE,
        *,
        use_feature_bounds: bool = True,
    ) -> Self:
        """Extract the graph of a feature from an Explainable Boosting Machine.

        This is a low-level function. It does not return the final format in which the graph is presented to the LLM.

        The purpose of this function is to extract the graph from the interals of the EBM and return it in a format
        that is easy to work with.

        Parameters
        ----------
        ebm: ExplainableBoostingModel
            The EBM model.
        feature_index: int
            The index of the feature in the EBM.
        normalization: Normalization, default=Normalization.NONE
            How to normalize the graph (shift on the y-axis).
        use_feature_bounds: bool, default=True
            If True, the first and last x-axis bins are the min and max values of the feature stored in the EBM.
            If false, the first and last value are -inf and inf.

        Returns
        -------
        EBMGraph
            The graph.

        Raises
        ------
        ValueError
            If the feature type is not continuous or nominal.
        InvalidOptionError
            If the normalization option is not valid.
        InvalidOptionError.from_enum
            If the normalization option is not valid.

        """
        # read the variables from the ebm
        feature_name = ebm.feature_names_in_[feature_index]
        feature_type = ebm.feature_types_in_[feature_index]
        scores = ebm.term_scores_[feature_index][1:-1].copy()  # Drop missing and unknown bins
        stds = ebm.standard_deviations_[feature_index][1:-1]

        # normalize the graph
        match normalization:
            case Normalization.Z_SCORE:
                normalization_constant = np.mean(scores)
            case Normalization.MIN_MAX:
                normalization_constant = np.min(scores)
            case Normalization.NONE:
                normalization_constant = 0
            case _:
                raise InvalidOptionError.from_enum(str(normalization), Normalization)
        scores -= normalization_constant

        # read the x-axis bins from the ebm
        match feature_type:
            case "continuous":
                x_vals = convert_to_intervals(ebm.bins_[feature_index][0])
                # feature bounds apply to continuous features only
                if use_feature_bounds:
                    x_vals[0] = (ebm.feature_bounds_[feature_index][0], x_vals[0][1])
                    x_vals[-1] = (x_vals[-1][0], ebm.feature_bounds_[feature_index][1])
            case "nominal":
                # the bins map each category to its (1-based, after the missing bin) position
                positions = {position - 1: category for category, position in ebm.bins_[feature_index][0].items()}
                x_vals = [positions[idx] for idx in range(len(positions))]
            case _:
                raise InvalidOptionError(feature_type, ["continuous", "nominal"])
        if len(x_vals) != len(scores):
            msg = (
                f"The number of bins ({len(x_vals)}) and scores ({len(scores)}) does not match"
                f" for feature {feature_name} of type {feature_type}."
            )
            raise ValueError(
                msg,
            )
        return EBMGraph(feature_name, feature_type, x_vals, scores, stds)

    def simplify(self, min_variation_per_cent: float = 0.0) -> Self:
        """Simplifies the graph. Removes redundant (flat) bins from the graph.

        Parameters
        ----------
        min_variation_per_cent: float, default=0.0
            Parameter that controlls the degree of simplification.
            If min_variation_per_cent>0, the function simplifies the graph by removing bins that correspond to a less
            that min_variation_per_cent change in the score, considering the overal min/max difference of score for the
            feature as 100%.

        Returns
        -------
        EBMGraph:
            The simplified graph.

        Raises
        ------
        InvalidOptionError
            If the feature type is not continuous.
        ValueError
            If min_variation_per_cent is not between 0 and 1.

        """
        match normalize_feature_type(self.feature_type):
            case FeatureType.CONTINUOUS:
                pass
            case _:
                raise InvalidOptionError(self.feature_type, ["continuous"])
        if not 0 <= min_variation_per_cent <= 1:
            msg = f"min_variation_per_cent must be between 0 and 1. Got {min_variation_per_cent}."
            raise ValueError(msg)
        x_vals, scores, stds = self.x_vals, self.scores, self.stds
        total_variation = np.max(scores) - np.min(scores)
        new_x_vals, new_scores, new_stds = [x_vals[0]], [scores[0]], [stds[0]]
        for idx in range(
            1,
            len(x_vals),
        ):  # pragma: no mutate -- comparing the first bin to itself changes nothing, so this mutant is equivalent
            b0, b1 = x_vals[idx]
            if np.abs(float(new_scores[-1]) - float(scores[idx])) <= total_variation * min_variation_per_cent:
                # extend the previous bin to b1
                new_x_vals[-1] = (new_x_vals[-1][0], b1)
                # guarantee that the the confidence bands of the simplified graph cover the original graph as well as
                # its confidence bands
                new_stds[-1] = max(new_stds[-1], stds[idx])
            else:
                new_x_vals.append((b0, b1))
                new_scores.append(scores[idx])
                new_stds.append(stds[idx])
        return EBMGraph(
            self.feature_name,
            self.feature_type,
            new_x_vals,
            new_scores,
            new_stds,
        )

    to_text = graph_to_text

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Parse the textual representation of a graph back into a graph. The inverse of ``to_text``.

        See ``ebaml.graph_text.parse_graph_text`` for the details of the format.

        Parameters
        ----------
        text: str
            The text of the graph, with the confidence bounds included.

        Returns
        -------
        EBMGraph
            The graph.

        """
        return cls(*parse_graph_text(text))
