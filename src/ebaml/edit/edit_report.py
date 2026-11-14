"""The report of an edit: what changed, how many training rows it reaches, and what it leaves out.

The report holds numbers and the names of the bins. The text claims no more than the numbers show, so it never says
that an edit is safer or better.
"""

import attrs
import numpy as np

from ebaml.core.custom_types import ExplainableBoostingModel
from ebaml.core.exceptions import EBaMlError
from ebaml.edit.edits import EditOutcome
from ebaml.edit.model_view import interaction_names, term_view
from ebaml.graphs.ebm_graph import EBMGraph

_TEXT_CHANGE_LIMIT = 10


@attrs.frozen
class BinChange:
    """The score of one bin before and after an edit."""

    position: int
    label: str
    before: float
    after: float


@attrs.frozen
class EditReport:
    """What an edit changes in a model.

    ``graph_before`` and ``graph_after`` hold the graph text without the preamble. They are ``None`` for an edit that
    removes terms, and for a feature that the graph text does not cover (an ordinal feature). The missing bin and the
    unseen bin are not in the graph text, so ``changes`` lists them too.

    ``weight_share`` is the share of the training weight that the changed bins hold: the number of training rows that
    the edit reaches, divided by the number of training rows. It needs no data. A ``bins_changed`` of 0 means that
    the edit set every selected score to the value that it had.

    ``base_revision`` is the revision that the edit started from. ``revision`` is the revision that ``apply`` made, and
    it is ``None`` for a ``preview``, which makes none.
    """

    feature: str | None
    edit_type: str
    description: str
    changes: tuple[BinChange, ...]
    largest_change: float
    covered_range: str | None
    weight_share: float
    weighted_mean_shift: float
    intercept_change: float
    interactions_still_using: tuple[str, ...]
    removed_terms: tuple[str, ...]
    graph_before: str | None
    graph_after: str | None
    base_revision: str
    revision: str | None = None

    @property
    def bins_changed(self) -> int:
        """The number of bins whose score changed."""
        return len(self.changes)

    def to_text(self) -> str:
        """Write the report as plain text.

        Returns
        -------
        str
            The description, the changed bins, the training weight that they hold, and the pair terms that still use
            the feature.

        """
        lines = [self.description + "."]
        if self.removed_terms:
            lines.append(f"Removed terms: {', '.join(self.removed_terms)}.")
        else:
            noun = "bin" if self.bins_changed == 1 else "bins"
            lines.append(
                f"{self.bins_changed} {noun} changed. The largest change of a score is {self.largest_change:.4g}.",
            )
            lines.extend(
                f"  {change.label}: {change.before:.4g} to {change.after:.4g}"
                for change in self.changes[:_TEXT_CHANGE_LIMIT]
            )
            if self.bins_changed > _TEXT_CHANGE_LIMIT:
                lines.append(f"  and {self.bins_changed - _TEXT_CHANGE_LIMIT} more bins")
            lines.append(
                f"The changed bins hold {self.weight_share:.1%} of the training weight."
                f" The weighted mean of the graph moves by {self.weighted_mean_shift:.4g}"
                f" and the intercept by {self.intercept_change:.4g}.",
            )
        if self.interactions_still_using:
            names = ", ".join(self.interactions_still_using)
            lines.append(f"The pair terms {names} still use the feature, and the edit does not change them.")
        return "\n".join(lines)


def _graph_text(ebm: ExplainableBoostingModel, feature: str) -> str | None:
    try:
        feature_index = list(ebm.feature_names_in_).index(feature)
        return EBMGraph.from_ebm(ebm, feature_index).to_text(include_description=False)
    except EBaMlError:
        # the graph text covers continuous and nominal features, and an ordinal feature raises
        return None


def _intercept(ebm: ExplainableBoostingModel) -> float:
    return float(np.ravel(ebm.intercept_)[0])


def build_report(
    before: ExplainableBoostingModel,
    after: ExplainableBoostingModel,
    edit_type: str,
    outcome: EditOutcome,
    base_revision: str,
) -> EditReport:
    """Compare a model before and after an edit.

    Parameters
    ----------
    before : ExplainableBoostingModel
        The model that the edit started from.
    after : ExplainableBoostingModel
        The model that the edit made.
    edit_type : str
        The name of the class of the edit.
    outcome : EditOutcome
        What the edit reported.
    base_revision : str
        The revision that ``before`` is, as ``EditSession.revision_info`` names it.

    Returns
    -------
    EditReport
        The report.

    """
    if outcome.removed_terms or outcome.feature is None:
        return EditReport(
            feature=outcome.feature,
            edit_type=edit_type,
            description=outcome.description,
            changes=(),
            largest_change=0.0,
            covered_range=None,
            weight_share=0.0,
            weighted_mean_shift=0.0,
            intercept_change=_intercept(after) - _intercept(before),
            interactions_still_using=(),
            removed_terms=outcome.removed_terms,
            graph_before=None,
            graph_after=None,
            base_revision=base_revision,
        )
    view_before = term_view(before, outcome.feature)
    view_after = term_view(after, outcome.feature)
    scores_before, scores_after = view_before.scores, view_after.scores
    weights = view_before.weights
    changed = np.flatnonzero(scores_after != scores_before)
    changes = tuple(
        BinChange(
            int(position),
            view_before.label(int(position)),
            float(scores_before[position]),
            float(scores_after[position]),
        )
        for position in changed
    )
    total_weight = float(weights.sum())
    mean_before = float(np.average(scores_before, weights=weights)) if total_weight > 0 else 0.0
    mean_after = float(np.average(scores_after, weights=weights)) if total_weight > 0 else 0.0
    return EditReport(
        feature=outcome.feature,
        edit_type=edit_type,
        description=outcome.description,
        changes=changes,
        largest_change=float(np.abs(scores_after - scores_before).max(initial=0.0)),
        covered_range=view_before.range_text(outcome.positions) if outcome.positions else None,
        weight_share=float(weights[changed].sum()) / total_weight if total_weight > 0 else 0.0,
        weighted_mean_shift=mean_after - mean_before,
        intercept_change=_intercept(after) - _intercept(before),
        interactions_still_using=interaction_names(after, outcome.feature),
        removed_terms=(),
        graph_before=_graph_text(before, outcome.feature),
        graph_after=_graph_text(after, outcome.feature),
        base_revision=base_revision,
    )
