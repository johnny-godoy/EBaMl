"""The strength of the claim that a set of changes or of inputs is the smallest one.

A counterfactual and a sufficient reason are sets, and the facts say how far the set is proved to be the smallest.
``MinimalityStatus`` names the reason, so that a result never claims more than its search established (contract 2 of
``.claude/rules/semantic-contracts.md``). The prompts read only ``MinimalityStatus.is_smallest``, because the reader of
an explanation needs to know whether the set is the smallest, and not how the code proved it.
"""

import enum


class MinimalityStatus(enum.StrEnum):
    """Why a set is, or is not, known to be the smallest.

    Members
    -------
    SORTING_PROOF
        A decision with one margin. Sorting the features finds the smallest set, as the proof paper shows.
    SOLVER_PROOF
        A covering problem. The solver proved that no smaller set is sufficient. The proof rests on the certificate of
        the solver, which Lean does not check.
    AT_LOWER_BOUND
        A covering problem. The set has as many members as the lower bound of the problem, so no smaller set exists.
        The bound rests on the LP solver, as the paper says.
    UNPROVED
        A covering problem. The search found a set, but it did not prove that the set is the smallest. A smaller set may
        exist.
    """

    SORTING_PROOF = "sorting_proof"
    SOLVER_PROOF = "solver_proof"
    AT_LOWER_BOUND = "at_lower_bound"
    UNPROVED = "unproved"

    @property
    def is_smallest(self) -> bool:
        """Tell whether no smaller set exists.

        Returns
        -------
        bool
            False only for ``UNPROVED``.

        """
        return self is not MinimalityStatus.UNPROVED
