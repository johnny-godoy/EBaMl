### Fixed

- The sufficient reason of a decision with several margins no longer comes back as "none" when a gain is rounding noise of zero (such as `2.8e-17`). The LP solver had called the problem infeasible, so the facts claimed that no set of inputs keeps the decision.
