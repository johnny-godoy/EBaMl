"""Edit the graphs of a fitted EBM, so that the model meets a business requirement.

The package follows the edit tools of GAM Changer (Wang et al., KDD 2022, MIT licence) that keep the bins. An edit
changes the scores of the bins of one main effect. ``EditSession`` applies edits to a copy of the model, keeps a
history with a reason for each edit, and saves the edited model. The code calls no LLM.
"""
