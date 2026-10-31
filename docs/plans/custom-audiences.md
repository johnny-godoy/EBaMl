# Plan: custom audiences (0.3.0)

You are a senior Python engineer who designs extensible APIs. You implement one milestone of `ROADMAP.md` in the EBaMl
repository. Read `CLAUDE.md` and every file in `.claude/rules/` first, because the project's hooks enforce those rules
and `prek run --all-files` must pass before you finish.

<goal>
Make the audience of a local explanation an open extension point. The package ships two audiences, `Layperson` and
`DataScientist`. A user writes an audience with its own rules by subclassing `Audience`, and passes an object of it as
`audience=`. Adding an audience needs no edit in `src/ebaml/` and no edit in `baml_src/`.
</goal>

<context>
- An enum with a private table of renderers and the instruction text in BAML needs an edit in three places for each
  new audience. A base class with one subclass for each audience needs none.
- An audience has two jobs: it decides which keys the facts of the prompt get (`facts_to_texts`), and it tells the LLM
  how to write (`guidance`). One class holds both, because the two must agree: a layperson prompt that forbids numbers
  works only if the facts hold no number.
- SOLID: O (a new audience is a new class), L (a subclass keeps the contract of `Audience`), D (the BAML code receives
  the guidance as text and does not know the classes).
</context>

<steps>
1. Add `ebaml.local.audience` with the abstract class `Audience`, the built-in `Layperson` and `DataScientist`, the
   helpers that they share (`likelihood_word`, `strength_word`, `round_value`, `points_towards`) and
   `resolve_audience`, which maps a name to a built-in audience.
2. Make all seven methods abstract. The reason: an author must decide on every model number, so no audience leaks a
   number by omission. The short way to a new audience is a subclass of a built-in one.
3. Let `facts_to_texts` and `explain_instance` accept an `Audience` or the name of a built-in audience.
4. Change the BAML local functions to take `audience_guidance: string`. Keep the guidance of the built-in audiences in
   Python, and copy it into `baml_src/prompt_tests.baml` for the live tests. A unit test checks that the copies match.
5. Export `Audience`, `Layperson` and `DataScientist` from `ebaml`.
6. Write the notebook `notebooks/audiences.py` with three audiences (a subclass of `Layperson`, a subclass of
   `DataScientist`, a subclass of `Audience`), the facts of each side by side, the two setup errors, and the LLM calls
   behind buttons. Register it as `.claude/rules/notebooks.md` says.
7. Document the feature in `docs/local-explanations.md` and add a changelog fragment that marks the removal of the enum
   as **BREAKING**.
</steps>

<release>
The PR that completes this milestone sets `version = "0.3.0"` in `pyproject.toml` and runs `uv lock`.
</release>
