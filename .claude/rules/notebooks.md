# Notebooks for features that need help

A feature with a multi-step setup, or one that a reader understands best by trying it, gets a marimo notebook in
`notebooks/`. A notebook shows the feature working, and it lets the maintainer review the code by running it. Text and
a unit test cannot do both.

## When to add one

- Add a notebook in the same change as a feature that has a cornerstone role in a `ROADMAP.md` milestone.
- Add one when the user must make a choice with several options (a provider, a model, a feature format), because a
  reader then sees the effect of each option side by side.
- Add one when the setup has more than two steps (keys, servers, extra packages), so the notebook walks through them.
- Do not add one for a small option or an internal change. A docstring example is enough.

## Content

1. Open with one cell that says what the reader learns and what they need (data, a key, a local server).
2. Run every step that needs no key first, and put each paid LLM call behind a button, as `local_explanations.py` does.
   The notebook then works offline up to the first call, and a reader never pays by accident.
3. Show one example for each option the feature offers. For model providers, that is one section per provider
   (OpenRouter, OpenAI, Anthropic, Ollama, a gateway with `base_url`), with the same call in each, so the reader
   compares them.
4. Show the failure and its fix steps for the setup errors a user can meet (a missing key, an unknown provider).
5. Use only the public API of `ebaml`. A notebook that needs a private name shows a gap in the API.
6. Write the prose by `writing-style.md` (ASD-STE100).

## Register the notebook

- Add the file to `NOTEBOOKS` in `scripts/build_docs.py`.
- Add a page `docs/<name>-notebook.md`, with the same parts as `docs/local-notebook.md`: the purpose, what needs a key,
  the run command, the link to the guide, and the iframe.
- Add the page to the nav in `zensical.toml`.
- Link the notebook from the guide of the feature.
- Run `uv run --group docs python -m scripts.build_docs`. The build exports each notebook, so a broken cell fails it.
