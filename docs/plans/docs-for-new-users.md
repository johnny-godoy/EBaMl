# Plan: documentation for new users (0.12.0)

You are a senior technical writer and developer advocate for open-source machine learning libraries. You will
implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in `.claude/rules/`
first, with special care for `.claude/rules/writing-style.md` and `.claude/rules/semantic-contracts.md`, because the
project's hooks enforce those rules and `prek run --all-files` must pass before you finish.

<goal>
A reader who has never heard of an Explainable Boosting Machine understands, in one minute on the README or the home
page of the site, what an EBM is, why a team would want one, what EBaMl adds, and how to try it. Today the README is
561 lines long and opens with "BAML for Explainable Boosting Machines" and a comparison with TalkToEBM. It speaks only
to a reader who already knows both, and it mixes the pitch, the setup, the reference and the contributor guide. The
README is also the PyPI page, so it is the first page most people see.
</goal>

<context>
- `README.md` holds the intro, "Supported models", the full setup with six steps, usage, errors, "How it works", the
  notebook, the MCP server, the benchmark, "Choosing the model" and "Development".
- The site (`docs/`, Zensical, `zensical.toml`) includes README sections through `--8<--` markers: `docs/index.md` takes
  `README.md:intro` and `README.md:supported-models`, and other pages take more. A marker that you remove breaks a page.
- The `python` blocks of `README.md` and `docs/*.md` run as tests (Sybil, `tests/docs/conftest.py`). Every example that
  you write must run there, with the fake LLM replies of that file.
- Examples import public names from `ebaml` only. `tests/unit/docs/test_public_imports.py` fails on a deep import.
- Contract 3 of `semantic-contracts.md` holds for marketing text too: "exact", "guaranteed", "optimal" and "minimal" need
  the proofs in `proofs/`. A pitch that claims more than the code is a bug.
- The audiences: a data scientist who knows gradient boosting but not EBMs; a person in a regulated field (credit,
  health care, insurance) who must explain decisions; a developer who builds an agent and wants the MCP server; and the
  reader who already knows EBMs and wants the API.
</context>

<steps>
1. **Measure first.** Ask two subagents with no context of the repository to read only the current README, and to
   answer: What is an EBM? Why use one instead of XGBoost? What does EBaMl do that `interpret` does not? What do I run
   first? Keep their answers. Repeat the test on the new pages at the end, and put both sets of answers in the PR
   description. A plan for readers needs a reader.
2. **Design the information structure.** Write the plan of pages in the PR description before you move text. Use this
   order, and change it only with a reason:
   - The README: one paragraph that says what EBaMl does for the reader, one image, a quick start of at most ten lines
     of code, the "Supported models" table, and links to the site. Aim for at most 150 lines. Every section that a
     reader needs only after the first try moves to the site.
   - A page "What is an EBM?" for the reader who does not know: a glass-box model that adds one learned graph for each
     feature, why that makes each prediction readable, and where accuracy comes close to boosted trees. Cite the
     InterpretML paper and the documentation of `interpret` for each claim about accuracy. Do not invent numbers.
   - A page "Why EBaMl?" that shows the gap: a graph as a picture, the same graph as the text that `describe_graph`
     writes, and one local explanation of a single prediction. Mark each LLM output as an example of one run.
   - A home page (`docs/index.md`) that sends each audience to its first page. If Zensical supports grid cards (check
     the `attr_list` and `md_in_html` extensions first), use one card for each audience.
   - The setup, usage, providers, MCP and development pages keep the reference content that leaves the README.
3. **Make the images in the build.** Generate each graph image from a fitted EBM in a script or notebook that the docs
   build runs, so that an image never drifts from the code. Give each image alt text that states what the graph shows.
   Commit an image only if the build cannot make it, and then commit the script that made it next to it.
4. **Move the reference text.** Move the README sections to the site with their `--8<--` markers, or invert the
   direction: the site page holds the text, and the README links to it. Pick one direction for the whole repository,
   and say why in the PR description. Keep every Sybil-tested block a tested block after the move.
5. **Fix the links for PyPI.** PyPI shows the README without the repository, so a relative link or image breaks there.
   Use absolute links to the site and absolute image addresses in the README. Add a contract test that fails on a
   relative link or image in `README.md`.
6. **Write in the house style.** Follow ASD-STE100 as `writing-style.md` describes it: short sentences, one term for one
   meaning (`graph`, not `plot`), and the present tense. Define "EBM", "glass-box model", "shape function" and "local
   explanation" the first time each appears, and link each later use to the definition.
7. **Check the claims.** For each sentence that claims a property (exact, guaranteed, minimal, accurate), name the proof,
   the test or the citation in the PR description. Remove the word if you find none.
8. **Guard the length.** Add a test that fails when the README grows past the limit that you chose in step 2, with the
   limit and its reason next to it. A README grows one section at a time, and nobody notices.
9. **Update the agent context.** If the reorganization moves a section that `CLAUDE.md` or a rule names, update the
   reference. Keep `CLAUDE.md` short.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.12.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment says under `Changed` that the README is a short introduction and that the site holds the guides, with the new
pages "What is an EBM?" and "Why EBaMl?". Mark the 0.12.0 row of `ROADMAP.md` as done.
</release>

<done>
- A reader with no context answers the four questions of step 1 correctly from the README alone, and the PR shows the
  answers before and after.
- The README is at most the chosen length, and a test guards it. Its links and images work on PyPI.
- Every claim of a property has a proof, a test or a citation. Every code block runs under Sybil.
- `prek run --all-files`, `uv run pytest README.md docs src --no-cov` and the strict docs build pass.
</done>
