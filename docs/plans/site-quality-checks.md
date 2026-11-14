# Plan: quality checks of the documentation site (0.13.0)

You are a senior front-end engineer who specializes in web accessibility (WCAG 2.2) and in automated tests of static
sites. You will implement one milestone of `ROADMAP.md` in the EBaMl repository. Read `CLAUDE.md` and every file in
`.claude/rules/` first, with special care for `.claude/rules/testing.md`, because the project's hooks enforce those
rules and `prek run --all-files` must pass before you finish.

<goal>
The built documentation site passes automated checks of accessibility, links, HTML, performance and layout on every
pull request that changes it, the way the Python code passes its gates. Today the docs build checks only that the
Markdown builds without a griffe warning and that each code example runs. Nothing checks that a person who uses a
screen reader, a keyboard, a phone or the dark theme can use the site, or that a link still leads somewhere.
</goal>

<context>
- `uv run --group docs python -m scripts.build_docs` builds the site strictly. It exports each marimo notebook to HTML
  and compiles each proof paper to a PDF. `.github/workflows/docs.yml` runs the build.
- The site has a light theme, a dark theme and a theme that follows the system (`zensical.toml`, `[[project.theme.palette]]`).
- The notebook pages and the proof pages embed an `<iframe>`. The content of a marimo export and of a PDF is not ours,
  but the page that embeds it is.
- 0.12.0 rewrote the home page, the README and the guides. This milestone checks the result.
- Chromium and Playwright for Python are in the cloud environment. CI must install them. Node.js is in CI already,
  because the `jscpd` hook needs it.
- `testing.md` gives each test suite one question. A check of the built site answers a new question, so it is a new
  suite with its own marker, and `scripts/check_test_suites.py` must know it.
</context>

<steps>
1. **Add the suite.** Add `tests/site/` with the marker `site`. Its tests take the built site from one session fixture:
   build once, serve the directory on a local port, and give each test the base address. Register the suite in
   `testing.md`, `tests/conftest.py` and `scripts/check_test_suites.py`. Run it in `docs.yml` after the build, as a
   required check. Update `.github/rulesets/main.json` and the `GATES` of `.github/scripts/ci_report.py`.
2. **Check accessibility with axe.** Load each page of the site, found from the sitemap or the nav and not from a hand
   list, in Playwright. Inject axe-core and fail on each violation of the WCAG 2.2 A and AA rules. Run each page in the
   light and the dark color scheme (`emulate_media(color_scheme=...)`), because the contrast of the two themes differs.
   Pin the axe-core version. A Python wrapper is acceptable if it is maintained, and else load the pinned script.
3. **Check what axe cannot.** Write tests for these, because an automated rule engine misses them:
   - The keyboard reaches the search, the nav, the theme toggle and each copy button, in a visible order, with a
     visible focus ring.
   - At a phone width of 360 pixels, no page scrolls sideways, except inside a code block or a table that scrolls by
     itself.
   - Each page has one `h1`, and its headings do not skip a level.
   - Each image has alt text that is not the file name, and each `iframe` has a `title`. Each embedded PDF has a link to
     the file next to it, for readers whose browser does not show the frame.
4. **Check the links.** Check every internal link and anchor on each pull request, offline. Check the external links on
   a weekly schedule only, because a remote site can fail for a day and must not block a merge. Use lychee or a small
   checker in the suite, and keep a short list of known flaky hosts with a reason for each.
5. **Validate the HTML.** Run an HTML validator (`html-validate` or the Nu validator) on the pages that the project
   writes. Exclude the marimo exports and the theme assets with one `# reason:` entry, not one ignore for each page.
6. **Measure performance and SEO.** Run Lighthouse CI on the home page, one guide, one notebook page and the API
   reference. Set budgets for accessibility, best practices and SEO at 100 unless a finding comes from the theme, and
   set a performance budget from the first measured run with a margin. Each page needs a `description` meta tag and a
   title that names the page.
7. **Fix what the checks find.** Fix each finding in our Markdown, `zensical.toml` or the build script. For a finding in
   the theme or in a marimo export that we cannot fix, add one ignore with the rule, the page and the reason
   (`prek-and-ignores.md`), and report it upstream if no report exists. Do not lower a rule to make the suite pass.
8. **Prove the guards.** For each check, break one page on purpose (remove an alt text, add a broken anchor, set a
   low-contrast color), watch the test fail, and restore the page (`bug-fixes.md`).
9. **Document the suite.** Add the run command to `CLAUDE.md` in one line, and a section to `CONTRIBUTING.md` that says
   how to run the suite locally and how to read an axe report.
</steps>

<release>
This PR completes the milestone: it sets `version = "0.13.0"` in `pyproject.toml` and runs `uv lock`. Its changelog
fragment says under `Fixed` which accessibility and link problems of the site the PR fixed. The new suite itself is
contributor tooling and needs no entry. Mark the 0.13.0 row of `ROADMAP.md` as done.
</release>

<done>
- The `site` suite runs in CI on every pull request that changes the docs, and it is a required check.
- Every page passes axe with WCAG 2.2 A and AA rules in both color schemes, and the keyboard, layout, heading and
  image checks pass.
- Every internal link works. A weekly job checks the external links.
- Lighthouse meets the budgets. Each remaining ignore names the rule, the page and the reason.
- `prek run --all-files`, `uv run pytest` and the strict docs build pass.
</done>
