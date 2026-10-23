# Writing style

Two audiences need two styles. People read documentation, so it follows Simplified Technical English. Models read
prompts, so they follow Anthropic's prompting guide. Choose the style by the reader, not by the file type.

## Documentation: 80% of the way to ASD-STE100

Scope: `README.md`, `docs/`, `benchmark/README.md`, `scripts/README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, docstrings,
code comments and notebook prose. Exempt: code, command output, quoted error text and proper names.

Write "80% of the way to ASD-STE100". The full specification is very strict, so follow its spirit: plain, short,
unambiguous sentences. Models know the specification well, so no tool or count measures compliance. Use the rules below
as a guide, and break one when a strict version would make the text harder to read.

1. Write one instruction per sentence, in the imperative. Procedure sentences have at most 20 words.
2. Descriptive sentences have at most 25 words. Put one idea in each.
3. Paragraphs have at most 6 sentences.
4. Use the active voice and the simple present tense. Use the future tense only for events that have not happened.
5. Keep articles (`a`, `the`) and write full sentences. Do not use fragments or telegraphic style.
6. Use one word for one meaning, and use it every time. Do not vary terms for style (`graph`, not `plot` or `chart`).
7. Use the simplest word (`use`, not `utilize`; `start`, not `initiate`; `check`, not `verify` when `check` is exact).
8. Do not use a noun as a verb, or a verb as a noun, when a common word exists. Avoid phrasal verbs with several senses.
9. Do not use the `-ing` form as a verb in a procedure. Do not use idioms, jokes or figures of speech.
10. Put a warning before the step it applies to. Put the condition before the instruction (`If X, do Y`).
11. Use a list for steps or items that are really parallel. Use a table for a lookup. Use prose for explanation.
12. Do not use noun strings of more than three nouns. Add a preposition or split the phrase.

The STE dictionary at <https://www.asd-ste100.org> is the reference for approved words and meanings. A technical name
(`describe_ebm`, `OPENROUTER_API_KEY`) is always allowed.

## LLM prompts: Anthropic prompting guide

Scope: every prompt string in `baml_src/*.baml`, and `CLAUDE.md`. Reference:
<https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview>, which sends you to
<https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>.

Prompts are not STE. A model gains from context and examples that STE would remove. Apply these rules instead:

- **Be clear and direct.** Say exactly what output you want, with its format and limits. Pass the colleague test:
  if a person with no context would be confused by the prompt, the model will be too.
- **Give the reason.** A rule with its motive generalizes better than a bare command. Write "the reader is a layperson,
  so give no model numbers", not "NEVER give numbers".
- **Say what to do, not what to avoid.** Write "answer in plain prose paragraphs", not "do not use markdown".
- **Use XML tags** with consistent, descriptive names to separate instructions, context, examples and inputs
  (`<instructions>`, `<graph>`, `<example>`). Nest tags when the content has a hierarchy.
- **Use examples** that are relevant, diverse and wrapped in `<example>` tags inside `<examples>`. Use 3 to 5 when the
  format matters.
- **Set a role** in one sentence at the start of the system prompt.
- **Put long data first** and the question or task last. For a long graph or document, ask the model to quote the
  relevant parts before it answers.
- **Match the prompt's formatting to the output you want.** A prompt full of markdown produces markdown.
- **Order steps** as a numbered list when order matters. Ask for reasoning before the final answer when accuracy
  matters.
- **Test the change.** Compare the new prompt with the old one on the offline BAML tests and, for the benchmark prompts,
  on the benchmark. Do not change a prompt on taste alone.

`CLAUDE.md` is a prompt too: give each rule with its reason, state it positively, and group content under clear
headings or tags. Keep it short, because every line costs context in every session.

## Checks

- After you edit a prose file, reread it once against the documentation list above, or the prompt list for a prompt.
- A prompt change in `baml_src/` needs `baml generate`, `baml fmt`, `baml check` and `OPENROUTER_API_KEY=unused baml test -x root::Live`. Also run the live prompt tests, `baml test -i root::Live`, with a real key.
- A user-visible wording change in a prompt needs a `CHANGELOG.md` entry under `[Unreleased]`.
